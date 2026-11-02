"""D1 验收测试：手写采样器 vs 仓库 MiniMindForCausalLM.generate

运行全部：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_sampling.py

只跑某几项（名字子串匹配，大小写不敏感）：
    python mymm\\tests\\test_sampling.py top_p parity

失败时会打印「期望值 / 实际值 / 首个不一致位置」，按提示改 sampling.py 即可。
实现前全部 FAIL（NotImplementedError）属正常。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

from mymm.sampling import (  # noqa: E402
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    generate,
    pick_next_token,
)

WEIGHT = REPO / "out" / "full_sft_768.pth"
TOKENIZER_DIR = REPO / "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEG_INF = float("-inf")

_STATE: dict = {}


# ============================================================================
# 基础设施
# ============================================================================
def get_model():
    if "model" not in _STATE:
        from transformers import AutoTokenizer
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        assert WEIGHT.exists(), f"缺少权重 {WEIGHT}"
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
        model.load_state_dict(torch.load(str(WEIGHT), map_location="cpu", weights_only=True), strict=True)
        model = model.eval()
        if DEVICE == "cuda":
            model = model.half()
        _STATE["tok"] = tok
        _STATE["model"] = model.to(DEVICE)
    return _STATE["model"], _STATE["tok"]


def make_ids() -> torch.Tensor:
    _, tok = get_model()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "用一句话介绍你自己"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok(text, return_tensors="pt").input_ids.to(DEVICE)


def fmt(x: float) -> str:
    if x == NEG_INF:
        return "-inf"
    return f"{x:.6g}"


def show_diff(label: str, got, want, atol: float = 1e-6):
    """数值不对时才打印 got vs want 的形状、最大误差、前若干个不一致位置。"""
    g = got.detach().to(torch.float32).cpu() if torch.is_tensor(got) else torch.tensor(got, dtype=torch.float32)
    w = want.detach().to(torch.float32).cpu() if torch.is_tensor(want) else torch.tensor(want, dtype=torch.float32)

    same_shape = g.shape == w.shape
    maxd = float("nan")
    if same_shape:
        diff = (g - w).abs()
        finite = torch.isfinite(diff)
        maxd = diff[finite].max().item() if finite.any() else float("nan")
        if maxd <= atol:
            return  # 通过时不打印

    print(f"    ┌─ {label}")
    print(f"    │  实际 shape = {tuple(g.shape)}")
    print(f"    │  期望 shape = {tuple(w.shape)}")
    if not same_shape:
        print("    │  shape 不一致，先修形状")
        print("    └─")
        raise AssertionError(f"{label}: shape 不一致 {tuple(g.shape)} vs {tuple(w.shape)}")
    print(f"    │  max|实际-期望| = {maxd:.3e}   (容差 {atol:.0e})")
    bad = (diff > atol).nonzero()
    print(f"    │  不一致 {int((diff > atol).sum())} 处，前 {min(8, len(bad))} 个：")
    for idx in bad[:8]:
        key = tuple(idx.tolist())
        print(
            f"    │    位置 {key}: 实际={fmt(g[key].item())}  "
            f"期望={fmt(w[key].item())}  差={diff[key].item():.3e}"
        )
    print("    └─")
    raise AssertionError(f"{label}: 数值不一致 (max|diff|={maxd:.3e})")


def show_seq(label: str, got, want):
    """打印 token 序列的形状与首个不同位置。"""
    g, w = got.tolist(), want.tolist()
    print(f"    ┌─ {label}")
    print(f"    │  实际 shape={tuple(got.shape)} 期望 shape={tuple(want.shape)}")
    if got.shape != want.shape:
        print("    │  shape 不一致")
        print("    └─")
        return
    for b, (gb, wb) in enumerate(zip(g, w)):
        if gb != wb:
            pos = next(i for i, (x, y) in enumerate(zip(gb, wb)) if x != y)
            print(f"    │  batch {b}: 首个不同位置 pos={pos}")
            print(f"    │    实际[{pos}]={gb[pos]}  期望[{pos}]={wb[pos]}")
            print(f"    │    实际后 10 个={gb[pos:pos + 10]}")
            print(f"    │    期望后 10 个={wb[pos:pos + 10]}")
            print("    └─")
            raise AssertionError(f"{label}: 首个不同位置 pos={pos}（实际 {gb[pos]} vs 期望 {wb[pos]}）")
    print("    └─")


# ============================================================================
# 单测 1~5：纯张量
# ============================================================================
def test_temperature():
    torch.manual_seed(0)
    logits = torch.randn(2, 7)
    out = apply_temperature(logits, 2.0)
    if not isinstance(out, torch.Tensor):
        raise AssertionError(f"apply_temperature 应返回 Tensor，实际返回 {type(out).__name__}")
    show_diff("logits / 2.0", out, logits / 2.0)  # 不符会打印数值并抛异常
    show_diff("T=1.0 恒等", apply_temperature(logits, 1.0), logits, atol=0)

    before = logits.clone()
    apply_temperature(logits, 2.0)
    show_diff("输入是否被修改", logits, before, atol=0)

    for bad in (0.0, -1.0):
        try:
            res = apply_temperature(logits, bad)
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"temperature={bad} 应 raise ValueError，实际抛了 {type(exc).__name__}")
        raise AssertionError(
            f"temperature={bad} 应 raise ValueError，实际正常返回了 shape={tuple(res.shape) if torch.is_tensor(res) else type(res).__name__}"
        )


def test_repetition_penalty():
    logits = torch.tensor([[2.0, -2.0, 0.5, 1.0, -0.5]])
    prev = torch.tensor([[0, 1, 2, 0]])
    expected = torch.tensor([[1.0, -4.0, 0.25, 1.0, -0.5]])

    out = apply_repetition_penalty(logits, prev, 2.0)
    if not torch.allclose(out, expected):
        print("    输入 logits =", [fmt(x) for x in logits[0].tolist()])
        print("    previous_ids =", prev[0].tolist())
        print("    逐 token 对照（penalty=2.0）:")
        for i in range(logits.size(1)):
            print(
                f"      token {i}: 原始={fmt(logits[0, i].item())}  "
                f"期望={fmt(expected[0, i].item())}  实际={fmt(out[0, i].item())}  "
                f"{'已出现' if i in prev[0].tolist() else '未出现'}"
            )
        show_diff("repetition penalty", out, expected)
    show_diff("penalty=1.0 恒等", apply_repetition_penalty(logits, prev, 1.0), logits, atol=0)

    logits2 = torch.stack([logits[0], logits[0]])
    prev2 = torch.tensor([[0], [1]])
    out2 = apply_repetition_penalty(logits2, prev2, 2.0)
    show_diff("batch[0] 只罚 token0", out2[0], torch.tensor([1.0, -2.0, 0.5, 1.0, -0.5]))
    show_diff("batch[1] 只罚 token1", out2[1], torch.tensor([2.0, -4.0, 0.5, 1.0, -0.5]))

    before = logits.clone()
    apply_repetition_penalty(logits, prev, 2.0)
    show_diff("输入是否被修改", logits, before, atol=0)


def show_topk(logits: torch.Tensor, out: torch.Tensor, k: int):
    for b in range(logits.size(0)):
        kept = torch.isfinite(out[b]).nonzero().flatten().tolist()
        kept_sorted = sorted(kept, key=lambda i: logits[b, i].item(), reverse=True)
        want = torch.topk(logits[b], k).indices.tolist()
        if set(kept) == set(want):
            continue
        print(f"    ┌─ top_k 第 {b} 行（k={k}）")
        print(f"    │  实际保留 {len(kept)} 个: {kept_sorted}")
        print(f"    │  期望保留 {len(want)} 个: {want}")
        print(f"    │  你多留了:  {sorted(set(kept) - set(want))}")
        print(f"    │  你少留了:  {sorted(set(want) - set(kept))}")
        print("    │  相关 token 的原始 logit：")
        for i in sorted(set(kept) | set(want)):
            mark_got = "保留" if i in kept else "-inf"
            mark_want = "保留" if i in want else "-inf"
            print(f"    │    token {i:>3}: logit={fmt(logits[b, i].item())}  实际={mark_got}  期望={mark_want}")
        print("    └─")


def test_top_k():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 50, generator=g)
    k = 5
    out = apply_top_k(logits, k)

    bad = [b for b in range(3) if int(torch.isfinite(out[b]).sum()) != k
        or set(torch.isfinite(out[b]).nonzero().flatten().tolist()) != set(torch.topk(logits[b], k).indices.tolist())]
    for b in range(3):
        kept = torch.isfinite(out[b]).nonzero().flatten()
        want = torch.topk(logits[b], k).indices
        if set(kept.tolist()) != set(want.tolist()):
            show_topk(logits, out, k)
            raise AssertionError(f"第 {b} 行保留的不是最大的 {k} 个")
    show_diff("保留项数值", out[torch.isfinite(out)], logits[torch.isfinite(out)], atol=0)

    for no_cut in (0, -1, 50, 999):
        show_diff(f"k={no_cut} 不截断", apply_top_k(logits, no_cut), logits, atol=0)

    before = logits.clone()
    apply_top_k(logits, k)
    show_diff("输入是否被修改", logits, before, atol=0)


def show_topp(logits: torch.Tensor, out: torch.Tensor, p: float):
    probs = torch.softmax(logits, dim=-1)
    for b in range(logits.size(0)):
        kept = torch.isfinite(out[b])
        idx = kept.nonzero().flatten()
        order = torch.argsort(probs[b], descending=True)
        prob_sorted = probs[b][order]
        cum = torch.cumsum(prob_sorted, dim=-1)
        cutoff = int((cum >= p).nonzero()[0].item()) if (cum >= p).any() else logits.size(1) - 1
        print(f"    ┌─ top_p 第 {b} 行（p={p}）")
        print(f"    │  实际保留 {int(kept.sum())} 个，概率和={probs[b][kept].sum().item():.6f}")
        print(f"    │  期望保留前 {cutoff + 1} 个（累计概率首次 >= {p} 的位置）")
        print("    │  rank  token   prob      cumsum    你保留?  期望保留?")
        lo, hi = max(0, cutoff - 4), min(logits.size(1), cutoff + 5)
        for r in range(lo, hi):
            i = int(order[r])
            print(
                f"    │  {r:>3}  {i:>5}  {prob_sorted[r].item():.6f}  {cum[r].item():.6f}"
                f"    {'yes' if kept[i] else 'no ':<7}  {'yes' if r <= cutoff else 'no'}"
            )
        print("    └─")


def test_top_p():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(4, 100, generator=g)
    p = 0.9
    out = apply_top_p(logits, p)
    probs = torch.softmax(logits, dim=-1)

    reasons = []
    for b in range(4):
        kept = torch.isfinite(out[b])
        idx = kept.nonzero().flatten()
        if int(kept.sum()) < 1:
            reasons.append(f"row{b}: 一个 token 都没保留")
            continue
        kept_sum = probs[b][kept].sum().item()
        if kept_sum < p - 1e-6:
            reasons.append(f"row{b}: 保留概率和 {kept_sum:.6f} < p={p}")
        if int(kept.sum()) > 1:
            smallest = idx[probs[b][idx].argmin()]
            rest = kept.clone()
            rest[smallest] = False
            rest_sum = probs[b][rest].sum().item()
            if rest_sum >= p:
                reasons.append(f"row{b}: 去掉最小保留项后仍有 {rest_sum:.6f} >= p，多保留了 token")
        if kept.any() and (~kept).any():
            if probs[b][~kept].max().item() > probs[b][kept].min().item() + 1e-6:
                reasons.append(f"row{b}: 截断发生在概率中间，不是从尾部截断")
    if reasons:
        show_topp(logits, out, p)
        raise AssertionError("；".join(reasons[:3]))

    for no_cut in (1.0, 1.5):
        show_diff(f"p={no_cut} 不截断", apply_top_p(logits, no_cut), logits, atol=0)
    one = apply_top_p(logits, 0.0)
    if torch.isfinite(one).sum(dim=-1).tolist() != [1, 1, 1, 1]:
        print("    p=0 时每行实际保留个数 =", torch.isfinite(one).sum(dim=-1).tolist())
        raise AssertionError("p<=0 应等价于 top-1（每行保留 1 个）")

    before = logits.clone()
    apply_top_p(logits, p)
    show_diff("输入是否被修改", logits, before, atol=0)


def test_pick_next_token():
    logits = torch.tensor([[0.0, 5.0, -1.0], [3.0, 0.0, 1.0]])
    tok = pick_next_token(logits, do_sample=False)
    if tok.shape != (2, 1) or tok.dtype != torch.long:
        print(f"    实际 shape={tuple(tok.shape)} dtype={tok.dtype}；期望 shape=(2, 1) dtype=torch.int64")
        raise AssertionError("pick_next_token 的输出形状/dtype 不对")
    if tok.flatten().tolist() != [1, 0]:
        print(f"    贪心结果 实际={tok.flatten().tolist()} 期望=[1, 0]")
        print("    输入 logits =", logits.tolist())
        raise AssertionError("argmax 结果不对")

    near_one_hot = torch.tensor([[-1e4, 1e4, -1e4]])
    torch.manual_seed(0)
    picks = [pick_next_token(near_one_hot, do_sample=True).item() for _ in range(5)]
    if any(x != 1 for x in picks):
        print(f"    近 one-hot 分布采样结果: {picks}（期望全是 1）")
        raise AssertionError("do_sample=True 采样落到了概率接近 0 的 token")


# ============================================================================
# 单测 6~9：需要模型
# ============================================================================
def top5_at(model, tok, seq: torch.Tensor, my_token: int):
    """在 seq 的末尾打印仓库模型 top-5 next-token，并标出你的选择排第几。"""
    with torch.inference_mode():
        out = model(seq)
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    vals, ids = torch.topk(logits, 5)
    rank = (ids == my_token).nonzero().flatten().tolist()
    print(f"    │  仓库模型在该位置的 top-5 next-token（你的选择 id={my_token} 排名={rank[0] + 1 if rank else '5 名之外'}）：")
    for v, i, pr in zip(vals.tolist(), ids.tolist(), probs[ids].tolist()):
        mark = "  ← 你的选择" if i == my_token else ""
        print(f"    │    id={i:<5} logit={v:8.3f}  p={pr:.5f}  {tok.decode([i])!r}{mark}")
    if not rank:
        print(f"    │    你的选择 id={my_token} logit={logits[my_token].item():.3f}  p={probs[my_token].item():.6f}")


def report_parity(tok, ours, ref, ids, model):
    prompt_len = ids.shape[1]
    print("    ┌─ 贪心对齐失败")
    print(f"    │  prompt_len={prompt_len}")
    print(f"    │  实际 shape={tuple(ours.shape)}  期望 shape={tuple(ref.shape)}")
    print(f"    │  实际生成文本: {tok.decode(ours[0, prompt_len:], skip_special_tokens=True)!r}")
    print(f"    │  期望生成文本: {tok.decode(ref[0, prompt_len:], skip_special_tokens=True)!r}")

    n = min(ours.shape[1], ref.shape[1])
    diff = (ours[:, :n] != ref[:, :n]).nonzero()
    if len(diff):
        b, pos = diff[0].tolist()
        print(f"    │  首个不同位置 pos={pos}（batch={b}，prompt_len={prompt_len}）")
        print(f"    │  仓库 token id={ref[b, pos].item()} -> {tok.decode([ref[b, pos].item()])!r}")
        print(f"    │  你的 token id={ours[b, pos].item()} -> {tok.decode([ours[b, pos].item()])!r}")
        ctx_lo = max(0, pos - 12)
        print(f"    │  此前 12 个 token: {ref[b, ctx_lo:pos].tolist()}")
        print(f"    │  此前解码文本: {tok.decode(ref[b, ctx_lo:pos])!r}")
        print(f"    │  你的 token 是否在仓库 top-5 里：见下方")
        if pos > 0:
            top5_at(model, tok, ref[:, :pos], ours[b, pos].item())
    if ours.shape != ref.shape:
        print("    │  总长度不同 → eos 提前停止 / finished 填充逻辑不一致")
        print(f"    │  实际尾部 8 个: {ours[0, -8:].tolist()}")
        print(f"    │  期望尾部 8 个: {ref[0, -8:].tolist()}")
        print(f"    │  实际 eos 位置（生成段，绝对位置）: {((ours[0, prompt_len:] == 2).nonzero().flatten() + prompt_len).tolist()}")
        print(f"    │  期望 eos 位置（生成段，绝对位置）: {((ref[0, prompt_len:] == 2).nonzero().flatten() + prompt_len).tolist()}")
    print("    └─")


def test_greedy_parity():
    model, tok = get_model()
    ids = make_ids()
    n = 24

    forbidden = lambda *a, **k: (_ for _ in ()).throw(AssertionError("generate 内部调用了 model.generate"))
    model.generate = forbidden
    try:
        ours = generate(
            model, ids, max_new_tokens=n, temperature=1.0, top_k=0, top_p=1.0,
            repetition_penalty=1.0, do_sample=False, eos_token_id=2, use_cache=True,
        )
    finally:
        del model.generate

    ref = model.generate(
        inputs=ids, max_new_tokens=n, temperature=1.0, top_k=0, top_p=1.0,
        do_sample=False, eos_token_id=2, use_cache=True,
    )

    if not torch.equal(ours[:, : ids.shape[1]], ids):
        same_len = min(ours.shape[1], ids.shape[1])
        show_seq("返回结果必须保留完整 prompt", ours[:, :same_len], ids[:, :same_len])
        raise AssertionError("返回结果没有保留完整 prompt（应返回 prompt + 新 token）")
    if ours.shape != ref.shape or not torch.equal(ours, ref):
        report_parity(tok, ours, ref, ids, model)
        if ours.shape != ref.shape:
            raise AssertionError(f"贪心生成长度不一致：实际 {tuple(ours.shape)} vs 期望 {tuple(ref.shape)}")
        raise AssertionError("贪心生成与仓库 generate 不一致（见上方首个不同位置）")

    print(f"      生成文本：{tok.decode(ours[0, ids.shape[1]:], skip_special_tokens=True)!r}")


def test_cache_equivalence():
    model, tok = get_model()
    ids = make_ids()
    kw = dict(max_new_tokens=16, temperature=1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, do_sample=False, eos_token_id=2)
    with_cache = generate(model, ids, use_cache=True, **kw)
    no_cache = generate(model, ids, use_cache=False, **kw)
    if not torch.equal(with_cache, no_cache):
        print(f"    ┌─ use_cache=True vs False")
        print(f"    │  实际生成: {tok.decode(with_cache[0, ids.shape[1]:], skip_special_tokens=True)!r}")
        print(f"    │  期望生成: {tok.decode(no_cache[0, ids.shape[1]:], skip_special_tokens=True)!r}")
        print("    └─")
        show_seq("cache 生成结果", with_cache, no_cache)
        raise AssertionError("use_cache=True/False 结果不一致")


def test_sampling_reproducible():
    model, tok = get_model()
    ids = make_ids()
    kw = dict(max_new_tokens=16, temperature=0.9, top_k=50, top_p=0.9,
        repetition_penalty=1.0, do_sample=True, eos_token_id=2)

    torch.manual_seed(123)
    a = generate(model, ids, **kw)
    torch.manual_seed(123)
    b = generate(model, ids, **kw)
    if not torch.equal(a, b):
        print("    ┌─ 同一 seed=123 两次采样结果不同")
        print(f"    │  1st: {tok.decode(a[0, ids.shape[1]:], skip_special_tokens=True)!r}")
        print(f"    │  2nd: {tok.decode(b[0, ids.shape[1]:], skip_special_tokens=True)!r}")
        print("    └─")
        raise AssertionError("同一 seed 两次采样应完全一致（检查是否用了固定 seed / 随机源不对）")

    torch.manual_seed(7)
    c = generate(model, ids, **kw)
    if torch.equal(a, c):
        print("    ┌─ 不同 seed 得到了相同结果")
        print(f"    │  文本: {tok.decode(a[0, ids.shape[1]:], skip_special_tokens=True)!r}")
        print("    └─")
        raise AssertionError("不同 seed 结果完全相同：do_sample=True 可能没真的采样（例如写死成 argmax）")


def test_stop_conditions():
    model, tok = get_model()
    ids = make_ids()
    out = generate(model, ids, max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, do_sample=False, eos_token_id=2)
    print(f"      生成 {out.shape[1] - ids.shape[1]} 个 token：{tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)!r}")
    if out.shape[1] > ids.shape[1] + 8:
        raise AssertionError(f"超过 max_new_tokens：实际生成了 {out.shape[1] - ids.shape[1]} 个")
    if out.shape[1] == ids.shape[1]:
        raise AssertionError("一个 token 都没生成")

    for row in out[:, ids.shape[1]:]:
        hit = (row == 2).nonzero()
        if len(hit):
            tail = row[hit[0]:]
            if not torch.all(tail == 2):
                print(f"    该行: {row.tolist()}")
                print(f"    eos 之后出现了非 eos token: {tail.tolist()}")
                raise AssertionError("出现 eos 后该行之后必须全部是 eos")


# ============================================================================
TESTS = [
    test_temperature,
    test_repetition_penalty,
    test_top_k,
    test_top_p,
    test_pick_next_token,
    test_greedy_parity,
    test_cache_equivalence,
    test_sampling_reproducible,
    test_stop_conditions,
]

HINTS = {
    "test_temperature": "T<=0 抛 ValueError；T=1 数值恒等；不要修改输入张量",
    "test_repetition_penalty": "正值 /= penalty，非正值 *= penalty；只处理 previous_ids 里出现过的 token，重复出现只罚一次",
    "test_top_k": "保留 >= 第 k 大值的位置；k<=0 或 k>=V 不截断；裁掉置 -inf",
    "test_top_p": "降序 softmax → cumsum；保留到第一个 cumsum>=p 的位置（含）；至少留 1 个；p>=1 直接返回",
    "test_pick_next_token": "argmax / multinomial(softmax(logits), 1)；返回 (B,1) long",
    "test_greedy_parity": "先查是否返回完整序列；再查 temperature→penalty→top_k→top_p 顺序、cache 切片 input_ids[:, past_len:]、不要调用 model.generate",
    "test_cache_equivalence": "use_cache=True 只喂新 token 并取 outputs.past_key_values；False 时每步喂完整序列",
    "test_sampling_reproducible": "采样必须用 torch.multinomial（全局 RNG），不要固定 seed 或退化成 argmax",
    "test_stop_conditions": "finished 后该行填 eos；finished.all() 提前 break；长度不超过 max_new_tokens",
}


def main() -> int:
    filters = [a.lower() for a in sys.argv[1:] if not a.startswith("-")]
    selected = [t for t in TESTS if not filters or any(f in t.__name__.lower() for f in filters)]
    if not selected:
        print(f"没有匹配 {filters} 的测试，可选：{[t.__name__ for t in TESTS]}")
        return 2

    print(f"device = {DEVICE} | torch = {torch.__version__} | 运行 {len(selected)} 项")
    failed = []
    for fn in selected:
        t0 = time.time()
        try:
            fn()
        except NotImplementedError as exc:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__:28s} 未实现（{exc}）")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:28s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:28s} ({time.time() - t0:.1f}s)")

    print("-" * 60)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D1 可以勾选 S2-02 / S2-03 / S2-07。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
