"""D5 验收测试：Transformer + CausalLM 组装

运行全部：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_model.py

只跑某几项：
    python mymm\\tests\\test_model.py transformer
    python mymm\\tests\\test_model.py causallm
    python mymm\\tests\\test_model.py generate

失败时打印「期望/实际/首个不一致位置」；实现前全部 FAIL（NotImplementedError）属正常。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

from mymm.config import MiniMindConfig  # noqa: E402
from mymm.model import CausalLM, Transformer  # noqa: E402
from mymm.sampling import generate as my_generate  # noqa: E402

from model.model_minimind import MiniMindConfig as RefConfig  # noqa: E402
from model.model_minimind import MiniMindForCausalLM as RefCausalLM  # noqa: E402
from model.model_minimind import MiniMindModel as RefTransformer  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 6400
HEAD_DIM = 96


# ============================================================================
# 诊断工具
# ============================================================================
def fmt(x: float) -> str:
    return f"{x:.6g}"


def show_diff(label: str, got, want, atol: float = 1e-6):
    if not torch.is_tensor(got) or not torch.is_tensor(want):
        raise AssertionError(f"{label}: 期望 Tensor，实际 got={type(got).__name__} want={type(want).__name__}")
    g = got.detach().to(torch.float32).cpu()
    w = want.detach().to(torch.float32).cpu()
    if g.shape != w.shape:
        print(f"    ┌─ {label}")
        print(f"    │  实际 shape={tuple(g.shape)}  期望 shape={tuple(w.shape)}")
        print("    └─")
        raise AssertionError(f"{label}: shape 不一致 {tuple(g.shape)} vs {tuple(w.shape)}")
    diff = (g - w).abs()
    maxd = diff.max().item() if diff.numel() else 0.0
    if maxd <= atol:
        return
    print(f"    ┌─ {label}")
    print(f"    │  max|实际-期望| = {maxd:.3e}   (容差 {atol:.0e})")
    bad = (diff > atol).nonzero()
    print(f"    │  不一致 {int((diff > atol).sum())} 处，前 {min(6, len(bad))} 个：")
    for idx in bad[:6]:
        key = tuple(idx.tolist())
        print(f"    │    位置 {key}: 实际={fmt(g[key].item())}  期望={fmt(w[key].item())}  差={diff[key].item():.3e}")
    print("    └─")
    raise AssertionError(f"{label}: 数值不一致 (max|diff|={maxd:.3e})")


def make_ref_cfg(cfg: MiniMindConfig):
    return RefConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
    )


def pair_transformer(cfg: MiniMindConfig | None = None):
    cfg = cfg or MiniMindConfig()
    ours = Transformer(cfg).eval()
    ref = RefTransformer(make_ref_cfg(cfg)).eval()
    ref.load_state_dict(ours.state_dict(), strict=True)
    return ours, ref


def pair_lm(cfg: MiniMindConfig | None = None):
    cfg = cfg or MiniMindConfig()
    ours = CausalLM(cfg).eval()
    ref = RefCausalLM(make_ref_cfg(cfg)).eval()
    ref.load_state_dict(ours.state_dict(), strict=True)
    return ours, ref


# ============================================================================
# Transformer（S3-09）
# ============================================================================
def test_transformer_init():
    cfg = MiniMindConfig()
    ours = Transformer(cfg)
    ref = RefTransformer(make_ref_cfg(cfg))

    ours_keys, ref_keys = set(ours.state_dict()), set(ref.state_dict())
    if ours_keys != ref_keys:
        print("    仓库有、你没有：", sorted(ref_keys - ours_keys)[:10])
        print("    你有、仓库没有：", sorted(ours_keys - ref_keys)[:10])
        raise AssertionError("Transformer 的 state_dict key 集合和仓库不一致")

    bufs = dict(ours.named_buffers())
    for name in ("freqs_cos", "freqs_sin"):
        if name not in bufs:
            raise AssertionError(f"缺少 buffer {name}")
        if tuple(bufs[name].shape) != (32768, HEAD_DIM):
            print(f"    {name}.shape={tuple(bufs[name].shape)}，期望 (32768, {HEAD_DIM})")
            raise AssertionError(f"{name} 形状不对")
        if name in ours.state_dict():
            raise AssertionError(f"{name} 不应出现在 state_dict（register_buffer 要 persistent=False）")

    if tuple(ours.embed_tokens.weight.shape) != (VOCAB, 768):
        raise AssertionError("embed_tokens 形状不对")
    if len(ours.layers) != 8:
        raise AssertionError("层数不对")

    total = sum(p.numel() for p in ours.parameters())
    if total != 63_912_192:
        print(f"    实际参数量={total}，期望 63,912,192（8 * 7,374,528 + 4,915,200 + final norm 768）")
        raise AssertionError("Transformer 参数量不对")


def test_transformer_forward():
    torch.manual_seed(0)
    ours, ref = pair_transformer()
    ids = torch.randint(0, VOCAB, (2, 7))
    with torch.no_grad():
        hidden, presents, aux = ours(ids)
        ref_hidden, ref_presents, ref_aux = ref(ids)

    if tuple(hidden.shape) != (2, 7, 768):
        print(f"    hidden.shape={tuple(hidden.shape)}，期望 (2, 7, 768)")
        raise AssertionError("Transformer 输出形状不对")
    show_diff("hidden_states vs 仓库", hidden, ref_hidden, atol=1e-5)
    if len(presents) != 8 or any(p is not None for p in presents):
        raise AssertionError("use_cache=False 时应返回 8 个 None")
    if aux is None or aux.numel() != 1 or aux.item() != 0.0:
        print(f"    aux_loss={aux!r}，期望 0 维 0")
        raise AssertionError("无 MoE 时 aux_loss 应为 0")


def test_transformer_cache():
    torch.manual_seed(1)
    ours, _ = pair_transformer()
    ids = torch.randint(0, VOCAB, (1, 9))
    with torch.no_grad():
        full, full_past, _ = ours(ids, use_cache=True)
        h1, past1, _ = ours(ids[:, :5], use_cache=True)
        h2, past2, _ = ours(ids[:, 5:], past_key_values=past1, use_cache=True)

    show_diff("cache 分块（后 4 位）vs 全量", h2, full[:, 5:], atol=1e-4)
    for layer in range(8):
        if tuple(past2[layer][0].shape) != (1, 9, 4, HEAD_DIM):
            print(f"    layer {layer} present k.shape={tuple(past2[layer][0].shape)}")
            raise AssertionError("present 形状不对")
        show_diff(f"layer {layer} present k vs 全量", past2[layer][0], full_past[layer][0], atol=1e-5)


# ============================================================================
# CausalLM（S3-10 / S3-13 / S3-14）
# ============================================================================
def test_causallm_init():
    ours = CausalLM(MiniMindConfig())
    ref = RefCausalLM(make_ref_cfg(MiniMindConfig()))

    ours_keys, ref_keys = set(ours.state_dict()), set(ref.state_dict())
    if ours_keys != ref_keys:
        print("    仓库有、你没有：", sorted(ref_keys - ours_keys)[:10])
        print("    你有、仓库没有：", sorted(ours_keys - ref_keys)[:10])
        raise AssertionError("CausalLM 的 state_dict key 集合和仓库不一致")

    if not (ours.lm_head.weight is ours.model.embed_tokens.weight):
        raise AssertionError("tie_word_embeddings=True 时 lm_head.weight 必须和 embed_tokens.weight 是同一个对象")
    if ours.lm_head.weight.data_ptr() != ours.model.embed_tokens.weight.data_ptr():
        raise AssertionError("lm_head 与 embedding 共享存储失败（被 copy 了？）")

    total = sum(p.numel() for p in ours.parameters())
    if total != 63_912_192:
        print(f"    实际参数量={total}，期望 63,912,192（tie 之后只算一次 embedding，含 final norm）")
        raise AssertionError("CausalLM 参数量不对")


def test_causallm_forward():
    torch.manual_seed(2)
    ours, _ = pair_lm()
    ids = torch.randint(0, VOCAB, (2, 7))
    with torch.no_grad():
        out = ours(ids)
        if out.loss is not None:
            raise AssertionError("labels=None 时 loss 应为 None")
        if tuple(out.logits.shape) != (2, 7, VOCAB):
            print(f"    logits.shape={tuple(out.logits.shape)}，期望 (2, 7, {VOCAB})")
            raise AssertionError("logits 形状不对")
        if len(out.past_key_values) != 8:
            raise AssertionError("past_key_values 长度应为 8")
        if tuple(out.hidden_states.shape) != (2, 7, 768):
            raise AssertionError("hidden_states 形状不对")

        out2 = ours(ids, labels=ids.clone())
    if out2.loss is None or out2.loss.numel() != 1:
        raise AssertionError("labels 不为 None 时必须返回标量 loss")
    loss = out2.loss.item()
    if not (7.0 < loss < 11.0):
        print(f"    初始 loss={loss:.4f}，期望在 ln(6400)≈8.76 附近（7~11）")
        raise AssertionError("随机初始化时 loss 异常")


def test_causallm_loss_shift():
    torch.manual_seed(3)
    ours, _ = pair_lm()
    ids = torch.randint(0, VOCAB, (2, 7))
    labels = ids.clone()
    labels[:, :3] = -100

    with torch.no_grad():
        out = ours(ids, labels=labels)
        logits = out.logits
        manual = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, VOCAB),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
    show_diff("shift 后的 loss", out.loss, manual, atol=1e-6)

    with torch.no_grad():
        wrong = F.cross_entropy(
            logits.contiguous().view(-1, VOCAB),
            labels.contiguous().view(-1),
            ignore_index=-100,
        )
    if torch.allclose(out.loss, wrong, atol=1e-3):
        raise AssertionError("loss 看起来没有做 shift（和未 shift 的版本一样）")


def test_causallm_parity():
    torch.manual_seed(4)
    ours, ref = pair_lm()
    ids = torch.randint(0, VOCAB, (2, 7))
    labels = ids.clone()
    labels[:, :2] = -100

    with torch.no_grad():
        a = ours(ids)
        b = ref(ids)
        show_diff("logits vs 仓库", a.logits, b.logits, atol=1e-5)

        a2 = ours(ids, labels=labels)
        b2 = ref(ids, labels=labels)
        show_diff("loss vs 仓库", a2.loss, b2.loss, atol=1e-6)

        a3 = ours(ids, use_cache=True)
        b3 = ref(ids, use_cache=True)
    for layer in (0, 7):
        show_diff(f"present k[{layer}] vs 仓库", a3.past_key_values[layer][0], b3.past_key_values[layer][0], atol=1e-5)


def test_causallm_generate():
    torch.manual_seed(5)
    ours, ref = pair_lm()
    ids = torch.randint(0, VOCAB, (1, 8))

    out_ours = my_generate(
        ours, ids, max_new_tokens=12, temperature=1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, do_sample=False, eos_token_id=2, use_cache=True,
    )
    out_ref = ref.generate(
        inputs=ids, max_new_tokens=12, temperature=1.0, top_k=0, top_p=1.0,
        do_sample=False, eos_token_id=2, use_cache=True,
    )
    if out_ours.shape != out_ref.shape or not torch.equal(out_ours, out_ref):
        print(f"    你的 generate 输出 shape={tuple(out_ours.shape)}")
        print(f"    仓库 generate 输出 shape={tuple(out_ref.shape)}")
        print(f"    你的: {out_ours.tolist()}")
        print(f"    仓库: {out_ref.tolist()}")
        raise AssertionError("用你自己的 CausalLM 跑 sampling.generate，结果和仓库不一致")


def test_causallm_128():
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    ours, ref = pair_lm(cfg)
    total = sum(p.numel() for p in ours.parameters())
    if total != 1_262_272:
        print(f"    128 维 2 层实际参数量={total}，期望 1,262,272")
        raise AssertionError("128 维模型参数量不对")

    ids = torch.randint(0, VOCAB, (1, 5))
    labels = ids.clone()
    labels[:, :2] = -100
    with torch.no_grad():
        a = ours(ids, labels=labels)
        b = ref(ids, labels=labels)
    show_diff("128 维 logits vs 仓库", a.logits, b.logits, atol=1e-5)
    show_diff("128 维 loss vs 仓库", a.loss, b.loss, atol=1e-6)


# ============================================================================
TESTS = [
    test_transformer_init,
    test_transformer_forward,
    test_transformer_cache,
    test_causallm_init,
    test_causallm_forward,
    test_causallm_loss_shift,
    test_causallm_parity,
    test_causallm_generate,
    test_causallm_128,
]

HINTS = {
    "test_transformer_init": "embed_tokens + layers + norm；freqs buffer 用 persistent=False",
    "test_transformer_forward": "start_pos 取 past 长度，RoPE 表按 start_pos 切片；逐层收集 present",
    "test_transformer_cache": "带 cache 时只喂新 token，位置由 start_pos 决定",
    "test_causallm_init": "lm_head.weight 直接赋同一个 Parameter，不要 copy",
    "test_causallm_forward": "返回 CausalLMOutput；labels=None 时 loss=None",
    "test_causallm_loss_shift": "logits[..., :-1, :] 对 labels[..., 1:]，ignore_index=-100",
    "test_causallm_parity": "先查 state_dict 是否和仓库一致，再查 logits/loss",
    "test_causallm_generate": "sampling.generate 依赖输出对象的 .logits 和 .past_key_values 字段",
    "test_causallm_128": "128 维先当小模型跑通，再去对齐官方权重",
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
            print(f"FAIL  {fn.__name__:26s} 未实现（{exc}）")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:26s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:26s} ({time.time() - t0:.1f}s)")

    print("-" * 64)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D5 可以勾选 S3-09 / S3-10 / S3-13 / S3-14。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
