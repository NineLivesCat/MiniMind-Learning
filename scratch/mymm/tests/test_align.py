"""D6 官方权重对齐验收（LEARNING_TASKS: S3-11 / S3-12 / S3-14）

这一部分没有 TODO——D5 的模型已经写完了，现在只做验收：
    1) full_sft_768.pth 能用 strict=True 加载进你的 CausalLM
    2) 同一输入下 logits / loss 与仓库实现对齐
    3) 用你自己的 sampling.generate + 官方权重能正常对话

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_align.py

如果失败，诊断信息会指出是 key 不匹配、数值偏差还是生成不一致；
对应要改的是 mymm/model.py / attention.py / rope.py 等，不要改测试来“通过”。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

from mymm.config import MiniMindConfig  # noqa: E402
from mymm.model import CausalLM  # noqa: E402
from mymm.sampling import generate as my_generate  # noqa: E402

from model.model_minimind import MiniMindConfig as RefConfig  # noqa: E402
from model.model_minimind import MiniMindForCausalLM as RefCausalLM  # noqa: E402

WEIGHT = REPO / "out" / "full_sft_768.pth"
TOKENIZER_DIR = REPO / "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_PARAMS = 63_912_192
EXPECTED_KEYS = 91

_CACHE: dict = {}


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


# ============================================================================
# 加载
# ============================================================================
def get_tokenizer():
    if "tok" not in _CACHE:
        _CACHE["tok"] = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return _CACHE["tok"]


def official_state_dict():
    if "sd" not in _CACHE:
        _CACHE["sd"] = torch.load(str(WEIGHT), map_location="cpu", weights_only=True)
    return _CACHE["sd"]


def build_ours(dtype: torch.dtype):
    key = ("ours", dtype)
    if key not in _CACHE:
        model = CausalLM(MiniMindConfig())
        sd = official_state_dict()

        ours_keys, want_keys = set(model.state_dict()), set(sd)
        missing = sorted(ours_keys - want_keys)
        extra = sorted(want_keys - ours_keys)
        if missing or extra:
            print("    权重里有、你的模型没有（缺失 key）：", missing[:20])
            print("    你的模型有、权重里没有（多余 key）：", extra[:20])
            raise AssertionError("state_dict 的 key 集合和官方权重不一致，先修 D5 的命名/结构")

        model.load_state_dict(sd, strict=True)
        _CACHE[key] = model.eval().to(dtype).to(DEVICE)
    return _CACHE[key]


def build_ref(dtype: torch.dtype):
    key = ("ref", dtype)
    if key not in _CACHE:
        model = RefCausalLM(RefConfig(hidden_size=768, num_hidden_layers=8))
        model.load_state_dict(official_state_dict(), strict=True)
        _CACHE[key] = model.eval().to(dtype).to(DEVICE)
    return _CACHE[key]


def encode(text: str) -> torch.Tensor:
    tok = get_tokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok(prompt, return_tensors="pt").input_ids.to(DEVICE)


# ============================================================================
# 测试
# ============================================================================
def test_checkpoint_keys():
    sd = official_state_dict()
    if len(sd) != EXPECTED_KEYS:
        print(f"    官方权重实际 key 数={len(sd)}，期望 {EXPECTED_KEYS}")
        raise AssertionError("权重 key 数与预期不符（权重文件是否完整？）")

    for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight"):
        if name not in sd:
            raise AssertionError(f"官方权重缺少 {name}")

    dtypes = {v.dtype for v in sd.values()}
    if dtypes != {torch.float16}:
        print(f"    实际 dtype 集合={dtypes}，期望 {{torch.float16}}")
        raise AssertionError("官方 pth 应该是 fp16")

    bad = [k for k in sd if "freqs" in k or "rope" in k]
    if bad:
        raise AssertionError(f"权重里不应有 RoPE buffer: {bad}")


def test_strict_load():
    model = build_ours(torch.float32)
    if "model.freqs_cos" in model.state_dict():
        raise AssertionError("freqs_cos 不应出现在 state_dict（register_buffer 要 persistent=False）")
    bufs = dict(model.named_buffers())
    if tuple(bufs["model.freqs_cos"].shape) != (32768, 96):
        print(f"    freqs_cos.shape={tuple(bufs['model.freqs_cos'].shape)}")
        raise AssertionError("RoPE buffer 形状不对")


def test_param_count():
    model = build_ours(torch.float32)
    total = sum(p.numel() for p in model.parameters())
    print(f"      Model Params: {total / 1e6:.2f}M")
    if total != EXPECTED_PARAMS:
        print(f"    实际参数量={total}，期望 {EXPECTED_PARAMS}")
        raise AssertionError("参数量不对")


def test_logits_parity():
    ours = build_ours(torch.float32)
    ref = build_ref(torch.float32)
    ids = encode("你好，请用一句话介绍你自己。")
    labels = ids.clone()
    labels[:, :3] = -100

    with torch.no_grad():
        a = ours(ids)
        b = ref(ids)
        show_diff("官方权重 logits vs 仓库 (fp32)", a.logits, b.logits, atol=1e-4)

        a2 = ours(ids, labels=labels)
        b2 = ref(ids, labels=labels)
        show_diff("官方权重 loss vs 仓库", a2.loss, b2.loss, atol=1e-4)

        a3 = ours(ids, use_cache=True)
        b3 = ref(ids, use_cache=True)
    show_diff("present k[0] vs 仓库", a3.past_key_values[0][0], b3.past_key_values[0][0], atol=1e-4)


def test_fp16_parity():
    if DEVICE != "cuda":
        print("      跳过：没有 CUDA")
        return
    ours = build_ours(torch.float16)
    ref = build_ref(torch.float16)
    ids = encode("用一句话解释什么是机器学习。")
    with torch.no_grad():
        a = ours(ids)
        b = ref(ids)
    if a.logits.dtype != torch.float16:
        print(f"    输出 dtype={a.logits.dtype}")
        raise AssertionError("fp16 权重应输出 fp16 logits")
    show_diff("官方权重 logits vs 仓库 (fp16)", a.logits, b.logits, atol=1e-3)


def test_generate_parity():
    ours = build_ours(torch.float32)
    ref = build_ref(torch.float32)
    ids = encode("请用一句话介绍你自己。")
    n = 24

    out_ours = my_generate(
        ours, ids, max_new_tokens=n, temperature=1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, do_sample=False, eos_token_id=2, use_cache=True,
    )
    out_ref = ref.generate(
        inputs=ids, max_new_tokens=n, temperature=1.0, top_k=0, top_p=1.0,
        do_sample=False, eos_token_id=2, use_cache=True,
    )
    if out_ours.shape != out_ref.shape or not torch.equal(out_ours, out_ref):
        print(f"    你的输出 shape={tuple(out_ours.shape)}")
        print(f"    仓库输出 shape={tuple(out_ref.shape)}")
        print(f"    你的: {out_ours[0].tolist()}")
        print(f"    仓库: {out_ref[0].tolist()}")
        raise AssertionError("用你的模型 + 你的 sampling.generate，贪心结果与仓库不一致")

    tok = get_tokenizer()
    text = tok.decode(out_ours[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"      生成：{text!r}")


def test_cache_equivalence_official():
    ours = build_ours(torch.float32)
    ids = encode("解释一下什么是光合作用。")
    kw = dict(max_new_tokens=16, temperature=1.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, do_sample=False, eos_token_id=2)
    with_cache = my_generate(ours, ids, use_cache=True, **kw)
    no_cache = my_generate(ours, ids, use_cache=False, **kw)
    if not torch.equal(with_cache, no_cache):
        print(f"    cache=True : {with_cache[0].tolist()}")
        print(f"    cache=False: {no_cache[0].tolist()}")
        raise AssertionError("官方权重下 use_cache=True/False 贪心结果不一致")


def test_chat():
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = build_ours(dtype)
    tok = get_tokenizer()
    prompts = [
        "你好，你是谁？",
        "用一句话解释什么是机器学习。",
        "为什么天空是蓝色的？",
    ]
    for prompt in prompts:
        torch.manual_seed(0)
        ids = encode(prompt)
        t0 = time.time()
        out = my_generate(
            model, ids, max_new_tokens=64, temperature=0.85, top_k=50, top_p=0.85,
            repetition_penalty=1.0, do_sample=True, eos_token_id=2, use_cache=True,
        )
        elapsed = time.time() - t0
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        tps = (out.shape[1] - ids.shape[1]) / max(elapsed, 1e-6)
        print(f"      💬 {prompt}")
        print(f"         🤖 {text!r}  ({tps:.1f} tok/s)")
        if len(text) < 4:
            raise AssertionError(f"回答为空或过短：{text!r}")


# ============================================================================
TESTS = [
    test_checkpoint_keys,
    test_strict_load,
    test_param_count,
    test_logits_parity,
    test_fp16_parity,
    test_generate_parity,
    test_cache_equivalence_official,
    test_chat,
]

HINTS = {
    "test_checkpoint_keys": "权重文件完整吗？full_sft_768.pth 应在 minimind/out 下",
    "test_strict_load": "检查 state_dict 命名和 freqs buffer 的 persistent=False",
    "test_param_count": "参数量差说明结构不对（层数/head_dim/intermediate_size/tie）",
    "test_logits_parity": "logits 偏了：按 RoPE → attention mask → qk norm → tie 的顺序排查",
    "test_fp16_parity": "fp16 误差 1e-3 内算通过；偏大说明有 fp32/fp16 混算问题",
    "test_generate_parity": "贪心不一致：先看 sampling 的 cache 切片和 op 顺序，再看模型 logits",
    "test_cache_equivalence_official": "cache 路径不对：present 长度或 start_pos 切片有误",
    "test_chat": "能生成但质量差先不管，先确认不报错、不是复读/乱码",
}


def main() -> int:
    filters = [a.lower() for a in sys.argv[1:] if not a.startswith("-")]
    selected = [t for t in TESTS if not filters or any(f in t.__name__.lower() for f in filters)]
    if not selected:
        print(f"没有匹配 {filters} 的测试，可选：{[t.__name__ for t in TESTS]}")
        return 2

    if not WEIGHT.exists():
        print(f"找不到官方权重：{WEIGHT}")
        print("先运行：modelscope download --model gongjy/minimind-3-pytorch full_sft_768.pth --local_dir D:\\minimind\\minimind\\out")
        return 2

    print(f"device = {DEVICE} | torch = {torch.__version__} | 权重 = {WEIGHT.name} | 运行 {len(selected)} 项")
    failed = []
    for fn in selected:
        t0 = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:30s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:30s} ({time.time() - t0:.1f}s)")

    print("-" * 68)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D6 可以勾选 S3-11 / S3-12 / S3-14，G3 达成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
