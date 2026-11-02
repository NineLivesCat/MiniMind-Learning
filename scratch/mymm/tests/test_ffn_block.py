"""D4 验收测试：SwiGLU FFN + pre-norm Block

运行全部：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_ffn_block.py

只跑某几项：
    python mymm\\tests\\test_ffn_block.py ffn
    python mymm\\tests\\test_ffn_block.py block

失败时打印「期望/实际/首个不一致位置」；实现前部分 FAIL（NotImplementedError）属正常。
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

from mymm.block import MiniMindBlock  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.ffn import FeedForward  # noqa: E402
from mymm.rope import precompute_freqs_cis  # noqa: E402

from model.model_minimind import FeedForward as RefFeedForward  # noqa: E402
from model.model_minimind import MiniMindBlock as RefBlock  # noqa: E402
from model.model_minimind import MiniMindConfig as RefConfig  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


def pair_ffn(cfg: MiniMindConfig | None = None):
    cfg = cfg or MiniMindConfig()
    ours = FeedForward(cfg)
    ref = RefFeedForward(RefConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
    ))
    ref.load_state_dict(ours.state_dict(), strict=True)
    return ours.eval(), ref.eval()


def pair_block(cfg: MiniMindConfig | None = None):
    cfg = cfg or MiniMindConfig()
    ours = MiniMindBlock(0, cfg)
    ref = RefBlock(0, RefConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
    ))
    ref.load_state_dict(ours.state_dict(), strict=True)
    return ours.eval(), ref.eval()


# ============================================================================
# FFN（S3-05）
# ============================================================================
def test_ffn_init():
    ours, ref = pair_ffn()
    ours_keys, ref_keys = set(ours.state_dict()), set(ref.state_dict())
    if ours_keys != ref_keys:
        print("    仓库有、你没有：", sorted(ref_keys - ours_keys))
        print("    你有、仓库没有：", sorted(ours_keys - ref_keys))
        raise AssertionError("FeedForward 的 state_dict key 集合和仓库不一致")

    params = dict(ours.named_parameters())
    expected = {
        "gate_proj.weight": (2432, 768),
        "up_proj.weight": (2432, 768),
        "down_proj.weight": (768, 2432),
    }
    for name, shape in expected.items():
        if name not in params:
            raise AssertionError(f"缺少参数 {name}")
        if tuple(params[name].shape) != shape:
            print(f"    {name}.shape={tuple(params[name].shape)}，期望 {shape}")
            raise AssertionError(f"参数 {name} 形状不对")
    if any(n.endswith("bias") for n in params):
        raise AssertionError("FeedForward 不应有 bias")

    total = sum(p.numel() for p in ours.parameters())
    if total != 5_603_328:
        print(f"    实际参数量={total}，期望 5,603,328（3 * 768 * 2432）")
        raise AssertionError("FeedForward 参数量不对")


def test_ffn_formula():
    torch.manual_seed(0)
    layer = FeedForward(MiniMindConfig()).eval()
    x = torch.randn(2, 5, 768)

    got = layer(x)
    want = layer.down_proj(F.silu(layer.gate_proj(x)) * layer.up_proj(x))
    show_diff("SwiGLU 公式", got, want, atol=1e-6)

    if tuple(got.shape) != (2, 5, 768):
        print(f"    输出 shape={tuple(got.shape)}，期望 (2, 5, 768)")
        raise AssertionError("FFN 输出形状不对")


def test_ffn_parity():
    torch.manual_seed(1)
    ours, ref = pair_ffn()
    x = torch.randn(2, 7, 768)
    show_diff("FFN 前向 vs 仓库", ours(x), ref(x), atol=1e-6)


def test_ffn_tiny():
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=1)
    ours, ref = pair_ffn(cfg)
    total = sum(p.numel() for p in ours.parameters())
    if total != 172_032:
        print(f"    128 维实际参数量={total}，期望 172,032（3 * 128 * 448）")
        raise AssertionError("128 维 FFN 参数量不对")
    x = torch.randn(2, 4, 128)
    show_diff("128 维 FFN vs 仓库", ours(x), ref(x), atol=1e-6)


def test_ffn_fp16():
    if DEVICE != "cuda":
        print("      跳过：没有 CUDA")
        return
    torch.manual_seed(2)
    ours, ref = pair_ffn()
    ours, ref = ours.half().cuda(), ref.half().cuda()
    x = torch.randn(2, 5, 768, device="cuda", dtype=torch.float16)
    got, want = ours(x), ref(x)
    if got.dtype != torch.float16:
        print(f"    输出 dtype={got.dtype}，期望 torch.float16")
        raise AssertionError("fp16 输入应返回 fp16")
    show_diff("fp16 FFN vs 仓库", got, want, atol=1e-3)


def test_ffn_no_mutation():
    torch.manual_seed(3)
    layer = FeedForward(MiniMindConfig()).eval()
    x = torch.randn(2, 3, 768)
    before = x.clone()
    layer(x)
    show_diff("FFN 不修改输入", x, before, atol=0)


# ============================================================================
# Block（S3-08）
# ============================================================================
def test_block_init():
    ours = MiniMindBlock(0, MiniMindConfig())
    ref = RefBlock(0, RefConfig(hidden_size=768, num_hidden_layers=1))
    ours_keys, ref_keys = set(ours.state_dict()), set(ref.state_dict())
    if ours_keys != ref_keys:
        print("    仓库有、你没有：", sorted(ref_keys - ours_keys))
        print("    你有、仓库没有：", sorted(ours_keys - ref_keys))
        raise AssertionError("Block 的 state_dict key 集合和仓库不一致")

    total = sum(p.numel() for p in ours.parameters())
    if total != 7_374_528:
        print(f"    实际参数量={total}，期望 7,374,528（attention 1,769,664 + FFN 5,603,328 + 2 个 RMSNorm 1,536）")
        raise AssertionError("Block 参数量不对")


def test_block_prenorm():
    torch.manual_seed(4)
    block = MiniMindBlock(0, MiniMindConfig()).eval()
    S = 5
    x = torch.randn(1, S, 768)
    cos, sin = precompute_freqs_cis(HEAD_DIM, S)

    with torch.no_grad():
        got, _ = block(x, (cos, sin))
        h = block.input_layernorm(x)
        attn_out, _ = block.self_attn(h, (cos, sin))
        h = x + attn_out
        want = h + block.mlp(block.post_attention_layernorm(h))

    show_diff("pre-norm 残差结构", got, want, atol=1e-6)


def test_block_parity():
    torch.manual_seed(5)
    ours, ref = pair_block()
    S = 6
    x = torch.randn(1, S, 768)
    cos, sin = precompute_freqs_cis(HEAD_DIM, S)

    got, _ = ours(x, (cos, sin))
    want, _ = ref(x, (cos, sin))
    show_diff("Block 前向 vs 仓库", got, want, atol=1e-5)

    got_c, past = ours(x, (cos, sin), use_cache=True)
    want_c, ref_past = ref(x, (cos, sin), use_cache=True)
    show_diff("Block 前向（use_cache=True）vs 仓库", got_c, want_c, atol=1e-5)
    show_diff("present k vs 仓库", past[0], ref_past[0], atol=1e-5)
    show_diff("present v vs 仓库", past[1], ref_past[1], atol=1e-5)


def test_block_cache():
    torch.manual_seed(6)
    block = MiniMindBlock(0, MiniMindConfig()).eval()
    S = 6
    x = torch.randn(1, S, 768)
    cos, sin = precompute_freqs_cis(HEAD_DIM, S)

    full, full_past = block(x, (cos, sin), use_cache=True)
    o1, past = block(x[:, :4], (cos[:4], sin[:4]), use_cache=True)
    o2, past2 = block(x[:, 4:], (cos[4:], sin[4:]), past_key_value=past, use_cache=True)

    show_diff("Block cache 分块（后 2 位）vs 全量", o2, full[:, 4:], atol=1e-4)
    show_diff("Block 分块 present k vs 全量", past2[0], full_past[0], atol=1e-5)
    show_diff("Block 分块 present v vs 全量", past2[1], full_past[1], atol=1e-5)


def test_block_tiny():
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=1)
    ours, ref = pair_block(cfg)
    S = 4
    x = torch.randn(1, S, 128)
    cos, sin = precompute_freqs_cis(cfg.head_dim, S)
    got, _ = ours(x, (cos, sin))
    want, _ = ref(x, (cos, sin))
    show_diff("128 维 Block vs 仓库", got, want, atol=1e-5)


# ============================================================================
TESTS = [
    test_ffn_init,
    test_ffn_formula,
    test_ffn_parity,
    test_ffn_tiny,
    test_ffn_fp16,
    test_ffn_no_mutation,
    test_block_init,
    test_block_prenorm,
    test_block_parity,
    test_block_cache,
    test_block_tiny,
]

HINTS = {
    "test_ffn_init": "gate/up 是 (intermediate, hidden)，down 是 (hidden, intermediate)；无 bias",
    "test_ffn_formula": "down_proj( silu(gate_proj(x)) * up_proj(x) )，注意是逐元素乘不是加",
    "test_ffn_parity": "和仓库 FeedForward 对齐；激活用 F.silu",
    "test_ffn_tiny": "128 维 intermediate_size 自动是 448",
    "test_ffn_fp16": "输出 dtype 跟输入一致",
    "test_ffn_no_mutation": "不要原地修改输入 x",
    "test_block_init": "Block 由 self_attn / input_layernorm / post_attention_layernorm / mlp 组成",
    "test_block_prenorm": "x = x + attn(ln1(x)); x = x + mlp(ln2(x))，norm 在子层输入侧",
    "test_block_parity": "present 要原样从 attention 透传出来",
    "test_block_cache": "Block 只是把 cache 参数传给 attention，别自己缓存",
    "test_block_tiny": "128 维 Block 也要和仓库一致",
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
            print(f"FAIL  {fn.__name__:24s} 未实现（{exc}）")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:24s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:24s} ({time.time() - t0:.1f}s)")

    print("-" * 64)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D4 可以勾选 S3-05 / S3-08。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
