"""D2 验收测试：RMSNorm + RoPE

运行全部：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_ops.py

只跑某几项（名字子串匹配）：
    python mymm\\tests\\test_ops.py rmsnorm
    python mymm\\tests\\test_ops.py rope_table rope_relative

失败时打印「期望值 / 实际值 / 首个不一致位置」；实现前全部 FAIL（NotImplementedError）属正常。
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

from mymm.norm import RMSNorm  # noqa: E402
from mymm.rope import apply_rotary_pos_emb, precompute_freqs_cis  # noqa: E402

# 仓库实现作为参考答案（只 import，不改仓库代码）
from model.model_minimind import RMSNorm as RefRMSNorm  # noqa: E402
from model.model_minimind import apply_rotary_pos_emb as ref_apply_rotary  # noqa: E402
from model.model_minimind import precompute_freqs_cis as ref_precompute_freqs  # noqa: E402


# ============================================================================
# 诊断工具：不对才打印
# ============================================================================
def fmt(x: float) -> str:
    return f"{x:.6g}"


def show_diff(label: str, got, want, atol: float = 1e-6):
    g = got.detach().to(torch.float32).cpu() if torch.is_tensor(got) else torch.tensor(got, dtype=torch.float32)
    w = want.detach().to(torch.float32).cpu() if torch.is_tensor(want) else torch.tensor(want, dtype=torch.float32)

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
    print(f"    │  不一致 {int((diff > atol).sum())} 处，前 {min(8, len(bad))} 个：")
    for idx in bad[:8]:
        key = tuple(idx.tolist())
        print(f"    │    位置 {key}: 实际={fmt(g[key].item())}  期望={fmt(w[key].item())}  差={diff[key].item():.3e}")
    print("    └─")
    raise AssertionError(f"{label}: 数值不一致 (max|diff|={maxd:.3e})")


# ============================================================================
# RMSNorm（S3-01）
# ============================================================================
def test_rmsnorm_params():
    layer = RMSNorm(16)
    params = dict(layer.named_parameters())
    if set(params) != {"weight"}:
        print(f"    named_parameters = {list(params)}（期望只有 weight，没有 bias）")
        raise AssertionError("RMSNorm 应该只有 weight 一个参数")
    if tuple(params["weight"].shape) != (16,):
        print(f"    weight.shape = {tuple(params['weight'].shape)}，期望 (16,)")
        raise AssertionError("weight 形状不对")
    if not torch.allclose(params["weight"], torch.ones(16)):
        raise AssertionError("weight 初始值应为全 1")


def test_rmsnorm_formula():
    torch.manual_seed(0)
    layer = RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(16))
    x = torch.randn(2, 5, 16) * 3.0
    got = layer(x)

    xf = x.float()
    want = (layer.weight.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6))).to(x.dtype)
    show_diff("RMSNorm 前向", got, want)

    out_shape = tuple(got.shape)
    if out_shape != (2, 5, 16):
        print(f"    输出 shape={out_shape}，期望 (2, 5, 16)")
        raise AssertionError("RMSNorm 输出形状不对")


def test_rmsnorm_match_repo():
    torch.manual_seed(1)
    for dtype in (torch.float32, torch.float16):
        x = (torch.randn(2, 7, 96) * 2.0).to(dtype)
        w = torch.randn(96)
        ours, ref = RMSNorm(96, 1e-6), RefRMSNorm(96, 1e-6)
        with torch.no_grad():
            ours.weight.copy_(w)
            ref.weight.copy_(w)
        tol = 1e-6 if dtype == torch.float32 else 1e-5
        show_diff(f"vs 仓库 RMSNorm ({dtype})", ours(x), ref(x), atol=tol)


def test_rmsnorm_stability():
    layer = RMSNorm(8)
    big = torch.tensor([[1e4, -1e4, 3e4, 0.0, -2e4, 1e4, 0.0, 5e3]], dtype=torch.float32)
    out = layer(big)
    if not torch.isfinite(out).all():
        print(f"    大数输入输出含 inf/nan: {out}")
        raise AssertionError("RMSNorm 大数输入不稳定")

    zero = layer(torch.zeros(1, 4, 8))
    show_diff("全零输入输出", zero, torch.zeros_like(zero), atol=0)


def test_rmsnorm_input_not_modified():
    layer = RMSNorm(8)
    x = torch.randn(2, 3, 8)
    before = x.clone()
    layer(x)
    show_diff("RMSNorm 输入是否被修改", x, before, atol=0)


# ============================================================================
# RoPE 预计算表（S3-02）
# ============================================================================
def test_rope_table():
    dim, end, base = 96, 32, 1e6
    cos, sin = precompute_freqs_cis(dim, end, base)

    for name, t in (("cos", cos), ("sin", sin)):
        if tuple(t.shape) != (end, dim):
            print(f"    {name}.shape={tuple(t.shape)}，期望 ({end}, {dim})")
            raise AssertionError(f"{name} 表形状不对")
        if t.dtype != torch.float32:
            print(f"    {name}.dtype={t.dtype}，期望 torch.float32")
            raise AssertionError(f"{name} 表 dtype 不对")

    show_diff("位置 0 的 cos 应全 1", cos[0], torch.ones(dim), atol=1e-6)
    show_diff("位置 0 的 sin 应全 0", sin[0], torch.zeros(dim), atol=1e-6)

    # 前后两半必须相同（rotate_half 的排列方式）
    show_diff("cos 前后两半相同", cos[:, : dim // 2], cos[:, dim // 2:], atol=0)
    show_diff("sin 前后两半相同", sin[:, : dim // 2], sin[:, dim // 2:], atol=0)

    # 直接按公式独立算一遍，检查频率/角度/排列
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    theta = torch.outer(torch.arange(end).float(), freqs)
    show_diff("cos 表（独立公式）", cos, torch.cat([theta.cos(), theta.cos()], dim=-1))
    show_diff("sin 表（独立公式）", sin, torch.cat([theta.sin(), theta.sin()], dim=-1))


def test_rope_match_repo():
    for dim in (16, 96):
        for base in (1e4, 1e6):
            got_cos, got_sin = precompute_freqs_cis(dim, 64, base)
            want_cos, want_sin = ref_precompute_freqs(dim, end=64, rope_base=base, rope_scaling=None)
            show_diff(f"cos vs 仓库 dim={dim} base={base:g}", got_cos, want_cos, atol=1e-6)
            show_diff(f"sin vs 仓库 dim={dim} base={base:g}", got_sin, want_sin, atol=1e-6)


def test_rope_odd_dim():
    for dim in (7, 33):
        try:
            precompute_freqs_cis(dim, 8)
        except ValueError:
            continue
        raise AssertionError(f"dim={dim} 是奇数，应 raise ValueError")


# ============================================================================
# RoPE 应用（S3-03）
# ============================================================================
def test_rope_relative():
    """同一对内容向量在不同位置的 q·k 只依赖相对距离：shift 之后点积不变。

    注意必须让每个位置的 q/k 内容相同（expand），否则比较的是不同内容向量的点积，
    RoPE 的相对性无从谈起（这是这个概念最常见的误解）。
    """
    torch.manual_seed(0)
    T, H, D = 16, 2, 16
    q = torch.randn(1, 1, H, D).expand(-1, T, -1, -1)
    k = torch.randn(1, 1, H, D).expand(-1, T, -1, -1)
    cos, sin = precompute_freqs_cis(D, T)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    scores = torch.einsum("bthd,bshd->bhts", q_rot, k_rot)

    for off in (1, 3, 5):
        a = scores[:, :, : T - off, : T - off]
        b = scores[:, :, off:, off:]
        show_diff(f"相对位置性质 offset={off}", a, b, atol=1e-4)


def test_rope_apply_shape_dtype():
    torch.manual_seed(2)
    B, T, H, Hkv, D = 2, 5, 8, 4, 96
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, Hkv, D)
    cos, sin = precompute_freqs_cis(D, T)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    if tuple(q_rot.shape) != (B, T, H, D) or tuple(k_rot.shape) != (B, T, Hkv, D):
        print(f"    q_rot.shape={tuple(q_rot.shape)}  k_rot.shape={tuple(k_rot.shape)}")
        raise AssertionError("输出 shape 应与输入一致（GQA 时 k 的 head 数可以更少）")
    if q_rot.dtype != q.dtype or k_rot.dtype != k.dtype:
        print(f"    dtype: q={q.dtype} q_rot={q_rot.dtype} | k={k.dtype} k_rot={k_rot.dtype}")
        raise AssertionError("输出 dtype 应与输入一致")

    qh, kh = q.half(), k.half()
    qh_rot, kh_rot = apply_rotary_pos_emb(qh, kh, cos, sin)
    if qh_rot.dtype != torch.float16 or kh_rot.dtype != torch.float16:
        print(f"    half 输入得到 dtype: {qh_rot.dtype} / {kh_rot.dtype}")
        raise AssertionError("fp16 输入应返回 fp16（cos/sin 算完要 type_as 回去）")


def test_rope_apply_match_repo():
    torch.manual_seed(3)
    for dtype in (torch.float32, torch.float16):
        B, T, H, Hkv, D = 2, 7, 8, 4, 96
        q = torch.randn(B, T, H, D).to(dtype)
        k = torch.randn(B, T, Hkv, D).to(dtype)
        cos, sin = precompute_freqs_cis(D, T)
        got_q, got_k = apply_rotary_pos_emb(q, k, cos, sin)
        want_q, want_k = ref_apply_rotary(q, k, cos, sin)
        tol = 1e-6 if dtype == torch.float32 else 1e-3
        show_diff(f"q 旋转 vs 仓库 ({dtype})", got_q, want_q, atol=tol)
        show_diff(f"k 旋转 vs 仓库 ({dtype})", got_k, want_k, atol=tol)


def test_rope_positions_matter():
    """同一批 q/k 用不同位置段的表，结果必须不同（否则说明 cos/sin 没用上）。"""
    torch.manual_seed(4)
    T, H, D = 6, 1, 16
    q = torch.randn(1, T, H, D)
    cos, sin = precompute_freqs_cis(D, 32)
    a, _ = apply_rotary_pos_emb(q, q, cos[:T], sin[:T])
    b, _ = apply_rotary_pos_emb(q, q, cos[5 : 5 + T], sin[5 : 5 + T])
    if torch.allclose(a, b, atol=1e-6):
        raise AssertionError("位置 0 和位置 5 的旋转结果相同：cos/sin 没有按位置生效")


def test_rope_apply_no_mutation():
    torch.manual_seed(5)
    B, T, H, Hkv, D = 2, 4, 2, 2, 16
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, Hkv, D)
    cos, sin = precompute_freqs_cis(D, T)
    q0, k0, c0, s0 = q.clone(), k.clone(), cos.clone(), sin.clone()
    apply_rotary_pos_emb(q, k, cos, sin)
    show_diff("q 是否被修改", q, q0, atol=0)
    show_diff("k 是否被修改", k, k0, atol=0)
    show_diff("cos 是否被修改", cos, c0, atol=0)
    show_diff("sin 是否被修改", sin, s0, atol=0)


# ============================================================================
TESTS = [
    test_rmsnorm_params,
    test_rmsnorm_formula,
    test_rmsnorm_match_repo,
    test_rmsnorm_stability,
    test_rmsnorm_input_not_modified,
    test_rope_table,
    test_rope_match_repo,
    test_rope_odd_dim,
    test_rope_relative,
    test_rope_apply_shape_dtype,
    test_rope_apply_match_repo,
    test_rope_positions_matter,
    test_rope_apply_no_mutation,
]

HINTS = {
    "test_rmsnorm_params": "只保留 weight（无 bias），初始化为 ones，形状 (dim,)",
    "test_rmsnorm_formula": "x.float() → x*rsqrt(mean(x^2)+eps) → *weight → type_as(x)",
    "test_rmsnorm_match_repo": "注意先转 float32 算 RMS，最后转回输入 dtype",
    "test_rmsnorm_stability": "float32 计算避免大数溢出；全零输入输出 0",
    "test_rmsnorm_input_not_modified": "不要原地改 x",
    "test_rope_table": "freq_i = 1/base^(2i/dim)；cos/sin 表 = cat([cos, cos])；位置 0 不旋转",
    "test_rope_match_repo": "与仓库公式/排列对齐，尤其 cat 两次而不是交错",
    "test_rope_odd_dim": "dim 为奇数时 raise ValueError",
    "test_rope_relative": "rotate_half = cat([-后半, 前半])；q*cos + rotate_half(q)*sin",
    "test_rope_apply_shape_dtype": "cos/sin 在 dim=1 插维广播；算完 .to(q.dtype)",
    "test_rope_apply_match_repo": "rotate_half 方向/顺序错了会在这里暴露",
    "test_rope_positions_matter": "确认用的是传入的 cos/sin，而不是自己另算/没用位置",
    "test_rope_apply_no_mutation": "不要原地改 q/k/cos/sin",
}


def main() -> int:
    filters = [a.lower() for a in sys.argv[1:] if not a.startswith("-")]
    selected = [t for t in TESTS if not filters or any(f in t.__name__.lower() for f in filters)]
    if not selected:
        print(f"没有匹配 {filters} 的测试，可选：{[t.__name__ for t in TESTS]}")
        return 2

    print(f"torch = {torch.__version__} | 运行 {len(selected)} 项")
    failed = []
    for fn in selected:
        t0 = time.time()
        try:
            fn()
        except NotImplementedError as exc:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__:32s} 未实现（{exc}）")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:32s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:32s} ({time.time() - t0:.1f}s)")

    print("-" * 64)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D2 可以勾选 S3-01 / S3-02 / S3-03。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
