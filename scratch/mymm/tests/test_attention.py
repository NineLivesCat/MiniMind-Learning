"""D3 验收测试：repeat_kv + GQA 因果注意力

运行全部：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_attention.py

只跑某几项：
    python mymm\\tests\\test_attention.py repeat_kv
    python mymm\\tests\\test_attention.py parity causal cache

失败时打印「期望/实际/首个不一致位置」；实现前全部 FAIL（NotImplementedError）属正常。
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

from mymm.attention import Attention, repeat_kv  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.rope import precompute_freqs_cis  # noqa: E402

from model.model_minimind import Attention as RefAttention  # noqa: E402
from model.model_minimind import MiniMindConfig as RefConfig  # noqa: E402
from model.model_minimind import repeat_kv as ref_repeat_kv  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEAD_DIM = 96  # 768 / 8


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


def pair(cfg: MiniMindConfig | None = None):
    """同一份权重分别装进「你的 Attention」和「仓库 Attention」。"""
    cfg = cfg or MiniMindConfig()
    ours = Attention(cfg)
    ref = RefAttention(RefConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
    ))
    ref.load_state_dict(ours.state_dict(), strict=True)
    return ours.eval(), ref.eval()


# ============================================================================
# repeat_kv（S3-04）
# ============================================================================
def test_repeat_kv():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 3, 8)
    before = x.clone()

    for n_rep in (1, 2, 3):
        got, want = repeat_kv(x, n_rep), ref_repeat_kv(x, n_rep)
        if tuple(got.shape) != (2, 5, 3 * n_rep, 8):
            print(f"    n_rep={n_rep} 实际 shape={tuple(got.shape)}，期望 (2, 5, {3 * n_rep}, 8)")
            raise AssertionError("repeat_kv 形状不对")
        show_diff(f"repeat_kv n_rep={n_rep} vs 仓库", got, want, atol=0)

        # 复制顺序：第 i 组连续 n_rep 个 head 都等于输入第 i 个 head
        for i in range(3):
            for j in range(n_rep):
                show_diff(
                    f"n_rep={n_rep}: 输出 head {i * n_rep + j} 应等于输入 head {i}",
                    got[:, :, i * n_rep + j, :],
                    x[:, :, i, :],
                    atol=0,
                )

    show_diff("repeat_kv 不修改输入", x, before, atol=0)


# ============================================================================
# Attention 构造 / 前向（S3-06）
# ============================================================================
def test_attention_init():
    ours = Attention(MiniMindConfig())
    ref = RefAttention(RefConfig(hidden_size=768, num_hidden_layers=1))

    ours_keys = set(ours.state_dict())
    ref_keys = set(ref.state_dict())
    if ours_keys != ref_keys:
        print("    仓库有、你没有：", sorted(ref_keys - ours_keys))
        print("    你有、仓库没有：", sorted(ours_keys - ref_keys))
        raise AssertionError("state_dict 的 key 集合和仓库不一致（权重名/模块名对不上）")

    params = dict(ours.named_parameters())
    expected = {
        "q_proj.weight": (768, 768),
        "k_proj.weight": (384, 768),
        "v_proj.weight": (384, 768),
        "o_proj.weight": (768, 768),
        "q_norm.weight": (96,),
        "k_norm.weight": (96,),
    }
    for name, shape in expected.items():
        if name not in params:
            raise AssertionError(f"缺少参数 {name}")
        if tuple(params[name].shape) != shape:
            print(f"    {name}.shape={tuple(params[name].shape)}，期望 {shape}")
            raise AssertionError(f"参数 {name} 形状不对")
    biases = [n for n in params if n.endswith("bias")]
    if biases:
        print(f"    多出的 bias 参数: {biases}")
        raise AssertionError("Attention 里不应有 bias")

    total = sum(p.numel() for p in ours.parameters())
    if total != 1_769_664:
        print(f"    实际参数量={total}，期望 1_769_664（4 个投影 1,769,472 + 两个 qk norm 192）")
        raise AssertionError("Attention 参数量不对")


def test_attention_parity():
    torch.manual_seed(1)
    ours, ref = pair()
    B, S = 2, 7
    x = torch.randn(B, S, 768)
    cos, sin = precompute_freqs_cis(HEAD_DIM, S)

    got, present = ours(x, (cos, sin))
    want, ref_present = ref(x, (cos, sin))
    show_diff("attention 前向 vs 仓库", got, want, atol=1e-5)
    if present is not None or ref_present is not None:
        raise AssertionError("use_cache=False 时 present 应为 None")

    got_c, past = ours(x, (cos, sin), use_cache=True)
    want_c, ref_past = ref(x, (cos, sin), use_cache=True)
    show_diff("attention 前向（use_cache=True）vs 仓库", got_c, want_c, atol=1e-5)
    show_diff("present k vs 仓库", past[0], ref_past[0], atol=1e-5)
    show_diff("present v vs 仓库", past[1], ref_past[1], atol=1e-5)
    if tuple(past[0].shape) != (B, S, 4, HEAD_DIM):
        print(f"    present k.shape={tuple(past[0].shape)}，期望 ({B}, {S}, 4, {HEAD_DIM})（未 repeat 的 KV head）")
        raise AssertionError("present 形状不对")


def test_attention_mask():
    torch.manual_seed(1)
    ours, ref = pair()
    B, S = 2, 7
    x = torch.randn(B, S, 768)
    cos, sin = precompute_freqs_cis(HEAD_DIM, S)
    mask = torch.ones(B, S)
    mask[0, 5:] = 0.0
    mask[1, 2:] = 0.0

    got, _ = ours(x, (cos, sin), attention_mask=mask)
    want, _ = ref(x, (cos, sin), attention_mask=mask)
    show_diff("attention + mask vs 仓库", got, want, atol=1e-5)

    nomask, _ = ours(x, (cos, sin))
    if torch.allclose(got, nomask, atol=1e-6):
        raise AssertionError("加了 attention_mask 但输出和无 mask 完全一样：mask 没有生效")


def test_attention_causal():
    torch.manual_seed(2)
    ours, _ = pair()
    S = 8
    x1 = torch.randn(1, S, 768)
    x2 = x1.clone()
    x2[:, -1, :] += 0.5  # 只改最后一个位置
    cos, sin = precompute_freqs_cis(HEAD_DIM, S)

    for flash in (True, False):
        ours.flash = flash
        o1, _ = ours(x1, (cos, sin))
        o2, _ = ours(x2, (cos, sin))
        show_diff(f"flash={flash}: 未来 token 不应影响过去", o1[:, :-1], o2[:, :-1], atol=1e-5)
        if torch.allclose(o1[:, -1], o2[:, -1], atol=1e-6):
            raise AssertionError(f"flash={flash}: 改了最后一个 token，最后位置输出却没变（注意力没看到自己？）")


def test_attention_cache():
    torch.manual_seed(3)
    ours, _ = pair()
    S = 9
    x = torch.randn(1, S, 768)
    cos, sin = precompute_freqs_cis(HEAD_DIM, S)

    full_out, full_past = ours(x, (cos, sin), use_cache=True)
    o1, past = ours(x[:, :5], (cos[:5], sin[:5]), use_cache=True)
    o2, past2 = ours(x[:, 5:], (cos[5:], sin[5:]), past_key_value=past, use_cache=True)

    show_diff("cache 分块输出（后 4 个位置）vs 全量", o2, full_out[:, 5:], atol=1e-4)
    show_diff("分块后的 present k vs 全量", past2[0], full_past[0], atol=1e-5)
    show_diff("分块后的 present v vs 全量", past2[1], full_past[1], atol=1e-5)
    if tuple(past2[0].shape) != (1, S, 4, HEAD_DIM):
        print(f"    present k.shape={tuple(past2[0].shape)}，期望 (1, {S}, 4, {HEAD_DIM})")
        raise AssertionError("分块后 present 长度不对")


def test_attention_gqa_tiny():
    """128 维小模型：8 个 Q head / 4 个 KV head，head_dim=16。"""
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=1)
    ours = Attention(cfg).eval()
    B, S = 2, 4
    x = torch.randn(B, S, 128)
    cos, sin = precompute_freqs_cis(cfg.head_dim, S)
    out, past = ours(x, (cos, sin), use_cache=True)
    if tuple(out.shape) != (B, S, 128):
        print(f"    输出 shape={tuple(out.shape)}，期望 ({B}, {S}, 128)")
        raise AssertionError("GQA 输出形状不对")
    if tuple(past[0].shape) != (B, S, 4, 16):
        print(f"    present k.shape={tuple(past[0].shape)}，期望 ({B}, {S}, 4, 16)")
        raise AssertionError("GQA 的 present 形状不对")


def test_attention_fp16():
    if DEVICE != "cuda":
        print("      跳过：没有 CUDA")
        return
    torch.manual_seed(4)
    ours, ref = pair()
    ours, ref = ours.half().cuda(), ref.half().cuda()
    S = 5
    x = torch.randn(1, S, 768, device="cuda", dtype=torch.float16)
    cos, sin = precompute_freqs_cis(HEAD_DIM, S)
    cos, sin = cos.cuda(), sin.cuda()
    got, _ = ours(x, (cos, sin))
    want, _ = ref(x, (cos, sin))
    if got.dtype != torch.float16:
        print(f"    输出 dtype={got.dtype}，期望 torch.float16")
        raise AssertionError("fp16 输入应返回 fp16 输出")
    show_diff("fp16 前向 vs 仓库", got, want, atol=1e-3)


# ============================================================================
TESTS = [
    test_repeat_kv,
    test_attention_init,
    test_attention_parity,
    test_attention_mask,
    test_attention_causal,
    test_attention_cache,
    test_attention_gqa_tiny,
    test_attention_fp16,
]

HINTS = {
    "test_repeat_kv": "expand(..., n_rep, ...).reshape(...)；不要用 torch.repeat（顺序会交错）",
    "test_attention_init": "检查 state_dict 名字是否和仓库一致（q_norm/k_norm、无 bias）",
    "test_attention_parity": "顺序：投影→qk norm→RoPE→拼 cache→repeat_kv→注意力→o_proj；present 存 RoPE 后未 repeat 的 k/v",
    "test_attention_mask": "mask 为 0 的 key 要加 -1e9 再 softmax；注意 mask 长度 = key 总长度",
    "test_attention_causal": "causal mask 用 triu(..., diagonal=1)；手写路径下 -T 列偏移别写错",
    "test_attention_cache": "past_key_value 沿 dim=1 拼；有 cache 时 q 只有新 token，mask 只加在最后 T 列",
    "test_attention_gqa_tiny": "present 的 head 数应为 n_kv=4，不是 n_heads=8（repeat_kv 只用于算 scores）",
    "test_attention_fp16": "softmax 用 float32 算再 type_as 回去；输出 dtype 跟输入一致",
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
    print(f"全部 {len(selected)} 项通过，D3 可以勾选 S3-04 / S3-06 / S3-07。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
