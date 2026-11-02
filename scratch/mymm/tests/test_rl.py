"""D15 验收测试：RL 核心公式（GAE / group advantage / clip 目标 / KL）

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_rl.py

覆盖：
    1) GAE 倒推公式（手算三层序列）
    2) GRPO 组内归一化优势（组间独立、组内均值 0/标准差 1、std=0 组为 0）
    3) PPO/GRPO clipped policy loss（clip 内/外、mask）
    4) KL penalty（mean policy-ref，支持 mask）
"""
from __future__ import annotations

import math
import sys

import torch

SCRATCH_MOD = r"D:\minimind\learning\scratch"
if SCRATCH_MOD not in sys.path:
    sys.path.insert(0, SCRATCH_MOD)

from mymm.rl import clipped_policy_loss, compute_gae, group_advantages, kl_penalty  # noqa: E402


def show_diff(label: str, got, want, atol: float = 1e-6):
    g = got.detach().to(torch.float32).cpu()
    w = want.detach().to(torch.float32).cpu()
    if g.shape != w.shape:
        print(f"    ┌─ {label}：shape {tuple(g.shape)} vs {tuple(w.shape)}")
        raise AssertionError(f"{label}: shape 不一致")
    diff = (g - w).abs()
    maxd = diff.max().item() if diff.numel() else 0.0
    if maxd <= atol:
        return
    print(f"    ┌─ {label}")
    print(f"    │  max|实际-期望| = {maxd:.3e}")
    print("    └─")
    raise AssertionError(f"{label}: 数值不一致 (max|diff|={maxd:.3e})")


# ============================================================================
def test_gae():
    rewards = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([[0.5, 0.2, 0.1]])
    adv = compute_gae(rewards, values, gamma=1.0, lam=1.0)
    # t2: delta=0+0-0.1=-0.1 -> last=-0.1
    # t1: delta=0+0.1-0.2=-0.1 -> last=-0.2
    # t0: delta=1+0.2-0.5=0.7 -> last=0.5
    want = torch.tensor([[0.5, -0.2, -0.1]])
    show_diff("GAE 手算示例 (gamma=1, lam=1)", adv, want, atol=1e-6)

    rewards2 = torch.stack([rewards[0], rewards[0]])
    values2 = torch.stack([values[0], values[0]])
    adv2 = compute_gae(rewards2, values2, gamma=1.0, lam=1.0)
    show_diff("batch 两行应一致", adv2, torch.stack([want[0], want[0]]), atol=1e-6)


def test_group_advantages():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
    adv = group_advantages(rewards, num_generations=3)

    groups = rewards.view(-1, 3)
    want = ((groups - groups.mean(dim=1, keepdim=True))
            / (groups.std(dim=1, unbiased=False, keepdim=True) + 1e-4)).reshape(-1)
    show_diff("group advantage 公式", adv, want, atol=1e-6)

    first = adv[:3]
    second = adv[3:]
    if abs(first.mean().item()) > 1e-5:
        raise AssertionError("第一组优势均值应为 0")
    if abs(first.std(unbiased=False).item() - 1.0) > 1e-3:
        print(f"    第一组 std={first.std(unbiased=False).item():.4f}")
        raise AssertionError("第一组优势标准差应约为 1")
    show_diff("全相同组优势应为 0", second, torch.zeros(3), atol=1e-6)
    print(f"      first group adv={first.tolist()}")


def test_clipped_policy_loss():
    log_ratio = torch.tensor([0.0, 0.0, math.log(2.0), math.log(0.5)])
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    got = clipped_policy_loss(log_ratio, advantages, clip_epsilon=0.2)
    # ratio       = [1.0, 1.0, 2.0, 0.5]
    # unclipped   = [1.0, -1.0, 1.0, -0.25]
    # clipped     = [1.0, -1.0, 0.6, -0.4]
    # -min(...)   = [-1.0, 1.0, -0.6, 0.4]
    want = torch.tensor([-1.0, 1.0, -0.6, 0.4]).mean()
    show_diff("clip 目标（无 mask）", got, want, atol=1e-6)

    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    got_masked = clipped_policy_loss(log_ratio, advantages, clip_epsilon=0.2, mask=mask)
    want_masked = torch.tensor([-1.0, -0.6]).mean()
    show_diff("clip 目标（mask 只保留 1/3 位）", got_masked, want_masked, atol=1e-6)

    # advantages 形状 (B,1) 时广播
    adv_col = advantages.view(-1, 1)
    got_col = clipped_policy_loss(log_ratio.view(-1, 1), adv_col, clip_epsilon=0.2)
    show_diff("advantages (B,1) 广播", got_col, want, atol=1e-6)


def test_kl_penalty():
    policy = torch.tensor([[0.1, 0.2, 0.3]])
    ref = torch.tensor([[0.0, 0.0, 0.0]])
    got = kl_penalty(policy, ref)
    show_diff("KL 估计（无 mask）", got, torch.tensor(0.2), atol=1e-6)

    mask = torch.tensor([[1.0, 0.0, 1.0]])
    got_masked = kl_penalty(policy, ref, mask=mask)
    show_diff("KL 估计（mask）", got_masked, torch.tensor(0.2), atol=1e-6)


# ============================================================================
TESTS = [test_gae, test_group_advantages, test_clipped_policy_loss, test_kl_penalty]

HINTS = {
    "test_gae": "从后往前：delta = r + γV(t+1) - V(t)；lastgaelam = delta + γλ·lastgaelam",
    "test_group_advantages": "组内 (r-mean)/(std+eps)，std 用 unbiased=False",
    "test_clipped_policy_loss": "ratio=exp(log_ratio)；-min(ratio*A, clamp(ratio)*A)；mask 用 sum/sum",
    "test_kl_penalty": "mean(policy_logp - ref_logp)",
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
            print(f"PASS  {fn.__name__:24s}")

    print("-" * 60)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D15 的 RL 公式部分可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
