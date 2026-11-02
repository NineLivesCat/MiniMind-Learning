"""D15 · RL 核心公式（LEARNING_TASKS: S8-03 / S8-04 / S8-05 / S8-09）

S8 不要求你完整复现 PPO/GRPO 训练，但要把这几个公式自己写出来：
    - GAE（广义优势估计，PPO 用）
    - group advantage（GRPO：同 prompt 的 G 条回复内做归一化）
    - clipped policy loss（PPO/GRPO 共用的 clip 目标）
    - KL penalty（相对 ref 策略的 KL）

验收：mymm/tests/test_rl.py 全 PASS。
对照材料：先自己写，测试失败后再看
    minimind/trainer/train_ppo.py:140-215（GAE + clip 目标）
    minimind/trainer/train_grpo.py:121-143（group advantage + clip 目标）
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["compute_gae", "group_advantages", "clipped_policy_loss", "kl_penalty"]


def compute_gae(rewards: Tensor, values: Tensor, gamma: float = 1.0, lam: float = 0.95) -> Tensor:
    """广义优势估计（TODO D15-2）。

    参数：rewards / values 形状都是 (B, T)（T 是回复 token 数）
    契约：从 t = T-1 倒推到 0（lastgaelam 初始为 0）：
        next_value = values[:, t+1] if t+1 < T else 0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages[:, t] = lastgaelam
    返回 advantages (B, T)。

    直白理解：delta 是「单步 TD 误差」，GAE 是这些误差按 γλ 衰减的加权和。
    """
    B, T = rewards.shape
    lastgaelam = rewards.new_zeros((B, ))
    advantages = rewards.new_zeros((B, T))
    for t in reversed(range(T )):
        next_value = values[:, t+1] if t+1 < T else 0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages[:, t] = lastgaelam
    return advantages
    # raise NotImplementedError("TODO D15-2: compute_gae")


def group_advantages(rewards: Tensor, num_generations: int, eps: float = 1e-4) -> Tensor:
    """GRPO 的组内归一化优势（TODO D15-3）。

    参数：
    - rewards: (B * num_generations,)，同一个 prompt 的 G 条回复连续排列
    - num_generations: 每个 prompt 采样几条

    契约：
    - rewards.view(-1, num_generations) → 每个 prompt 一组
    - 组内 mean / std（std 用 unbiased=False，总体标准差）
    - advantages = (r - group_mean) / (group_std + eps)，再 reshape 回 (B*G,)
    - 不用 critic：GRPO 用组内相对好坏当优势
    """
    group = rewards.view(-1, num_generations)
    group_mean = group.mean(dim=1, keepdim=True)
    group_std = group.std(dim=1, unbiased=False, keepdim=True)
    advantages = (group - group_mean) / (group_std + eps)
    return advantages.reshape(-1)
    # raise NotImplementedError("TODO D15-3: group_advantages")


def clipped_policy_loss(log_ratio: Tensor, advantages: Tensor, clip_epsilon: float = 0.2,
                        mask: Tensor | None = None) -> Tensor:
    """PPO / GRPO 的 clip 策略损失（TODO D15-4）。

    参数：
    - log_ratio: (B, T) = log π(a|s) - log π_old(a|s)
    - advantages: (B, T) 或 (B, 1)（会广播）
    - mask: (B, T) 的 0/1，只在回复 token 上算 loss；None 表示全算

    契约：
    - ratio = exp(log_ratio)
    - unclipped = ratio * advantages
    - clipped   = clamp(ratio, 1-eps, 1+eps) * advantages
    - per_token = -min(unclipped, clipped)   ← 目标越大越好，所以取负做 loss
    - mask 不为 None 时返回 sum(per_token*mask)/mask.sum()，否则 per_token.mean()

    为什么这样裁剪：ratio 偏离 1 太多时不再给额外收益（也不让损失无界），
    防止一次更新把策略推得太远（PPO 的 trust region 近似）。
    """
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
    per_token = -torch.min(unclipped, clipped)
    if mask is not None:
        return torch.sum(per_token*mask)/mask.sum()
    else:
        return per_token.mean()
    # raise NotImplementedError("TODO D15-4: clipped_policy_loss")


def kl_penalty(policy_logp: Tensor, ref_logp: Tensor, mask: Tensor | None = None) -> Tensor:
    """简单 KL 估计：mean(policy_logp - ref_logp)（已给好，不用改）。

    注意这是 k1 估计，单样本方差大；仓库 GRPO 里用的是逐 token 的 logp 差再平均。
    """
    diff = policy_logp - ref_logp
    if mask is not None:
        return (diff * mask).sum() / mask.sum().clamp(min=1.0)
    return diff.mean()
