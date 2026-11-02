"""D14 · DPO 损失（LEARNING_TASKS: S7-07 / S7-08 / S7-09）

    L_DPO = -log σ( β · [ (logπ(y_w) - logπ(y_l)) - (logπ_ref(y_w) - logπ_ref(y_l)) ] )

验收：mymm/tests/test_dpo.py 全 PASS。
对照材料：先自己写，测试失败后再看 minimind/trainer/train_dpo.py 的 dpo_loss。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["sequence_logps", "dpo_loss"]


def sequence_logps(model, input_ids: Tensor, labels: Tensor, loss_mask: Tensor) -> Tensor:
    """返回每个样本在 mask=1 位置的 logp 之和，形状 (B,)。（已给好，不用改）

    参数：
    - input_ids: (B, T) 输入
    - labels:    (B, T) 目标 token（就是 input_ids 左移一位后的序列）
    - loss_mask: (B, T) 0/1，只用 assistant 回复段
    """
    out = model(input_ids)
    logits = out.logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * loss_mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    beta: float = 0.1,
) -> dict:
    """DPO 损失

    契约：
    - pi_logratios  = policy_chosen - policy_rejected
    - ref_logratios = ref_chosen - ref_rejected
    - logits = beta * (pi_logratios - ref_logratios)
    - loss = -logsigmoid(logits).mean()
    - chosen_rewards  = beta * (policy_chosen - ref_chosen)
        rejected_rewards = beta * (policy_rejected - ref_rejected)
        reward_margin = chosen_rewards - rejected_rewards（逐样本，不是标量）
        accuracy = (reward_margin > 0).float().mean()
    - 返回 dict：{"loss", "reward_margin", "accuracy", "chosen_rewards", "rejected_rewards"}

    自检：
    - policy 和 ref 权重完全一样时，logits 全 0，loss = -log σ(0) = ln 2 ≈ 0.693147
    - beta 越大，对偏好差距的惩罚越强
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    chosen_rewards  = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    reward_margin = chosen_rewards - rejected_rewards
    accuracy = (reward_margin > 0).float().mean()
    return {"loss": loss, 
            "reward_margin": reward_margin, 
            "accuracy": accuracy, 
            "chosen_rewards": chosen_rewards, 
            "rejected_rewards": rejected_rewards}
    # raise NotImplementedError("TODO D14-2: dpo_loss")
