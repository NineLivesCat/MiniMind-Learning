"""D2 · RMSNorm（LEARNING_TASKS: S3-01）

目标：手写 MiniMind 用的 RMSNorm，并搞清它和 LayerNorm 的差别。

验收：mymm/tests/test_ops.py 里所有 test_rmsnorm_* 全 PASS。

对照材料：先自己写，测试失败后再看 minimind/model/model_minimind.py:50。
提示：RMSNorm 只做「除以均方根 + 乘一个可学习缩放」，没有减均值、没有 bias。
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight。

    契约：
    - 构造：保存 self.eps；self.weight = nn.Parameter(torch.ones(dim))
    - 参数：只有 weight，形状 (dim,)，**没有 bias**（测试会检查 named_parameters 只有一项）
    - 前向：先转 float32 计算 RMS（数值稳定 + 和仓库一致），再乘 weight，
        最后用 .type_as(x) 转回输入 dtype
    - 输入 (..., dim)，返回同 shape；不修改输入张量
    - 对全零输入输出也为 0（0 / sqrt(eps) = 0）

    和 LayerNorm 的区别（自己写完后能一句话说清）：
    - 不减均值，所以对整体平移不不变；
    - 只有 gamma，没有 beta；参数量 dim vs 2*dim；
    - 通常更省、也更稳（LLaMA 系）。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        input = x.clone().float()
        return (input * torch.rsqrt((input ** 2).mean(-1, keepdim=True) + self.eps) * self.weight).type_as(x)