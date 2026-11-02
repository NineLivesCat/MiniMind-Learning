"""D4 · SwiGLU 前馈网络（LEARNING_TASKS: S3-05）

目标：
    1) FeedForward：gate_proj / up_proj / down_proj + SiLU
    2) 手算参数量并核对：3 * hidden_size * intermediate_size
        768 维：intermediate_size = ceil(768 * pi / 64) * 64 = 2432
        每层参数 = 3 * 768 * 2432 = 5,603,328

验收：mymm/tests/test_ffn_block.py 里所有 test_ffn_* 全 PASS。
对照材料：先自己写，测试失败后再看 minimind/model/model_minimind.py:136。
关键点：SwiGLU = down_proj( silu(gate_proj(x)) * up_proj(x) )，
        两个“上投影”分别算门控和值，逐元素相乘后再降维。
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn, sigmoid

__all__ = ["FeedForward"]


class FeedForward(nn.Module):
    """构造部分已给，只需实现 forward。"""

    def __init__(self, config, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """契约：
        - 输入 (..., hidden_size)，输出 (..., hidden_size)
        - out = down_proj( silu(gate_proj(x)) * up_proj(x) )
        - silu(z) = z * sigmoid(z)，可直接用 F.silu
        - 不修改输入

        提示（不是实现）：
        - 是 gate 和 up 逐元素相乘，不是相加；
        - 仓库用 ACT2FN[config.hidden_act]，本次 config.hidden_act 固定为 "silu"，
            所以 F.silu 与仓库等价；
        - 参数量 = gate + up + down = 3 * hidden_size * intermediate_size。
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        # raise NotImplementedError("TODO D4-1: FeedForward.forward")
