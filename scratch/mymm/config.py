"""MiniMind 的配置对象（D3 之后各模块共用）。

这个文件不是练习内容，直接提供：字段名与仓库 MiniMindConfig 保持一致，
这样后面手写的 RMSNorm / RoPE / Attention / FFN / 模型可以直接拼起来，
也能和官方权重 strict 对齐。

小模型（例如 128 维 2 层）只需要改 hidden_size 和 num_hidden_layers，
head_dim / intermediate_size 会自动按公式推导：
    head_dim          = hidden_size // num_attention_heads
    intermediate_size = ceil(hidden_size * pi / 64) * 64
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MiniMindConfig:
    # 词表与层数
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8

    # attention
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int | None = None
    flash_attn: bool = True

    # FFN
    intermediate_size: int | None = None
    hidden_act: str = "silu"

    # RoPE / norm
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6

    # 其他
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    use_moe: bool = False

    # MoE（use_moe=True 时生效）
    num_experts: int = 4
    num_experts_per_tok: int = 2
    moe_intermediate_size: int | None = None
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 5e-4

    # VLM（D16 起使用；纯语言模型忽略这些字段）
    image_special_token: str = "<|image_pad|>"
    image_ids: list | None = None
    image_hidden_size: int = 768
    image_token_len: int = 64

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.intermediate_size is None:
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64
        if self.image_ids is None:
            self.image_ids = [12]
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size
