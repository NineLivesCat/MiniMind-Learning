"""D4 · pre-norm Transformer Block（LEARNING_TASKS: S3-08）

目标：把 RMSNorm + Attention + FeedForward 按 pre-norm 残差结构拼成一层。

    x = x + attn(input_layernorm(x))
    x = x + mlp(post_attention_layernorm(x))

验收：mymm/tests/test_ffn_block.py 里所有 test_block_* 全 PASS。
对照材料：先自己写，测试失败后再看 minimind/model/model_minimind.py:183。
关键点：
    - pre-norm：norm 放在子层**输入侧**，残差从子层**输入**直连到输出；
    - block 返回 (hidden_states, present_key_value)，present 直接来自 attention；
    - 768 维每层参数 = attention 1,769,664 + FFN 5,603,328 + 两个 RMSNorm 1,536
        = 7,374,528（D6 手算模型总参数量时会用到）。
"""
from __future__ import annotations

from torch import Tensor, nn

from .attention import Attention
from .ffn import FeedForward
from .norm import RMSNorm

__all__ = ["MiniMindBlock"]


class MiniMindBlock(nn.Module):
    """构造部分已给，只需实现 forward。"""

    def __init__(self, layer_id: int, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """参数：
        - hidden_states: (B, T, hidden_size)
        - position_embeddings: (cos, sin)，各 (T, head_dim)
        - 其余同 Attention.forward

        返回 (hidden_states, present_key_value)。

        实现步骤（顺序即契约）：
        1. 残差直连：residual = hidden_states
        2. attn_out, present = self_attn(self.input_layernorm(hidden_states),
                                        position_embeddings, past_key_value,
                                        use_cache, attention_mask)
        3. hidden_states = residual + attn_out
        4. hidden_states = hidden_states + mlp(post_attention_layernorm(hidden_states))
        5. 返回 (hidden_states, present)

        提示（不是实现）：
        - 两个 RMSNorm 都在子层输入侧；不要写成 norm(x + sublayer(x))（那是 post-norm）；
        - attention 的输出已经过 o_proj + dropout，这里只负责残差相加；
        - present 原样透传给上层，不要做任何加工。
        """
        residual = hidden_states
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = residual + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return (hidden_states, present)
        # raise NotImplementedError("TODO D4-2: MiniMindBlock.forward")
