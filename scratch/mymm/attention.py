"""D3 · GQA 因果注意力（LEARNING_TASKS: S3-04 / S3-06 / S3-07）

目标：
    1) repeat_kv：把 n_kv 个 KV head 按「相邻复制」扩成 n_heads 个
    2) Attention.forward：
        - q/k/v 投影 → head 拆分 → q_norm/k_norm → RoPE
        - 可选的 KV cache 拼接
        - 因果 mask + softmax + @v（先手写）
        - 满足条件时用 F.scaled_dot_product_attention 快路径
    3) 输出过 o_proj，返回 (output, present_kv)

验收：mymm/tests/test_attention.py 全 PASS。
对照材料：先自己写，测试失败后再看 minimind/model/model_minimind.py:86（repeat_kv）
        和 :91-134（Attention）；官方权重 strict 对齐是后续 D6 的事，这里只对齐仓库实现。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn, softmax

from .norm import RMSNorm
from .rope import apply_rotary_pos_emb

__all__ = ["repeat_kv", "Attention"]


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """(B, T, H_kv, D) → (B, T, H_kv * n_rep, D)，每个 KV head 连续复制 n_rep 份。

    契约：
    - n_rep == 1 时原样返回（返回输入本身或 clone 均可，数值必须相同）
    - 复制顺序：输出第 i 组（i*n_rep 到 i*n_rep+n_rep-1）都等于输入第 i 个 head
        —— 用 expand(...).reshape(...) 或 repeat_interleave 实现；
        **不能**用 torch.repeat，它会变成「先铺一遍所有 head 再铺一遍」的交错顺序
    - 不修改输入

    形状示例：x (2, 5, 4, 96)，n_rep=2 → (2, 5, 8, 96)，
    其中输出 head 0、1 都来自输入 head 0，2、3 来自 head 1……
    """
    if n_rep == 1: return x.clone()
    return torch.repeat_interleave(x, n_rep ,dim=2)
    # raise NotImplementedError("TODO D3-1: repeat_kv")


class Attention(nn.Module):
    """GQA + RoPE + KV cache 的因果自注意力（构造部分已给，只需实现 forward）。"""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention") and config.flash_attn

    def forward(
        self,
        x: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """参数：
        - x: (B, T, hidden_size)，当前步输入；带 cache 时 T 可能为 1
        - position_embeddings: (cos, sin)，各 (T, head_dim)，由 precompute_freqs_cis 切片
        - past_key_value: 上一轮的 (k, v)，各 (B, T_past, n_kv, head_dim)
        - use_cache: 是否返回拼接后的 (k, v) 供下一步使用
        - attention_mask: (B, S)，1 保留、0 屏蔽；S = T_past + T（key 的总长度）

        返回 (output, present_key_value)：
        - output: (B, T, hidden_size)
        - present_key_value: use_cache=True 时为（RoPE 之后、未 repeat 的 k, v），
            各 (B, T_past + T, n_kv, head_dim)；False 时为 None

        实现步骤（顺序即契约）：
        1. q/k/v 投影 → view 成 (B, T, n_heads, head_dim) / (B, T, n_kv, head_dim)
        2. q、k 分别过 q_norm / k_norm（RMSNorm 作用在最后一维 head_dim 上）
        3. q、k 过 RoPE：apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        4. past_key_value 不为 None 时，把 k/v 沿 seq 维(dim=1) 拼到 past 后面；
            present = (k, v) if use_cache else None（注意是 RoPE 后的张量）
        5. 转成 (B, n_heads, T, D) / (B, n_kv, S, D)；k、v 用 repeat_kv 扩到 n_heads
        6. 注意力：
            a. 快路径：self.flash 为 True、T > 1、past_key_value is None，
                且 attention_mask 为 None 或全 1 时：
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
            b. 否则手写：
                    scores = (q @ k.transpose(-2, -1)).float() / sqrt(head_dim)
                    causal mask：只加在 scores 的最后 T 列上
                        scores[..., -T:] += triu(ones(T, T), diagonal=1) * -inf
                    attention_mask 不为 None 时再叠加：
                        scores += (1 - attention_mask)[:, None, None, :] * -1e9
                    attn = softmax(scores, -1).type_as(q)
                    output = attn_dropout(attn) @ v
        7. output.transpose(1, 2).reshape(B, T, n_heads * head_dim) → o_proj → resid_dropout
        8. 返回 (output, present)

        细节提醒：
        - 手写 mask 时 -T: 指「当前这一段的最后 T 列」；没有 cache 时 S == T，
            有 cache 时前面 S-T 列都是历史 key，不能 mask。
        - softmax 用 float32 算（仓库这么做，避免 fp16 溢出），算完 .type_as(q)。
        - repeat_kv 只用于算 scores；present 里存的必须是未 repeat 的 k/v。
        - 不修改输入 x、cos/sin、past_key_value。
        """
        
        B, T, hidden_size = x.shape
        # 1
        q = self.q_proj(x).view(B, T, self.n_local_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_local_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_local_kv_heads, self.head_dim)
        # 2
        q, k = self.q_norm(q), self.k_norm(k)
        # 3 
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        # 4
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        present = (k,v) if use_cache else None
        # 5
        q, k, v = (q.transpose(1, 2), repeat_kv(k, self.n_rep).transpose(1, 2), repeat_kv(v, self.n_rep).transpose(1, 2))
        # 6
        if ((self.flash == True) and 
            (T > 1) and 
            (past_key_value is None) and 
            (attention_mask is None or torch.all(attention_mask == 1))):
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)).float() / torch.sqrt(torch.tensor(self.head_dim))
            if (self.is_causal == True):
                scores[:,:,:,-T:] += torch.full((T, T), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1 - attention_mask)[:, None, None, :] * -1e9
            attn = torch.softmax(scores, -1).type_as(q)
            output = self.attn_dropout(attn) @ v
        # 7
        result = output.transpose(1, 2).reshape(B, T, self.n_local_heads * self.head_dim)
        result = self.resid_dropout(self.o_proj(result))
        # 8
        return (result, present)

        # raise NotImplementedError("TODO D3-2: Attention.forward")
