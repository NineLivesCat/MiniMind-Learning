"""D2 · RoPE 旋转位置编码（LEARNING_TASKS: S3-02 / S3-03）

目标：手写
    1) precompute_freqs_cis(dim, end, rope_base)：预计算 cos/sin 表，形状 (end, dim)
    2) apply_rotary_pos_emb(q, k, cos, sin)：把 q/k 按位置旋转

验收：mymm/tests/test_ops.py 里所有 test_rope_* 全 PASS。

对照材料：先自己写，测试失败后再看 minimind/model/model_minimind.py:62 与 :80。
关键点：MiniMind 的 cos/sin 排列是 cat([cos, cos])（前后两半相同），
        配合 rotate_half 把向量劈成前后两半旋转，不是逐维交错（interleaved）的那种写法。
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["precompute_freqs_cis", "apply_rotary_pos_emb"]


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """预计算每个位置的 cos/sin 表，返回 (cos, sin)，两者形状都是 (end, dim)。

    契约：
    - dim 必须是偶数，否则 raise ValueError（奇数时前半/后半没法对齐旋转）
    - 每个频率对 i（0 <= i < dim/2）：freq_i = 1 / rope_base ** (2i / dim)
    - 位置 t 的角度：theta[t, i] = t * freq_i
    - cos 表 = [cos(theta), cos(theta)]（前后两半各复制一份），sin 表同理；
        **不是**交错排列，这点必须和仓库一致，否则 logits 对不上
    - 返回 dtype = torch.float32，device 默认 CPU（模型里再 register_buffer 搬走）
    - rope_scaling 是 YaRN 外推用的（S2-08 再补）：现在传 None 走基础逻辑；
        传非 None 可以先 raise NotImplementedError

    自检提示（测试会验）：
    - cos[0] 全为 1，sin[0] 全为 0（位置 0 不旋转）
    - 和仓库 precompute_freqs_cis(dim, end, rope_base) 数值一致
    """
    if dim % 2 != 0:
        raise ValueError
    if rope_scaling is not None:
        raise NotImplementedError("rope_scaling not implemented")
    theta = 1 / rope_base ** (torch.arange(0, dim, 2) / dim)
    theta = torch.outer(torch.arange(end), theta)
    freq_cos = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
    freq_sin = torch.cat([torch.sin(theta), torch.sin(theta)], dim=-1)
    return freq_cos, freq_sin
    # raise NotImplementedError("TODO D2-2: precompute_freqs_cis")


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    """对 q、k 施加旋转位置编码，返回 (q_emb, k_emb)，shape 与输入相同。

    参数：
    - q: (B, T, H, D)，k: (B, T, H_kv, D)（GQA 时 H_kv < H 也可以）
    - cos/sin: (T, D)，由 precompute_freqs_cis(...)[start:start+T] 切片得到
    - unsqueeze_dim: cos/sin 插入哪个维度做广播，默认 1

    契约：
    - rotate_half(x) = cat([-x[..., D/2:], x[..., :D/2]], dim=-1)
        即后半段取负放到前面、前半段放到后面
    - q_emb = q * cos + rotate_half(q) * sin；k_emb 同理
    - 返回 dtype 与 q/k 一致（cos/sin 是 float32，算完要 .to(q.dtype) 转回去）
    - 不修改输入 q、k、cos、sin

    广播细节（容易写错）：
    - cos 是 (T, D)，q 是 (B, T, H, D)。在 dim=1 插一维得到 (T, 1, D)，
        按右对齐广播正好是 (1, T, 1, D) → 每个 head 共享同一组 cos/sin。
    - 如果写成 unsqueeze(0) 得到 (1, T, D)，会和 (B, T, H, D) 的 T/H 维错位
        （除非 T 恰好等于 H），形状可能不报错但数值全错。
    """
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
    
    # raise NotImplementedError("TODO D2-3: apply_rotary_pos_emb")
