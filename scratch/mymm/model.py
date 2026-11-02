"""D5 · 组装 MiniMind：Transformer + CausalLM（LEARNING_TASKS: S3-09 / S3-10 / S3-13 / S3-14）

目标：
    1) Transformer：embedding + N × MiniMindBlock + final RMSNorm + 注册 RoPE buffer
    2) CausalLM：lm_head、tie_word_embeddings、shift 后的交叉熵 loss
    3) 参数量核对：768 维 8 层 = 63,912,192（≈ 64M）

验收：mymm/tests/test_model.py 全 PASS。
对照材料：先自己写，测试失败后再看 minimind/model/model_minimind.py:201（Model）
        和 :239（ForCausalLM）。

关键约定：属性名必须让 state_dict 长成下面这样
    model.embed_tokens.weight
    model.layers.{i}.self_attn.q_proj.weight / k_proj / v_proj / o_proj
    model.layers.{i}.self_attn.q_norm.weight / k_norm.weight
    model.layers.{i}.input_layernorm.weight / post_attention_layernorm.weight
    model.layers.{i}.mlp.gate_proj.weight / up_proj.weight / down_proj.weight
    model.norm.weight
    lm_head.weight
这样 D6 才能用 strict=True 加载官方 full_sft_768.pth。
"""
from __future__ import annotations

from dataclasses import dataclass
from turtle import position

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .block import MiniMindBlock
from .norm import RMSNorm
from .rope import precompute_freqs_cis

__all__ = ["CausalLMOutput", "Transformer", "CausalLM"]


@dataclass
class CausalLMOutput:
    """和仓库 MoeCausalLMOutputWithPast 对齐的最小输出对象（直接给好，不用改）。"""

    loss: Tensor | None = None
    logits: Tensor | None = None
    past_key_values: list | None = None
    aux_loss: Tensor | None = None
    hidden_states: Tensor | None = None


class Transformer(nn.Module):
    """embedding + N 层 Block + final norm（构造与 forward 都需要你写）。"""

    def __init__(self, config):
        """契约：
        - self.config / self.vocab_size / self.num_hidden_layers 保存下来
        - self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        - self.dropout = nn.Dropout(config.dropout)
        - self.layers = nn.ModuleList([MiniMindBlock(i, config)
            for i in range(config.num_hidden_layers)])
        - self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        - 位置表：
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=config.head_dim,
                end=config.max_position_embeddings,
                rope_base=config.rope_theta,
            )
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)
            persistent=False 很重要：它不能出现在 state_dict 里，否则 D6 strict 加载会多 key。
        """
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=config.head_dim,
                end=config.max_position_embeddings,
                rope_base=config.rope_theta,
            )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # raise NotImplementedError("TODO D5-1: Transformer.__init__")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list, Tensor]:
        """参数：
        - input_ids: (B, T) long
        - attention_mask: (B, S) 或 None，S = 已缓存长度 + T
        - past_key_values: 列表，长度 = num_hidden_layers；每项 (k, v) 或 None
        - use_cache: 是否收集每层的 present

        返回 (hidden_states, presents, aux_loss)：
        - hidden_states: (B, T, hidden_size)，已过 final norm
        - presents: 长度 = num_hidden_layers 的列表，每项 (k, v) 或 None
        - aux_loss: 0 维 tensor；本阶段没有 MoE，返回 hidden_states.new_zeros(())
            （仓库对 MoE 的 aux loss 求和，这里保持接口一致即可）

        实现步骤：
        1. past_key_values 为 None 时补成 [None] * num_hidden_layers
        2. start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        3. position_embeddings = (freqs_cos[start_pos:start_pos+T],
            freqs_sin[start_pos:start_pos+T])
        4. hidden = dropout(embed_tokens(input_ids))
        5. 逐层调用 layer(hidden, position_embeddings, past, use_cache, attention_mask)，
            收集 present，hidden 更新为返回的新 hidden
        6. hidden = norm(hidden)
        7. 返回 (hidden, presents, aux_loss)

        提示：
        - RoPE 表按 start_pos 切片，这就是为什么带 cache 时只喂新 token 也能拿到正确位置；
        - 不要把 freqs_cos/freqs_sin 放进 state_dict（persistent=False）。
        """
        B, T = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * self.num_hidden_layers
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        position_embeddings = (self.freqs_cos[start_pos : start_pos + T], self.freqs_sin[start_pos : start_pos + T])
        hidden = self.dropout(self.embed_tokens(input_ids))
        presents_history = []
        for layer, past in zip(self.layers, past_key_values):
            hidden, present = layer(
                hidden,
                position_embeddings,
                past_key_value=past,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents_history.append(present)
        hidden = self.norm(hidden)
        return (hidden, presents_history, hidden.new_zeros(()))
        # raise NotImplementedError("TODO D5-2: Transformer.forward")


class CausalLM(nn.Module):
    """Transformer + lm_head + tie weights + loss（构造与 forward 都需要你写）。"""

    def __init__(self, config):
        """契约：
        - self.config = config
        - self.model = Transformer(config)
        - self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        - config.tie_word_embeddings 为 True 时：
            self.lm_head.weight = self.model.embed_tokens.weight
            必须是**同一个 Parameter 对象**（不能 clone / 不能 copy 数值），
            这样参数量只算一次、训练时梯度自动共享。
        - 初始化：tie 完之后，对 self.modules() 里所有 nn.Linear / nn.Embedding 做
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            （tie 的两个模块会各初始化一次同一个张量，没有影响；仓库用 HF 的 post_init
            做等价初始化。没有这步的话随机初始化 loss 会大得离谱，不会落在 ln(6400)≈8.76 附近）
        """
        super().__init__()
        self.config = config
        self.model = Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # raise NotImplementedError("TODO D5-3: CausalLM.__init__")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        """参数：
        - input_ids / attention_mask / past_key_values / use_cache：透传给 self.model
        - labels: (B, T) long，-100 表示不参与 loss（padding 或 prompt 部分）

        返回 CausalLMOutput（字段名固定）：
        - logits: (B, T, vocab_size)
        - loss: labels 为 None 时是 None；否则是标量
        - past_key_values / aux_loss / hidden_states：直接来自 self.model

        loss 契约（顺序即验收）：
        1. logits 取前 T-1 个位置：logits[..., :-1, :]
        2. labels 取后 T-1 个位置：labels[..., 1:]
        3. 两者 reshape 成 (B*(T-1), vocab_size) / (B*(T-1),)
        4. F.cross_entropy(..., ignore_index=-100)
        也就是“用位置 t 的 logits 预测位置 t+1 的 token”。

        提示（不是实现）：
        - 长度是 T-1 不是 T，不要忘记 shift；
        - reshape 前先 .contiguous()，少了它某些形状会报错；
        - 不要对 labels 做任何 torch.where / 替换，-100 交给 ignore_index 处理。
        """
        B,T = input_ids.shape
        hidden, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache)
        logits = self.lm_head(hidden)
        vocab_size = logits.shape[-1]
        loss = None
        if labels is not None:
            sel_logits = logits[..., :-1, :].contiguous().reshape((B*(T-1), vocab_size))
            sel_labels = labels[..., 1:].contiguous().reshape((B*(T-1),))
            loss = F.cross_entropy(sel_logits.view(-1, sel_logits.size(-1)), sel_labels.view(-1), ignore_index=-100)
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            past_key_values=past_key_values,
            hidden_states=hidden,
        )
        
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])
        # loss = None
        # if labels is not None:
        #     x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        #     loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        # return MoeCausalLMOutputWithPast(
        #     loss=loss, 
        #     aux_loss=aux_loss, 
        #     logits=logits, 
        #     past_key_values=past_key_values, 
        #     hidden_states=hidden_states)
    
        ...
        # raise NotImplementedError("TODO D5-4: CausalLM.forward")
