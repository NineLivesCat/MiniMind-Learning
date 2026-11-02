"""D18 · MoE（LEARNING_TASKS: S3-15 选做 / S10-06 对照实验）

MoE 层 = 一个 router + 多个 expert（每个 expert 就是原来的 FFN）：
    router(x) = softmax(gate(x))          # 每个 token 对 E 个 expert 的分数
    topk       = 取分数最高的 k 个 expert，权重归一化
    y          = Σ_{e ∈ topk} weight_e · expert_e(x)
    aux_loss   = coef · E · Σ_e (load_e · mean_score_e)   # 负载均衡，防止全挤到 1 个 expert

验收：mymm/tests/test_moe.py 全 PASS。
对照材料：先自己写，测试失败后再看 minimind/model/model_minimind.py:148（MOEFeedForward）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniMindConfig
from .ffn import FeedForward
from .model import CausalLM, CausalLMOutput

__all__ = ["MoEFFN", "apply_moe", "MoECausalLM"]


class MoEFFN(nn.Module):
    def __init__(self, config: MiniMindConfig, num_experts: int | None = None, top_k: int | None = None):
        super().__init__()
        self.config = config
        self.num_experts = num_experts or config.num_experts
        self.top_k = top_k or config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.num_experts)])
        self.aux_loss = torch.zeros(())

    def forward(self, x):
        """TODO D18-1：路由 + top-k + 加权合并 + aux loss。

        参数 x: (B, T, H)
        步骤：
        1. x_flat = x.view(-1, H)
        2. scores = F.softmax(self.gate(x_flat), dim=-1)              # (N, E)
        3. topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        4. 归一化（config.norm_topk_prob）：
            k > 1：topk_weight /= topk_weight.sum(-1, keepdim=True) + 1e-20
            k = 1：使用直通技巧 topk_weight = topk_weight - topk_weight.detach() + 1.0
            （前向权重为 1，梯度仍能通过 gate；否则 gate 无梯度）
        5. y = zeros_like(x_flat)
            对每个 expert e：
                mask = (topk_idx == e)                                # (N, k)
                token_idx = mask.any(dim=-1).nonzero().flatten()      # 选中 e 的 token
                weight = topk_weight[mask].view(-1, 1)                # 每个 token 的权重
                y.index_add_(0, token_idx, self.experts[e](x_flat[token_idx]) * weight)
        6. aux loss（训练时且 router_aux_loss_coef > 0）：
                load = F.one_hot(topk_idx, self.num_experts).float().mean(0)   # (E,) 各 expert 分到的比例
                self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * coef
            推理或 coef=0 时 self.aux_loss = 0
        7. 返回 y.view(B, T, H)
        """
        B, T, H = x.shape
        x_flat = x.view(-1, H)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.top_k > 1:
                topk_weight /= topk_weight.sum(-1, keepdim=True) + 1e-20
            else:
                topk_weight = topk_weight - topk_weight.detach() + 1.0
        y = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            mask = (topk_idx == e)                                   # (N, k)
            token_idx = mask.any(dim=-1).nonzero().flatten()         # 选中 e 的 token 行号
            weight = topk_weight[mask].view(-1, 1)               # 每个 token 对应权重
            y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))

        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(0)   # (E,) 各 expert 分到的比例
            self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = x.new_zeros(())

        return y.view(B, T, H)
        # raise NotImplementedError("TODO D18-1: MoEFFN.forward")


def apply_moe(model: CausalLM, num_experts: int = 4, top_k: int | None = None) -> CausalLM:
    """把每个 block 的 FFN 换成 MoE（已给好，不用改）。

    - 每个 expert 用原 dense FFN 的权重初始化（同一份参数复制 E 份）
    - gate 用 N(0, 0.02) 初始化
    """
    config = model.config
    for layer in model.model.layers:
        dense = layer.mlp
        moe = MoEFFN(config, num_experts=num_experts, top_k=top_k)
        with torch.no_grad():
            for expert in moe.experts:
                expert.load_state_dict(dense.state_dict())
        nn.init.normal_(moe.gate.weight, mean=0.0, std=0.02)
        param = next(dense.parameters())
        moe = moe.to(device=param.device, dtype=param.dtype)
        layer.mlp = moe
    return model


class MoECausalLM(CausalLM):
    """forward 里把各层 MoE 的 aux_loss 汇总（已给好，不用改）。"""

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, labels=None):
        batch, seq = input_ids.shape
        hidden, presents, _ = self.model(input_ids, attention_mask, past_key_values, use_cache)
        aux_loss = hidden.new_zeros(())
        for layer in self.model.layers:
            if hasattr(layer.mlp, "aux_loss"):
                aux_loss = aux_loss + layer.mlp.aux_loss
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            sel_logits = logits[:, :-1, :].contiguous().reshape(-1, logits.size(-1))
            sel_labels = labels[:, 1:].contiguous().reshape(-1)
            loss = F.cross_entropy(sel_logits, sel_labels, ignore_index=-100)
        return CausalLMOutput(
            loss=loss, logits=logits, past_key_values=presents, aux_loss=aux_loss, hidden_states=hidden
        )
