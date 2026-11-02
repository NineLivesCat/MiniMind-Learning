"""D1 · 自回归生成与采样（对应 LEARNING_TASKS: S2-02/03/07）

目标：不调用 model.generate，自己实现
    1) 温度缩放
    2) repetition penalty
    3) top-k
    4) top-p（nucleus）
    5) 贪心 / 多项式采样
    6) 带 KV cache 的逐 token 生成循环

验收：tests/test_sampling.py 全 PASS。其中 greedy_parity 要求与仓库
        MiniMindForCausalLM.generate(temperature=1.0, do_sample=False) 逐 token 完全一致。

实现顺序建议（每写一个就跑一次对应单测，别攒到最后）：
    apply_temperature → apply_top_k → apply_top_p → apply_repetition_penalty
    → pick_next_token → generate

对照材料：先自己写，测试失败后再看 minimind/model/model_minimind.py:261。
注意：仓库 generate 里是 logits / temperature，所以 temperature 不能是 0；
        贪心对齐用 do_sample=False, temperature=1.0。
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "NEG_INF",
    "apply_temperature",
    "apply_repetition_penalty",
    "apply_top_k",
    "apply_top_p",
    "pick_next_token",
    "generate",
]

NEG_INF = float("-inf")


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """1) 温度缩放：softmax(logits / T) 的等价实现。

    契约：
    - temperature > 0，否则 raise ValueError（0 会让 logits 变成 ±inf/nan）
    - temperature == 1.0 时数值不变（返回新张量即可，不要求是同一个对象）
    - 形状和 dtype 与输入一致；**不修改输入张量**
    - 作用在 softmax 之前、repetition penalty/top-k/top-p 之前
    """
    if temperature <= 0:
        raise ValueError
    else:
        return logits.clone() / temperature


def apply_repetition_penalty(logits: Tensor, previous_ids: Tensor, penalty: float) -> Tensor:
    """2) 重复惩罚：对已出现过的 token 按符号缩放（与仓库 generate 同语义）。

    参数：
    - logits: (B, V) 当前步的原始 logits
    - previous_ids: (B, T) 当前全部 input_ids（prompt + 已生成）
    - penalty: > 1 惩罚重复；1.0 表示不处理

    契约：
    - 对第 b 行、previous_ids[b] 中出现过的每个 token id v：
        logits[b, v] >  0  →  logits[b, v] / penalty
        logits[b, v] <= 0  →  logits[b, v] * penalty
    - 未出现过的 token 不动；同一 token 重复出现多次也只处理一次
    - 每行按自己的 previous_ids 计算，batch 之间互不影响
    - 返回新张量,不修改输入
    """
    if penalty == 1.0:
        return logits.clone()
    seen = logits.gather(1, previous_ids)            
    score = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits.clone().scatter_(1, previous_ids, score)
        


def apply_top_k(logits: Tensor, k: int) -> Tensor:
    """3) top-k：每行只保留最大的 k 个 logits，其余置为 -inf。

    契约：
    - k <= 0 或 k >= vocab_size 时原样返回（不截断）——对应仓库的 `if top_k > 0`
    - 以“每行第 k 大的值”为界，>= 该值的全部保留（并列多留不算错）
    - 返回新张量，不修改输入
    """
    result = logits.clone()
    if k <= 0 or k >= logits.size(1):
        return result
    result[result < torch.topk(result, k).values[...,-1, None]] = -float('inf')
    return result


def apply_top_p(logits: Tensor, p: float) -> Tensor:
    """4) top-p / nucleus：保留累计概率刚达到 p 的最短前缀，其余置 -inf。

    契约（与仓库 generate 里“排序 + cumsum + mask 右移”写法等价）：
    - 对每行 softmax 后的概率降序累计，保留到第一个 cumsum >= p 的位置（含该位置）
    - **至少保留 1 个 token**（p <= 0 时等价于 top-1）
    - p >= 1.0 时直接原样返回，不做数值截断
    - 返回新张量，**不修改输入**
    """
    result = logits.clone()
    if p >= 1.0 :
        return result
    sorted_logits, stored_indices = torch.sort(result, descending=True)
    mask = torch.cumsum(torch.softmax(sorted_logits, dim=1), dim=1) > p
    mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
    result[mask.scatter(1,stored_indices,mask)] = -float("inf")
    return result


def pick_next_token(logits: Tensor, do_sample: bool) -> Tensor:
    """5) 从 (B, V) 的 logits 选出 (B, 1) 的下一个 token。

    契约：
    - do_sample=False → argmax（贪心）
    - do_sample=True  → softmax 后 torch.multinomial(..., num_samples=1)
    - 调用方保证 logits 已做完 temperature / penalty / top-k / top-p
    - 返回 dtype=torch.long、形状 (B, 1)、与 logits 同 device
    """
    if do_sample == False:
        return torch.argmax(logits, dim=1, keepdim=True)
    else:
        return torch.multinomial(torch.softmax(logits,dim=1),num_samples=1)


@torch.inference_mode()
def generate(
    model,
    input_ids: Tensor,
    *,
    max_new_tokens: int = 8192,
    temperature: float = 0.85,
    top_k: int = 50,
    top_p: float = 0.85,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    eos_token_id: int = 2,
    use_cache: bool = True,
    attention_mask: Tensor | None = None,
    pixel_values: Tensor | None = None,
) -> Tensor:
    """6) 自回归生成，返回 **prompt + 新 token** 的完整 (B, T+N) 序列。

    模型约定：传入的 model 兼容仓库 MiniMindForCausalLM，调用方式
        outputs = model(cur_ids, attention_mask=mask, past_key_values=past,
                        use_cache=use_cache)
        logits = outputs.logits            # (B, 当前输入长度, V)
        past   = outputs.past_key_values   # use_cache=True 时每层 (k, v)

    契约（顺序即验收）：
    1. 每步先算 past_len = past[0][0].shape[1]（无 cache 或首步为 0），
        只把 input_ids[:, past_len:] 喂给模型；
    2. logits 取最后一步 (B, V)，依次执行：
        temperature → repetition penalty（用当前完整 input_ids）
        → top_k → top_p；
    3. do_sample=False 用 argmax，否则多项采样；
    4. 维护 finished: (B,) bool。已结束的行从下一步起强制填 eos；
        finished.all() 为 True 时提前 break；
    5. attention_mask 不为 None 时，每步在末尾拼一个 1（长度与 input_ids 同步）；
    6. use_cache=False 时每步喂完整序列、past_key_values 传 None；
    7. 生成满 max_new_tokens 或全部结束即停；返回 torch.cat 后的 input_ids。

    额外要求：
    - 不允许调用 model.generate（测试会临时把它替换成抛异常的函数来检测）；
    - 支持 batch > 1（不同行可以先后结束）；
    - 不要假设词表大小 / 设备 / dtype（测试里可能是 half）。

    提示（不是实现）：
    - 先让 128 维小模型跑通，再用 768 官方权重做严格对齐；
    - 第一步与后续步的 attention_mask 长度不同，别写死；
    - 返回的是完整序列，不是只有新 token。
    """
    
    past_len = None
    past = None
    batch_size = input_ids.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    extra = {"pixel_values": pixel_values} if pixel_values is not None else {}
    
    for _ in range(max_new_tokens):
        ## 1
        past_len = past[0][0].shape[1] if (past_len is not None) and (use_cache == True) else 0
        output = model(
            input_ids=input_ids[:, past_len:], 
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=use_cache,
            **extra,
            )
        
        ## 2
        logits = apply_temperature(output.logits[:,-1, :], temperature=temperature)
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        logits = apply_top_k(logits, top_k)
        logits = apply_top_p(logits, top_p)
        
        ## 3
        next_token = pick_next_token(logits, do_sample)
        
        ## 4
        if eos_token_id is not None:
            next_token = torch.where(
                finished.unsqueeze(-1),
                next_token.new_full((batch_size,1), eos_token_id),
                next_token
            )
        
        # ## 5
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size,1))], dim=1)
        
        # ## 6
        input_ids = torch.cat([input_ids, next_token], dim=1)
        past = output.past_key_values if use_cache else None

        # ## 7
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all():
                break

    return input_ids
        
    # raise NotImplementedError("TODO D1-6: generate")
