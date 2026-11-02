"""D7 · 预训练数据（LEARNING_TASKS: S4-03）

目标：把 `pretrain_*.jsonl` 里的 `text` 字段变成 `(input_ids, labels)`。
验收：mymm/tests/test_data_train.py 里 test_dataset / test_dataset_parity 通过。
对照材料：先自己写，测试失败后再看 minimind/dataset/lm_dataset.py:40（PretrainDataset）。

数据格式（learning/data/pretrain_5000.jsonl）：
    {"text": "……一段文本……"}
"""
from __future__ import annotations

import json
import torch

from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["PretrainDataset", "SFTDataset", "DPODataset"]


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """契约：
        - text = str(self.samples[index]["text"])
        - tokens = self.tokenizer(
            text, add_special_tokens=False,
            max_length=self.max_length - 2, truncation=True,
            ).input_ids
        - 拼成 [bos_token_id] + tokens + [eos_token_id]（留 2 个位置给 bos/eos）
        - 右侧补 pad_token_id 到 self.max_length
        - labels = input_ids.clone()，再把 **pad 位置** 全部设成 -100
        - 返回两个 1D 的 torch.long tensor

        注意：
        - MiniMind 的 tokenizer 里 bos=1、eos=2、pad=0，不要按 0/1/2 的常识写死；
        - add_special_tokens=False + 手工加 bos/eos 是故意的（避免 tokenizer 自动加错）；
        - 只有 pad 位置是 -100，正文里即使出现 eos id 也照常算 loss。
        """
        text = str(self.samples[index]["text"])
        tokens = self.tokenizer(
            text, 
            add_special_tokens=False,
            max_length=self.max_length - 2, 
            truncation=True,
            ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
        # raise NotImplementedError("TODO D7-1: PretrainDataset.__getitem__")


class SFTDataset(Dataset):
    """D10 · SFT 数据：只对 assistant 回复算 loss（TODO D10-1 实现 __getitem__）。

    数据格式（learning/data/sft_2000.jsonl）：
        {"conversations": [{"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}, ...]}
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line)["conversations"])
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """契约：
        - conversations = self.samples[index]
        - tool 数据预处理（对齐仓库 create_chat_prompt，lm_dataset.py:74-89）：
            tool_calls 在数据里是 JSON 字符串，模板要求 list[dict]，必须先 json.loads，
            否则含 tool_calls 的样本会触发
            TypeError: Object of type Undefined is not JSON serializable；
            system 消息里的 tools 同理解析后作为 tools= 传给模板
        - prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools)
        - input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
            右侧补 pad_token_id 到 self.max_length
        - labels 先全部填 -100，然后从前往后扫描 input_ids：
            遇到 self.bos_id（"<|im_start|>assistant\n"）时，从它**之后**开始，
            到下一个 self.eos_id（"<|im_end|>\n"）为止（含 eos 段）的 label 复制 input_ids，
            其余部分（system / user / 空 <think>）保持 -100
        - 返回两个 (max_length,) 的 torch.long tensor

        提示：
        - 仓库实现见 minimind/dataset/lm_dataset.py:91（generate_labels），逻辑一致；
        - 截断可能把某个 eos 段切掉，扫描时注意别越界；
        - 先把 input_ids 补到 max_length 再生成 labels，两者长度才能对齐。
        """
        conversations = self.samples[index]
        # tool 数据预处理（对齐仓库 create_chat_prompt，lm_dataset.py:74-89）：
        # tool_calls 是 JSON 字符串，模板按 list[dict] 迭代，不解析会触发
        # TypeError: Object of type Undefined is not JSON serializable
        messages, tools = [], None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = [-100] * self.max_length
        idx = 0
        while idx < len(labels):
            if input_ids[idx:idx + len(self.bos_id)] == self.bos_id:
                start = idx + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for i in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[i] = input_ids[i]
                idx = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                idx += 1
        return (torch.tensor(input_ids,dtype=torch.long), torch.tensor(labels,dtype=torch.long))

        # raise NotImplementedError("TODO D10-1: SFTDataset.__getitem__")


class DPODataset(Dataset):
    """D14 · DPO 偏好数据（TODO D14-1 实现 __getitem__）。

    数据格式（learning/data/dpo_500.jsonl）：
        {"chosen":   [{"role": ..., "content": ...}, ...],
        "rejected": [{"role": ..., "content": ...}, ...]}
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_and_mask(self, messages):
        """渲染对话 → (input_ids, loss_mask)，mask=1 的位置是 assistant 回复段（已给好，不用改）。"""
        # 与 SFTDataset 相同的 tool 数据预处理：tool_calls 可能是 JSON 字符串
        cleaned, tools = [], None
        for message in messages:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            cleaned.append(message)
        prompt = self.tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=False, tools=tools)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.pad_id] * (self.max_length - len(input_ids))
        loss_mask = [0] * self.max_length
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return input_ids, loss_mask

    def __getitem__(self, index: int) -> dict:
        """契约：
        - sample = self.samples[index]
        - chosen_ids, chosen_mask = self._encode_and_mask(sample["chosen"])
            rejected_ids, rejected_mask = self._encode_and_mask(sample["rejected"])
        - 三者都左移一位做 next-token 预测：
            x = input_ids[:-1]；y = input_ids[1:]；mask = loss_mask[1:]
        - 返回 dict（键固定，全部 torch.long，长度 max_length - 1）：
            x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected
        提示：
        - mask 用 0/1，不是 -100；dpo_loss 里用 mask 加权求 logp 之和；
        - chosen / rejected 分别独立编码，长度对齐 max_length。
        """
        sample = self.samples[index]
        chosen_ids, chosen_mask = self._encode_and_mask(sample["chosen"])
        rejected_ids, rejected_mask = self._encode_and_mask(sample["rejected"])
        return {
            "x_chosen":    torch.tensor(chosen_ids[:-1],    dtype=torch.long),
            "y_chosen":    torch.tensor(chosen_ids[1:],     dtype=torch.long),
            "mask_chosen": torch.tensor(chosen_mask[1:],    dtype=torch.long),
            "x_rejected":  torch.tensor(rejected_ids[:-1],  dtype=torch.long),
            "y_rejected":  torch.tensor(rejected_ids[1:],   dtype=torch.long),
            "mask_rejected": torch.tensor(rejected_mask[1:], dtype=torch.long),
        }
        # raise NotImplementedError("TODO D14-1: DPODataset.__getitem__")
