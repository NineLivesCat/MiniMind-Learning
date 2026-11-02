"""D16 · VLM 数据集（LEARNING_TASKS: S9-07 / S9-08）

数据格式（parquet）：两列
    conversations: JSON 字符串，形如
        [{"role": "user", "content": "Describe this image.<image>"},
        {"role": "assistant", "content": "..."}]
    image_bytes: JPEG 二进制（可能是一张，也可能是多张的 list）

处理流程：
    <image> → 64 个 <|image_pad|> → tokenize → 补 pad
    labels：只有 assistant 回复段算 loss
    image_bytes → SiglipImageProcessor → pixel_values (3, 256, 256)

验收：mymm/tests/test_vlm.py 里的 dataset 测试。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .vlm import MiniMindVLM

__all__ = ["VLMDataset"]


class VLMDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        processor,
        max_length: int = 512,
        image_special_token: str = "<|image_pad|>",
        image_token_len: int = 64,
    ):
        from datasets import load_dataset

        self.dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_special_token = image_special_token * image_token_len
        self.image_pad_id = tokenizer.convert_tokens_to_ids(image_special_token)
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def create_chat_prompt(self, conversations) -> str:
        """把 <image> 替换成 64 个 image_pad，再套 chat template（已给好，不用改）。"""
        messages = []
        for turn in conversations:
            content = turn.get("content", "")
            if turn.get("role") != "system":
                content = content.replace("<image>", self.image_special_token)
            messages.append({"role": turn["role"], "content": content})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def generate_labels(self, input_ids) -> list[int]:
        """assistant 段填 token id，其余 -100（已给好，不用改）。"""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), len(input_ids))):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int) -> dict:
        """契约（TODO D16-3）：
        - row = self.dataset[index]；conversations 是 JSON 字符串，先 json.loads
        - image_bytes 可能是 bytes 或 list，统一成 list
        - prompt = self.create_chat_prompt(conversations)
        - input_ids = tokenizer(prompt).input_ids[: self.max_length]，右侧补 pad_token_id
        - labels = self.generate_labels(input_ids)
        - 每张图：Image.open(io.BytesIO(b)) → MiniMindVLM.image2tensor(img, self.processor)
            → 取 ["pixel_values"] 并 torch.cat(dim=0)，得到 (num_images, 3, 256, 256)
        - 返回 dict：{"input_ids": long (max_length,), "labels": long (max_length,),
            "pixel_values": float (num_images, 3, 256, 256)}
        """
        row = self.dataset[index]
        conversations = row["conversations"]
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        
        image_bytes = row["image_bytes"]
        if isinstance(image_bytes, (bytes, bytearray)):
            image_bytes = [image_bytes]

        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        pixel_values = [
            MiniMindVLM.image2tensor(Image.open(io.BytesIO(b)), self.processor)["pixel_values"]
            for b in image_bytes]
        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else torch.zeros(0, 3, 256, 256)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }
        # raise NotImplementedError("TODO D16-3: VLMDataset.__getitem__")
