"""D16 · 手写 VLM（LEARNING_TASKS: S9-01 / S9-02 / S9-03 / S9-04 / S9-05）

结构：
        image → SigLIP2 vision encoder（冻结）→ 64 个 patch 特征（768 维）
        → MLP projector（768 → LLM hidden）
        → 替换输入序列里 64 个 <|image_pad|> 位置的 embedding
        → 剩下的文本走原来的 MiniMind LLM

验收：mymm/tests/test_vlm.py 全 PASS。
对照材料：先自己写，测试失败后再看 minimind-v/model/model_vlm.py（MMVisionProjector / count_vision_proj / forward）。
"""
from __future__ import annotations

import io
from tracemalloc import start
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .config import MiniMindConfig
from .model import CausalLM, CausalLMOutput

VISION_MODEL_PATH = r"D:\minimind\minimind-v\model\siglip2-base-p32-256-ve"

__all__ = [
    "VISION_MODEL_PATH",
    "MMVisionProjector",
    "MiniMindVLM",
    "freeze_vlm",
    "run_transformer_layers",
]


class MMVisionProjector(nn.Module):
    """把 768 维视觉特征映射到 LLM 的 hidden 维（已给好，不用改）。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MiniMindVLM(CausalLM):
    """LLM + SigLIP2 + projector。构造已给好，count_vision_proj / forward 需要你写。"""

    def __init__(self, config: MiniMindConfig, vision_model_path: str = VISION_MODEL_PATH):
        super().__init__(config)
        self.config = config
        self.vision_encoder, self.processor = self.get_vision_model(vision_model_path)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        """加载 SigLIP2 vision encoder + image processor，并冻结（已给好）。"""
        if not model_path:
            return None, None
        from transformers import SiglipImageProcessor, SiglipVisionModel

        model = SiglipVisionModel.from_pretrained(model_path)
        processor = SiglipImageProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        return processor(images=image, return_tensors="pt")

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if isinstance(image_inputs, torch.Tensor):
            image_inputs = {"pixel_values": image_inputs}
        elif hasattr(image_inputs, "keys"):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=None):
        """把 h 里连续的 image_pad 位置替换成视觉特征（TODO D16-1）。

        参数：
        - tokens: (B, T) long，image_pad 的 id 是 self.config.image_ids[0]（12）
        - h: (B, T, hidden) 文本 embedding
        - vision_tensors: (B, 64, hidden) 或 (B, num_images, 64, hidden)
        - seqlen: 截断长度，默认 h.size(1)

        契约：
        - vision_tensors 是 3 维时先 unsqueeze(1) 变成 (B, 1, 64, hidden)
        - 对每个 batch b，从左到右扫描 tokens[b]：
            遇到连续的 marker（长度为 n）时，用 vision_tensors[b][k][:n] 替换
            h[b][start:start+n]；k += 1（第 k 张图）
        - 替换后截断到 seqlen；返回 (B, T, hidden)
        - 图片数比 marker 段少时，多出来的 marker 段保持原 embedding（或跳过）
        """
        B, T, H = h.shape
        marker = self.config.image_ids[0]
        output = []
        vf = vision_tensors.unsqueeze(1) if vision_tensors.dim() == 3 else vision_tensors
        for b in range(B):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:      # 找连续 marker 段
                        i += 1
                    if k < vf.size(1):                            # 第 k 张图
                        hb = torch.cat((hb[:start], vf[b][k][: i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            output.append(hb)
        return torch.stack(output)
        # raise NotImplementedError("TODO D16-1: count_vision_proj")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        """VLM 前向（TODO D16-2）。

        契约：
        1. past_key_values 为 None → [None] * num_hidden_layers；start_pos 同 Transformer
        2. hidden = self.model.dropout(self.model.embed_tokens(input_ids))
        3. 若 pixel_values 非空且 start_pos == 0：
            vision = self.vision_proj(self.get_image_embeddings(pixel_values, self.vision_encoder))
            注意 DataLoader 会把每个样本的 (num, 3, 256, 256) 叠成 (B, num, 3, 256, 256)：
            若 pixel_values 是 5 维，先 flatten(0,1) 过 encoder，再把特征 view(B, num, 64, hidden)
            hidden = self.count_vision_proj(input_ids, hidden, vision, seqlen=T)
        4. hidden, presents, aux = run_transformer_layers(
            self.model, hidden, attention_mask, past_key_values, use_cache)
        5. logits = self.lm_head(hidden)
        6. loss：和 CausalLM.forward 一样，logits[..., :-1] 对 labels[..., 1:]，ignore_index=-100
        7. 返回 CausalLMOutput(loss=..., logits=..., past_key_values=presents,
            aux_loss=aux, hidden_states=hidden)
        """
        B, T = input_ids.shape
        past_key_values = [None] * self.model.num_hidden_layers if past_key_values is None else past_key_values
        hidden = self.model.dropout(self.model.embed_tokens(input_ids))
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        if (pixel_values is not None) and (start_pos == 0):
            if pixel_values.dim() == 5:                        # DataLoader: (B, num, 3, 256, 256)
                bs, num = pixel_values.shape[:2]
                feats = self.vision_proj(
                    self.get_image_embeddings(pixel_values.flatten(0, 1), self.vision_encoder)
                )                                              # (B*num, 64, hidden)
                vision = feats.view(bs, num, self.config.image_token_len, -1)
            else:                                              # (num, 3, 256, 256)
                vision = self.vision_proj(
                    self.get_image_embeddings(pixel_values, self.vision_encoder)
                )                                              # (num, 64, hidden) 或 (1,64,hidden)
            hidden = self.count_vision_proj(input_ids, hidden, vision, seqlen=T)
        
        hidden, presents, aux_loss = run_transformer_layers(self.model, hidden, attention_mask, past_key_values, use_cache)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1), ignore_index=-100)

        return CausalLMOutput(
            loss=loss, logits=logits, past_key_values=presents,
            aux_loss=aux_loss, hidden_states=hidden,
        )
        # raise NotImplementedError("TODO D16-2: MiniMindVLM.forward")


def run_transformer_layers(model, hidden, attention_mask=None, past_key_values=None, use_cache=False):
    """给定 embedding，跑完 Transformer 各层 + final norm（已给好，VLM.forward 用）。"""
    if past_key_values is None:
        past_key_values = [None] * model.num_hidden_layers
    start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
    T = hidden.size(1)
    position_embeddings = (model.freqs_cos[start_pos : start_pos + T], model.freqs_sin[start_pos : start_pos + T])
    presents = []
    for layer, past in zip(model.layers, past_key_values):
        hidden, present = layer(
            hidden,
            position_embeddings,
            past_key_value=past,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        presents.append(present)
    hidden = model.norm(hidden)
    return hidden, presents, hidden.new_zeros(())


def freeze_vlm(model: MiniMindVLM, freeze_llm: int = 2) -> int:
    """按阶段设置 requires_grad，返回可训练参数量（已给好）。

    freeze_llm=2：全部冻结，只训练 vision_proj（对齐阶段，S9-05）
    freeze_llm=1：再放开第 0 层和最后一层（S9-06）
    freeze_llm=0：除 vision_encoder 外全部可训练
    """
    for name, param in model.named_parameters():
        param.requires_grad = "vision_proj" in name
    if freeze_llm == 0:
        for name, param in model.named_parameters():
            if "vision_encoder" not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        last = model.config.num_hidden_layers - 1
        for name, param in model.model.named_parameters():
            if "layers.0." in name or f"layers.{last}." in name:
                param.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
