"""D13 · 手写 LoRA（LEARNING_TASKS: S7-01 / S7-02 / S7-05 / S7-06）

目标：
    1) LoRA 模块：A(rank, in) 高斯初始化、B(out, rank) 零初始化
    2) apply_lora：只挂在 in_features == out_features 的 Linear 上（MiniMind 里是 q_proj / o_proj）
    3) save_lora / load_lora / merge_lora：保存、加载、把 B@A 合并回 W

验收：mymm/tests/test_lora.py 全 PASS。
对照材料：先自己写，测试失败后再看 minimind/model/model_lora.py。

关键约定：
    - LoRA 的 forward = B(A(x))，挂载后 Linear 的 forward 变成 base(x) + lora(x)
    - B 初始为 0 ⇒ apply_lora 之后模型输出必须和原来完全一致
    - merge 后：W_new = W_old + B @ A，模型输出不变，且不再有 lora 子模块
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

__all__ = [
    "LoRA",
    "apply_lora",
    "save_lora",
    "load_lora",
    "merge_lora",
    "freeze_base_model",
    "count_trainable",
]


class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # TODO D13-1: 初始化
        # - A.weight 用正态分布 N(0, 0.02)（打破对称、给低秩更新一个非零起点）
        # - B.weight 全 0（保证 apply 之后 B@A=0，模型输出不变）
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.data.zero_()
        # raise NotImplementedError("TODO D13-1: LoRA 初始化（A ~ N(0,0.02)，B = 0）")

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank: int = 16) -> int:
    """给所有 in_features == out_features 的 nn.Linear 挂 LoRA，返回挂载数量。

    契约：
    - 遍历 model.named_modules()，对每个 square Linear：
        setattr(module, "lora", LoRA(in, out, rank))
        base_forward = module.forward
        module.forward = lambda x, base_forward=base_forward, lora=module.lora: base_forward(x) + lora(x)
        （用默认参数绑定，避免闭包捕获循环变量）
    - 原始权重保持不动；挂在正确 device/dtype 上
    - 返回挂载的模块个数（768 维每层 q/o 两个，8 层共 16 个）
    """
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (module.in_features == module.out_features):
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            setattr(module, "lora", lora)
            base_forward = module.forward
            module.forward = lambda x, base_forward=base_forward, lora=module.lora: base_forward(x) + lora(x)
            cnt += 1
    return cnt
    # raise NotImplementedError("TODO D13-2: apply_lora")


def save_lora(model, path):
    """只保存 lora 参数（已给好，不用改）：key 形如 model.layers.0.self_attn.q_proj.lora.A.weight。"""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            for key, value in module.lora.state_dict().items():
                state[f"{name}.lora.{key}"] = value.detach().cpu().half()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


def load_lora(model, path):
    """把 save_lora 的权重加载回已经 apply_lora 的模型（已给好，不用改）。"""
    state = torch.load(path, map_location="cpu", weights_only=True)
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            prefix = f"{name}.lora."
            sub = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if sub:
                module.lora.load_state_dict(sub, strict=True)
                loaded += 1
    return loaded


def merge_lora(model, lora_path: str = ""):
    """把 B@A 合并进 W，并卸载 lora（S7-06）。

    契约：
    - lora_path 非空时先 load_lora(model, lora_path)
    - 对每个带 lora 的 square Linear：
        with torch.no_grad():
            module.weight.data += module.lora.B.weight.data @ module.lora.A.weight.data
    - 然后删掉实例上的 forward 覆盖和 lora 子模块：
        del module.forward
        del module.lora
        （del module.forward 会恢复 nn.Linear 的原始 forward）
    - 注意遍历时先收集列表再删除，不要边遍历边改 named_modules()
    - 合并后同一输入输出应与合并前一致（误差 < 1e-5）
    """
    if lora_path != "":
        load_lora(model, lora_path)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "lora"):
                module.weight.data += module.lora.B.weight.data @ module.lora.A.weight.data
                del module.forward
                del module.lora
    # raise NotImplementedError("TODO D13-3: merge_lora")


def freeze_base_model(model):
    """冻结除 lora 之外的所有参数（已给好，不用改）。"""
    for name, param in model.named_parameters():
        param.requires_grad = ".lora." in name


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
