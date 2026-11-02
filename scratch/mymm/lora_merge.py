"""D13 · LoRA 合并工具（LEARNING_TASKS: S7-06）

把 LoRA 权重合并进基础权重，得到一份普通（无 lora 子模块）的 checkpoint。

用法（在 learning\\scratch 下）：
    python -m mymm.lora_merge --base out/full_sft_128.pth --lora out/lora_medical_tiny_128.pth `
        --out out/full_sft_lora_merged_128.pth --hidden_size 128 --num_hidden_layers 2

合并后：
    W_new = W_old + B @ A
    新权重可以直接用 mymm.chat 加载（不需要 apply_lora）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .lora import apply_lora, load_lora, merge_lora  # noqa: E402
from .model import CausalLM  # noqa: E402

__all__ = ["merge_files", "parse_args", "main"]


def merge_files(base_path: str, lora_path: str, out_path: str,
                hidden_size: int, num_hidden_layers: int) -> Path:
    """加载 base + lora → merge → 保存成普通 state_dict（fp16）。"""
    state = torch.load(lora_path, map_location="cpu", weights_only=True)
    ranks = {value.shape[0] for key, value in state.items() if key.endswith(".lora.A.weight")}
    if not ranks:
        raise ValueError(f"{lora_path} 里没有 lora.A.weight，可能不是 LoRA 权重")
    rank = ranks.pop()

    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    model = CausalLM(config)
    model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True), strict=True)

    attached = apply_lora(model, rank=rank)
    loaded = load_lora(model, lora_path)
    merge_lora(model)
    print(f"merged {loaded}/{attached} lora modules (rank={rank})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(merged, out_path)
    print(f"saved merged weights: {out_path}")
    return out_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Merge LoRA into base weights（D13）")
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    merge_files(args.base, args.lora, args.out, args.hidden_size, args.num_hidden_layers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
