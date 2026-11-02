"""D13 · LoRA 训练（LEARNING_TASKS: S7-03 / S7-04 / S7-05）

在 full_sft_{hidden}.pth 基础上挂 LoRA，只训练低秩矩阵，产出 lora 权重。

用法（在 learning\\scratch 下）：
    # 128 维小模型 + 医疗 LoRA（S7-03）
    python -m mymm.train_lora --hidden_size 128 --num_hidden_layers 2 --rank 16 `
        --data_path D:\\minimind\\learning\\data\\lora_medical_500.jsonl `
        --save_name lora_medical_tiny --epochs 2 --lr 1e-3

    # 训完用 chat 脚本加载 LoRA 推理（需要先 apply_lora + load_lora，见 S7-04）
    # 也可以直接 merge 成普通权重（S7-06）：
    python -m mymm.lora_merge --base out/full_sft_128.pth --lora out/lora_medical_tiny_128.pth `
        --out out/full_sft_lora_merged_128.pth --hidden_size 128 --num_hidden_layers 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .data import SFTDataset  # noqa: E402
from .lora import apply_lora, count_trainable, freeze_base_model, save_lora  # noqa: E402
from .model import CausalLM  # noqa: E402
from .train import SkipBatchSampler, train_epoch  # noqa: E402

LORA_DATA = r"D:\minimind\learning\data\lora_medical_500.jsonl"
TOKENIZER_DIR = r"D:\minimind\minimind\model"

__all__ = ["parse_args", "main"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind LoRA 训练（D13）")
    parser.add_argument("--data_path", default=LORA_DATA)
    parser.add_argument("--tokenizer_dir", default=TOKENIZER_DIR)
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--base_weight", default="full_sft", help="基础权重前缀（save_dir/<base_weight>_{hidden}.pth）")
    parser.add_argument("--save_name", default="lora_medical")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--show_only", type=int, default=0, choices=[0, 1], help="1=只打印 LoRA 挂载信息后退出")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model = CausalLM(config)

    base_path = Path(args.save_dir) / f"{args.base_weight}_{args.hidden_size}.pth"
    if args.base_weight != "none":
        if not base_path.exists():
            print(f"找不到基础权重：{base_path}")
            print("先跑 D10 的 SFT，或加 --base_weight none 从随机初始化开始")
            return 2
        model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True), strict=True)
        print(f"Loaded base weights: {base_path}")

    attached = apply_lora(model, rank=args.rank)
    freeze_base_model(model)
    model = model.to(device)
    print(f"LoRA attached: {attached} modules | trainable params: {count_trainable(model) / 1e6:.3f}M")
    if args.show_only:
        for name, module in model.named_modules():
            if hasattr(module, "lora"):
                print(f"  {name}: {module.in_features} -> {module.out_features}, rank={args.rank}")
        return 0

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    total_batches = len(dataset) // args.batch_size
    if total_batches <= 0:
        raise RuntimeError("数据集不足一个 batch")
    print(f"LoRA Dataset: {len(dataset)} samples | {total_batches} steps/epoch")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    train_args = SimpleNamespace(
        lr=args.lr,
        epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        device=str(device),
        dtype=args.dtype,
        save_interval=0,
        resume_path=None,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    lora_path = save_dir / f"{args.save_name}_{args.hidden_size}.pth"
    started = time.time()
    for epoch in range(args.epochs):
        torch.manual_seed(args.seed + epoch)
        indices = torch.randperm(len(dataset)).tolist()[: total_batches * args.batch_size]
        loader = DataLoader(
            dataset,
            batch_sampler=SkipBatchSampler(indices, args.batch_size, 0),
            num_workers=args.num_workers,
            pin_memory=("cuda" in str(device)),
        )
        train_epoch(epoch, loader, model, optimizer, scaler, train_args, start_step=0, iters=total_batches)
        save_lora(model, lora_path)
        print(f"[epoch {epoch + 1}/{args.epochs}] saved {lora_path} ({lora_path.stat().st_size / 1024:.1f} KB)")

    print(f"Done in {time.time() - started:.1f}s | LoRA weights: {lora_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
