"""D10 · SFT 训练（LEARNING_TASKS: S5-01 / S5-02 / S5-03 / S5-04）

目标：
    1) SFTDataset：chat template + 只对 assistant 回复算 loss（TODO 在 data.py 里，D10-1）
    2) 在 pretrain_{hidden}.pth 基础上做 SFT，产出 full_sft_{hidden}.pth
    3) 支持 --from_weight none 做“无预训练直接 SFT”的对照

用法（在 learning\\scratch 下）：
    # 先看一条样本的 token/label 对齐（S5-01）
    python -m mymm.train_sft --show_samples 1 --max_seq_len 256

    # 正常 SFT：从 pretrain_128.pth 继续（S5-03）
    python -m mymm.train_sft --hidden_size 128 --num_hidden_layers 2 --max_seq_len 256 \\
        --batch_size 8 --accumulation_steps 2 --epochs 1 --lr 1e-5 \\
        --data_path D:\\minimind\\learning\\data\\sft_2000.jsonl

    # 无预训练对照（S5-04）
    python -m mymm.train_sft --from_weight none --save_weight full_sft_none ...

权重约定：
    --save_dir/pretrain_{hidden}.pth    SFT 的起点（D7/D9 产出）
    --save_dir/full_sft_{hidden}.pth    SFT 产出
    --save_dir/full_sft_{hidden}_resume.pth  resume checkpoint
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .data import SFTDataset  # noqa: E402
from .model import CausalLM  # noqa: E402
from .train import SkipBatchSampler, load_checkpoint, save_checkpoint, train_epoch  # noqa: E402

__all__ = ["print_samples", "parse_args", "main"]


def print_samples(dataset: SFTDataset, tokenizer, count: int, max_tokens: int = 80):
    """逐 token 打印 input_ids / labels（-100 显示为 --），用于 S5-01 验收。"""
    for idx in range(min(count, len(dataset))):
        input_ids, labels = dataset[idx]
        non_mask = int((labels != -100).sum())
        print(f"\n===== sample {idx} | len={len(input_ids)} | 参与 loss 的 token 数={non_mask} =====")
        for j in range(min(max_tokens, len(input_ids))):
            x = int(input_ids[j])
            y = int(labels[j])
            x_text = repr(tokenizer.decode([x]))
            y_text = "--" if y == -100 else repr(tokenizer.decode([y]))
            print(f"  {j:3d}  X={x_text:18s}  label={y_text}")
        if len(input_ids) > max_tokens:
            print(f"  ... 后面还有 {len(input_ids) - max_tokens} 个 token")
    print()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind SFT（D10）")
    parser.add_argument("--data_path", default=r"D:\minimind\learning\data\sft_2000.jsonl")
    parser.add_argument("--tokenizer_dir", default=r"D:\minimind\minimind\model")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--save_weight", default="full_sft")
    parser.add_argument("--from_weight", default="pretrain", help="从 save_dir/<from_weight>_{hidden}.pth 继续；none 表示不加载")
    parser.add_argument("--weight_path", default="", help="直接指定预训练权重路径（优先于 --from_weight）")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--show_samples", type=int, default=0, help=">0 时只打印样本后退出（S5-01）")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    print(f"SFT Dataset: {len(dataset)} samples | max_seq_len={args.max_seq_len}")

    if args.show_samples > 0:
        print_samples(dataset, tokenizer, args.show_samples)
        return 0

    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model = CausalLM(config).to(device)

    if args.from_weight != "none":
        weight_path = Path(args.weight_path) if args.weight_path else (
            Path(args.save_dir) / f"{args.from_weight}_{args.hidden_size}.pth"
        )
        if not weight_path.exists():
            print(f"找不到预训练权重：{weight_path}")
            print("先跑 D7/D9 的 pretrain，或用 --from_weight none 做对照")
            return 2
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"Loaded pretrain weights: {weight_path}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {total / 1e6:.2f}M")

    total_batches = len(dataset) // args.batch_size
    if total_batches <= 0:
        raise RuntimeError("数据集不足一个 batch")
    print(f"Steps/epoch: {total_batches}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args.resume_path = str(save_dir / f"{args.save_weight}_{args.hidden_size}_resume.pth")

    start_epoch, start_step = 0, 0
    if args.from_resume == 1 and Path(args.resume_path).exists():
        start_epoch, start_step = load_checkpoint(args.resume_path, model, optimizer, scaler)
        print(f"Resume from epoch {start_epoch} step {start_step}")

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        torch.manual_seed(args.seed + epoch)
        indices = torch.randperm(len(dataset)).tolist()[: total_batches * args.batch_size]
        loader = DataLoader(
            dataset,
            batch_sampler=SkipBatchSampler(indices, args.batch_size, skip),
            num_workers=args.num_workers,
            pin_memory=("cuda" in args.device),
        )
        train_epoch(epoch, loader, model, optimizer, scaler, args, start_step=skip, iters=total_batches)

        ckpt = save_dir / f"{args.save_weight}_{args.hidden_size}.pth"
        state = {k: v.half().cpu() for k, v in model.state_dict().items()}
        torch.save(state, ckpt)
        save_checkpoint(args.resume_path, model, optimizer, scaler, epoch + 1, 0)
        print(f"[epoch {epoch + 1}/{args.epochs}] saved {ckpt} | {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
