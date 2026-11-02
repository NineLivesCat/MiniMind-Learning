"""D15 · 蒸馏训练（LEARNING_TASKS: S8-02）

学生模型 + 教师模型（通常同一份权重做自蒸馏，或教师更大）：
    L = alpha * CE + (1 - alpha) * T² * KL(teacher/T || student/T)

用法（在 learning\\scratch 下）：
    # 128 维自蒸馏 50 步（S8-02），student/teacher 都用 full_sft_128.pth
    python -m mymm.train_distill --hidden_size 128 --num_hidden_layers 2 `
        --max_steps 50 --batch_size 8 --accumulation_steps 2 --lr 1e-5 `
        --alpha 0.5 --temperature 2.0 --log_interval 10

    # 教师用 768、学生 128 的结构蒸馏（可选，需要 teacher 与 student 词表一致）
    # 当前脚本默认 student/teacher 同 config，更大教师请自行扩展 --teacher_hidden_size

产出：out/distill_{hidden}.pth，日志同时打印 CE 与 KD 两项。
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .data import SFTDataset  # noqa: E402
from .distill import distill_loss  # noqa: E402
from .model import CausalLM  # noqa: E402
from .train import get_lr  # noqa: E402

SFT_DATA = r"D:\minimind\learning\data\sft_2000.jsonl"
TOKENIZER_DIR = r"D:\minimind\minimind\model"

__all__ = ["parse_args", "train_distill", "main"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind 蒸馏训练（D15）")
    parser.add_argument("--data_path", default=SFT_DATA)
    parser.add_argument("--tokenizer_dir", default=TOKENIZER_DIR)
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--save_weight", default="distill")
    parser.add_argument("--student_weight", default="full_sft")
    parser.add_argument("--teacher_weight", default="full_sft", help="none 表示不用教师（退化成纯 CE）")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help=">0 时只训练这么多步（S8-02 的 50 步冒烟）")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def train_distill(student, teacher, loader, optimizer, scaler, args, ctx) -> dict:
    student.train()
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
    iters = len(loader)
    total_steps = args.epochs * iters
    totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "steps": 0}
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(step, total_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with ctx:
            student_logits = student(input_ids).logits
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids).logits
            else:
                teacher_logits = student_logits.detach()
        result = distill_loss(student_logits, teacher_logits, labels, alpha=args.alpha, temperature=args.temperature)
        loss = result["loss"] / args.accumulation_steps
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for key in ("loss", "ce", "kd"):
            totals[key] += result[key].item()
        totals["steps"] += 1
        if step % args.log_interval == 0 or step == iters:
            print(
                f"step {step}/{iters} lr {lr:.2e} "
                f"loss {result['loss'].item():.4f} ce {result['ce'].item():.4f} kd {result['kd'].item():.4f}"
            )
    return totals


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and args.dtype != "float32" else nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    save_dir = Path(args.save_dir)

    student = CausalLM(config)
    student_path = save_dir / f"{args.student_weight}_{args.hidden_size}.pth"
    if not student_path.exists():
        print(f"找不到学生权重：{student_path}")
        return 2
    student.load_state_dict(torch.load(student_path, map_location="cpu", weights_only=True), strict=True)
    student = student.to(device)
    print(f"Student: {student_path}")

    teacher = None
    if args.teacher_weight != "none":
        teacher_path = save_dir / f"{args.teacher_weight}_{args.hidden_size}.pth"
        if not teacher_path.exists():
            print(f"找不到教师权重：{teacher_path}（或加 --teacher_weight none 只做 CE）")
            return 2
        teacher = CausalLM(config)
        teacher.load_state_dict(torch.load(teacher_path, map_location="cpu", weights_only=True), strict=True)
        teacher = teacher.eval().to(device)
        print(f"Teacher: {teacher_path}")

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if args.max_steps > 0:
        keep = min(len(dataset), args.max_steps * args.batch_size)
        dataset = Subset(dataset, list(range(keep)))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("数据集不足一个 batch")
    print(f"Distill Dataset: {len(dataset)} samples | {len(loader)} steps/epoch | alpha={args.alpha} T={args.temperature}")

    optimizer = optim.AdamW(student.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.save_weight}_{args.hidden_size}.pth"

    started = time.time()
    for epoch in range(args.epochs):
        stats = train_distill(student, teacher, loader, optimizer, scaler, args, ctx)
        state = {k: v.half().cpu() for k, v in student.state_dict().items()}
        torch.save(state, out_path)
        n = max(stats["steps"], 1)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] avg_loss={stats['loss'] / n:.4f} "
            f"avg_ce={stats['ce'] / n:.4f} avg_kd={stats['kd'] / n:.4f} "
            f"| saved {out_path} | {time.time() - started:.1f}s"
        )
    print(f"Done: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
