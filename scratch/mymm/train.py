"""D7 · 预训练主循环（LEARNING_TASKS: S4-01 / S4-02 / S4-03）

目标：
    1) get_lr：余弦学习率调度
    2) train_epoch：bf16 autocast + 梯度累积 + grad clip + 日志
    3) main：把 jsonl 喂给 128 维 2 层小模型，训练 1 个 epoch 并存权重

验收：
    python mymm/tests/test_data_train.py 全 PASS
    python -m mymm.train --hidden_size 128 --num_hidden_layers 2 \\
        --max_seq_len 128 --batch_size 16 --accumulation_steps 2 \\
        --epochs 1 --data_path D:\\minimind\\learning\\data\\pretrain_5000.jsonl
    期望：5k 样本约 1 分钟（5070 Ti），loss 从 ~8.7 降到 ~7.1。

对照材料：先自己写，测试失败后再看
    minimind/trainer/train_pretrain.py:24（训练循环）
    minimind/trainer/trainer_utils.py:42（get_lr）和 :121（init_model）
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .data import PretrainDataset  # noqa: E402
from .model import CausalLM  # noqa: E402

__all__ = ["get_lr", "save_checkpoint", "load_checkpoint", "train_epoch", "main"]


def get_lr(step: int, total_steps: int, base_lr: float) -> float:
    """余弦学习率，与仓库 get_lr 一致。

    契约：
    - 公式：lr = base_lr * (0.1 + 0.45 * (1 + cos(pi * step / total_steps)))
    - step 从 1 到 total_steps；step=0 时恰好是 base_lr，step=total_steps 时是 0.1 * base_lr
    - 随着 step 增大单调递减
    - total_steps <= 0 时返回 base_lr（防止除零）
    """
    if total_steps <= 0:
        return base_lr
    else:
        return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))
    # raise NotImplementedError("TODO D7-2: get_lr")


def save_checkpoint(path, model, optimizer, scaler, epoch: int, step: int):
    """保存 resume 用的 checkpoint（已给好，不用改）。

    - path 里存 model / optimizer / scaler / epoch / step
    - model 权重按仓库习惯存成 fp16（.half().cpu()），加载时 load_state_dict 会自动转回参数 dtype
    """
    state = {k: v.half().cpu() for k, v in model.state_dict().items()}
    payload = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """加载 resume checkpoint（已给好，不用改），返回 (epoch, step)。"""
    data = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model"])
    if optimizer is not None and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
    if scaler is not None and data.get("scaler"):
        scaler.load_state_dict(data["scaler"])
    return data.get("epoch", 0), data.get("step", 0)


class SkipBatchSampler(Sampler):
    """断点续训用：按原顺序跳过前 skip_batches 个 batch（TODO D9-1）。

    契约：
    - __init__(indices, batch_size, skip_batches=0)：indices 是打乱后的样本下标列表
    - __iter__：把 indices 按 batch_size 切块，跳过前 skip_batches 块，逐块 yield list[int]
    - __len__：切块总数 - skip_batches，且不小于 0
    - 最后不足一个 batch 的块也要 yield
    """
    def __init__(self, indices, batch_size: int, skip_batches: int = 0):
        self.indices = list(indices)
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        start = self.skip_batches * self.batch_size
        for i in range(start, len(self.indices), self.batch_size):
            yield self.indices[i : i + self.batch_size]

    def __len__(self):
        return max((len(self.indices) + self.batch_size - 1) // self.batch_size - self.skip_batches,0)
        # raise NotImplementedError("TODO D9-2: SkipBatchSampler.__len__")


def train_epoch(epoch, loader, model, optimizer, scaler, args, start_step=0, iters: int | None = None) -> list[float]:
    """一个 epoch 的训练循环（iters 是 D9 为断点续训新增的参数）。

    参数：
    - epoch: 当前 epoch 编号（从 0 开始）
    - loader: DataLoader，每个 batch 是 (input_ids, labels)
    - model / optimizer / scaler: 标准训练对象
    - args: 需要这些字段
        args.lr                 初始学习率
        args.epochs             总 epoch 数（用于算 total_steps）
        args.accumulation_steps 梯度累积步数
        args.grad_clip          梯度裁剪阈值
        args.log_interval       日志间隔
        args.device             "cuda" 或 "cpu"
        args.dtype              "bfloat16" / "float16" / "float32"
        args.save_interval      每隔多少步保存一次 resume checkpoint（0 表示只在 epoch 末保存）
        args.resume_path        resume checkpoint 的路径（main 里设置）
    - start_step: 断点续训用，从第几步开始（正常训练传 0）
    - iters: 该 epoch 的总 batch 数（续训时由 main 传入；None 时取 len(loader)）

    返回：
    - losses: list[float]，每一步的 loss（已经乘回 accumulation_steps，便于和日志对照）

    实现步骤（顺序即契约）：
    1. model.train()
    2. iters = len(loader)；total_steps = args.epochs * iters
    3. 对 (step, (input_ids, labels)) in enumerate(loader, start=start_step + 1)：
            a. input_ids、labels 移到 args.device
            b. lr = get_lr(epoch * iters + step, total_steps, args.lr)
                写入 optimizer 的每个 param_group
            c. autocast：args.device 是 cuda 时用 torch.autocast("cuda", dtype=...)
                （"bfloat16"/"float16" 对应 torch.bfloat16/torch.float16），cpu 用 nullcontext()
            d. out = model(input_ids, labels=labels)
                loss = out.loss + out.aux_loss
                loss = loss / args.accumulation_steps
            e. scaler.scale(loss).backward()
            f. 当 step % accumulation_steps == 0 或 step == iters：
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            g. 每 log_interval 步或最后一步打印：
                    epoch、step/iters、loss（乘回 accumulation_steps）、当前 lr
            h. losses.append(loss.item() * args.accumulation_steps)
    4. 返回 losses

    提示（不是实现）：
    - 日志里的 loss 要乘回 accumulation_steps，否则看起来只有真实值的 1/accum；
    - total_steps 用 args.epochs * iters，不是只算当前 epoch；
    - scaler 在 bf16 下是 disabled 的，但代码可以照常调用 scale/unscale（no-op）；
    - 不要每步都 optimizer.step()，否则梯度累积失效。
    """
    iters = iters or len(loader)
    total_steps = args.epochs * iters
    losses = []
    for (step, (input_ids, labels)) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(epoch * iters + step, total_steps=total_steps,base_lr=args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # 修复：autocast 必须跟随 args.dtype。
        # 旧写法 torch.amp.autocast(device_type="cuda") 在 CUDA 上默认 float16，
        # 而 args.dtype 默认是 bfloat16（GradScaler 只在 float16 时启用），
        # 结果是 768 维模型在 fp16 下无 loss scaling 训练 → 溢出 → loss 变 NaN。
        if args.dtype == "float32" or args.device == "cpu":
            autocast_ctx = nullcontext()
        else:
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        with autocast_ctx:
            out = model(input_ids, labels=labels)
            loss = out.loss + out.aux_loss
            loss = loss / args.accumulation_steps
        
        scaler.scale(loss).backward()

        if (step % args.accumulation_steps == 0) or (step == iters):
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (step % args.log_interval == 0) or (step == iters):
            print(f"epoch: {epoch},\tstep: {step},\t loss: {loss * args.accumulation_steps}, lr: {lr}")

        losses.append(loss.item() * args.accumulation_steps)

        save_interval = getattr(args, "save_interval", 0)
        resume_path = getattr(args, "resume_path", None)
        if save_interval and resume_path and (step % save_interval == 0 or step == iters):
            save_checkpoint(resume_path, model, optimizer, scaler, epoch, step)

    return losses
    # raise NotImplementedError("TODO D7-3: train_epoch")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind pretrain（D7）")
    parser.add_argument("--data_path", default=r"D:\minimind\learning\data\pretrain_5000.jsonl")
    parser.add_argument("--tokenizer_dir", default=r"D:\minimind\minimind\model")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--save_weight", default="pretrain")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000, help="每隔多少步保存一次 resume checkpoint")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="1=从 save_dir 下的 resume checkpoint 继续")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model = CausalLM(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {total / 1e6:.2f}M")

    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    total_batches = len(dataset) // args.batch_size
    if total_batches <= 0:
        raise RuntimeError("数据集不足一个 batch")
    print(f"Dataset: {len(dataset)} samples | {total_batches} steps/epoch")

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
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
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
