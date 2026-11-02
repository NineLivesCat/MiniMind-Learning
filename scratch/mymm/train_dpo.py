"""D14 · DPO 训练（LEARNING_TASKS: S7-07 / S7-08）

在 full_sft 权重上做偏好对齐：
    policy（可训练） + ref（冻结，初始 = full_sft） + 偏好对 chosen/rejected

用法（在 learning\\scratch 下）：
    # 128 维小模型，几百步即可看到 margin 变化（S7-08）
    python -m mymm.train_dpo --hidden_size 128 --num_hidden_layers 2 `
        --data_path D:\\minimind\\learning\\data\\dpo_500.jsonl `
        --epochs 1 --batch_size 2 --accumulation_steps 2 --lr 4e-8 --beta 0.1

    # 训练前后 margin 对比（S7-09）
    python -m mymm.dpo_metrics --hidden_size 128 --num_hidden_layers 2 --num_samples 100

产出：out/dpo_{hidden}.pth
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .data import DPODataset  # noqa: E402
from .dpo import dpo_loss, sequence_logps  # noqa: E402
from .model import CausalLM  # noqa: E402

DPO_DATA = r"D:\minimind\learning\data\dpo_500.jsonl"
TOKENIZER_DIR = r"D:\minimind\minimind\model"

__all__ = ["parse_args", "main"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind DPO（D14）")
    parser.add_argument("--data_path", default=DPO_DATA)
    parser.add_argument("--tokenizer_dir", default=TOKENIZER_DIR)
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--base_weight", default="full_sft")
    parser.add_argument("--save_weight", default="dpo")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def _move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and args.dtype != "float32" else nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)

    save_dir = Path(args.save_dir)
    base_path = save_dir / f"{args.base_weight}_{args.hidden_size}.pth"
    if not base_path.exists():
        print(f"找不到 SFT 权重：{base_path}")
        print("先跑 D10 的 SFT，或用 --base_weight 指定别的前缀")
        return 2

    policy = CausalLM(config)
    policy.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True), strict=True)
    ref = CausalLM(config)
    ref.load_state_dict(policy.state_dict(), strict=True)
    ref = ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    policy = policy.to(device)
    ref = ref.to(device)
    print(f"Loaded SFT weights: {base_path} | params: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")

    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    total_batches = len(dataset) // args.batch_size
    if total_batches <= 0:
        raise RuntimeError("数据集不足一个 batch")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=True,
    )
    print(f"DPO Dataset: {len(dataset)} samples | {total_batches} steps/epoch | beta={args.beta}")

    optimizer = optim.AdamW(policy.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.save_weight}_{args.hidden_size}.pth"

    started = time.time()
    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_margin = 0.0
        epoch_acc = 0.0
        steps = 0
        for step, batch in enumerate(loader, start=1):
            batch = _move(batch, device)
            with ctx:
                policy_chosen = sequence_logps(policy, batch["x_chosen"], batch["y_chosen"], batch["mask_chosen"])
                policy_rejected = sequence_logps(policy, batch["x_rejected"], batch["y_rejected"], batch["mask_rejected"])
                with torch.no_grad():
                    ref_chosen = sequence_logps(ref, batch["x_chosen"], batch["y_chosen"], batch["mask_chosen"])
                    ref_rejected = sequence_logps(ref, batch["x_rejected"], batch["y_rejected"], batch["mask_rejected"])
            result = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=args.beta)
            loss = result["loss"] / args.accumulation_steps
            scaler.scale(loss).backward()

            if step % args.accumulation_steps == 0 or step == total_batches:
                scaler.unscale_(optimizer)
                clip_grad_norm_(policy.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            margin = result["reward_margin"].mean().item()
            acc = result["accuracy"].item()
            epoch_loss += result["loss"].item()
            epoch_margin += margin
            epoch_acc += acc
            steps += 1
            if step % args.log_interval == 0 or step == total_batches:
                print(
                    f"epoch {epoch + 1} step {step}/{total_batches} "
                    f"dpo_loss {result['loss'].item():.4f} margin {margin:+.4f} acc {acc:.3f}"
                )

        state = {k: v.half().cpu() for k, v in policy.state_dict().items()}
        torch.save(state, out_path)
        print(
            f"[epoch {epoch + 1}/{args.epochs}] avg_loss={epoch_loss / max(steps, 1):.4f} "
            f"avg_margin={epoch_margin / max(steps, 1):+.4f} avg_acc={epoch_acc / max(steps, 1):.3f} "
            f"| saved {out_path} | {time.time() - started:.1f}s"
        )
    print(f"Done: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
