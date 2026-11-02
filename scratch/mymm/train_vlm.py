"""D16 · VLM 两阶段训练（LEARNING_TASKS: S9-05 / S9-06 / S9-07 / S9-08 / S9-09）

阶段 1（pretrain_vlm）：只训练 projector（freeze_llm=2），让视觉特征对齐 LLM 词嵌入
阶段 2（sft_vlm）    ：再放开 LLM 首尾层（freeze_llm=1），学对话格式

用法（在 learning\\scratch 下）：
    # 128 维冒烟（数据用 1000 条子集，几十步即可跑通）
    python -m mymm.train_vlm --stage pretrain --hidden_size 128 --num_hidden_layers 2 `
        --data_path D:\\minimind\\learning\\data\\vlm_1000.parquet `
        --max_steps 50 --batch_size 2 --accumulation_steps 2 --lr 1e-4

    # 768 正式训练：先投影对齐
    python -m mymm.train_vlm --stage pretrain --hidden_size 768 --num_hidden_layers 8 `
        --init_weight D:\\minimind\\minimind-v\\out\\llm_768.pth `
        --data_path D:\\minimind\\learning\\data\\vlm_1000.parquet --epochs 1 --batch_size 4

    # 再 SFT（自动加载上一步的 pretrain_vlm_768.pth）
    python -m mymm.train_vlm --stage sft --hidden_size 768 --num_hidden_layers 8 `
        --data_path D:\\minimind\\learning\\data\\vlm_sft_500.parquet --epochs 1 --batch_size 4

产出：out/pretrain_vlm_{hidden}.pth、out/sft_vlm_{hidden}.pth（不含冻结的 vision_encoder，体积小）
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
from .train import get_lr  # noqa: E402
from .vlm import VISION_MODEL_PATH, MiniMindVLM, freeze_vlm  # noqa: E402
from .vlm_data import VLMDataset  # noqa: E402

TOKENIZER_DIR = r"D:\minimind\minimind\model"
PRETRAIN_DATA = r"D:\minimind\learning\data\vlm_1000.parquet"
SFT_DATA = r"D:\minimind\learning\data\vlm_sft_500.parquet"

__all__ = ["parse_args", "main"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind VLM 训练（D16）")
    parser.add_argument("--stage", default="pretrain", choices=["pretrain", "sft"])
    parser.add_argument("--data_path", default="")
    parser.add_argument("--tokenizer_dir", default=TOKENIZER_DIR)
    parser.add_argument("--vision_model_path", default=VISION_MODEL_PATH)
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--init_weight", default="", help="pretrain 阶段可指定 llm_768.pth 等；sft 默认加载 pretrain_vlm")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_llm", type=int, default=-1, choices=[-1, 0, 1, 2],
                        help="-1=按阶段自动（pretrain=2, sft=1）")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and args.dtype != "float32" else nullcontext()

    if not args.data_path:
        args.data_path = PRETRAIN_DATA if args.stage == "pretrain" else SFT_DATA
    freeze_llm = (2 if args.stage == "pretrain" else 1) if args.freeze_llm == -1 else args.freeze_llm

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model = MiniMindVLM(config, vision_model_path=args.vision_model_path)
    if model.processor is None:
        print(f"找不到 vision encoder：{args.vision_model_path}")
        return 2

    save_dir = Path(args.save_dir)
    if args.stage == "pretrain":
        init_weight = args.init_weight
    else:
        init_weight = args.init_weight or str(save_dir / f"pretrain_vlm_{args.hidden_size}.pth")
    if init_weight and init_weight != "none":
        init_path = Path(init_weight)
        if not init_path.exists():
            print(f"找不到初始权重：{init_path}")
            return 2
        state = torch.load(init_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded init: {init_path.name} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("Init: random（只适合冒烟）")

    trainable = freeze_vlm(model, freeze_llm)
    model = model.to(device)
    print(f"VLM hidden={args.hidden_size} layers={args.num_hidden_layers} | freeze_llm={freeze_llm} | trainable={trainable / 1e6:.3f}M")

    dataset = VLMDataset(args.data_path, tokenizer, model.processor, max_length=args.max_seq_len)
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
    total_steps = args.epochs * len(loader)
    print(f"VLM Dataset: {len(dataset)} samples | {len(loader)} steps/epoch")

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.stage}_vlm_{args.hidden_size}.pth"

    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            lr = get_lr(step, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with ctx:
                out = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = out.loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if step % args.accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0 or step == len(loader):
                print(f"epoch {epoch + 1} step {step}/{len(loader)} lr {lr:.2e} loss {out.loss.item():.4f}")

        state = {k: v.half().cpu() for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")}
        torch.save(state, out_path)
        print(f"[epoch {epoch + 1}/{args.epochs}] saved {out_path} ({out_path.stat().st_size / 1024 ** 2:.1f} MB) | {time.time() - started:.1f}s")
    print(f"Done: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
