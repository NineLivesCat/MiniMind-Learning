"""D18 · MoE vs dense 对照实验（LEARNING_TASKS: S3-15 选做 / S10-06）

同一份数据、同样步数，比较：
    dense    : 每个 token 都过同一个 FFN
    moe4_top2: 4 个 expert，每个 token 只走 top-2（加负载均衡 aux loss）

产出：out/moe/exp/<name>.csv（step, loss, ce, aux, lr）、out/moe/curves.png、终端对照表。

用法（在 learning\\scratch 下）：
    python -m mymm.moe_compare --run dense moe --steps 100 --batch_size 8 --plot
    python -m mymm.moe_compare --run dense moe --steps 300 --lr 5e-4 --out_dir out/moe
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
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
from .data import PretrainDataset  # noqa: E402
from .model import CausalLM  # noqa: E402
from .moe import MoECausalLM, apply_moe  # noqa: E402
from .train import get_lr  # noqa: E402

PRETRAIN_DATA = r"D:\minimind\learning\data\pretrain_5000.jsonl"
TOKENIZER_DIR = r"D:\minimind\minimind\model"

EXPERIMENTS = {
    "dense": {"use_moe": False},
    "moe": {"use_moe": True, "num_experts": 4, "top_k": 2},
    "moe_top1": {"use_moe": True, "num_experts": 4, "top_k": 1},
}

__all__ = ["EXPERIMENTS", "run_one", "plot_curves", "parse_args", "main"]


def run_one(name: str, args, out_dir: Path) -> dict:
    spec = EXPERIMENTS[name]
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=spec["use_moe"],
        num_experts=spec.get("num_experts", 4),
        num_experts_per_tok=spec.get("top_k", 2),
        router_aux_loss_coef=args.aux_coef,
    )
    if spec["use_moe"]:
        model = MoECausalLM(config)
        apply_moe(model, num_experts=config.num_experts, top_k=config.num_experts_per_tok)
    else:
        model = CausalLM(config)
    model = model.to(device)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and args.dtype == "bfloat16" else torch.autocast("cpu", enabled=False)

    exp_dir = out_dir / "exp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / f"{name}.csv"
    rows = []
    if "cuda" in args.device:
        torch.cuda.reset_peak_memory_stats()

    it = iter(loader)
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            input_ids, labels = next(it)
        except StopIteration:
            it = iter(loader)
            input_ids, labels = next(it)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with ctx:
            out = model(input_ids, labels=labels)
        loss = out.loss + out.aux_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        lr = get_lr(step, args.steps, args.lr)
        rows.append((step, loss.item(), out.loss.item(), out.aux_loss.item(), lr))
        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(
                f"  {name:10s} step {step}/{args.steps} loss {loss.item():.4f} "
                f"ce {out.loss.item():.4f} aux {out.aux_loss.item():.5f}"
            )

    elapsed = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if "cuda" in args.device else None
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss", "ce", "aux", "lr"])
        for row in rows:
            writer.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.8f}", f"{row[4]:.8e}"])

    summary = {
        "name": name,
        "params_m": params_m,
        "loss_last": rows[-1][1],
        "ce_last": rows[-1][2],
        "aux_last": rows[-1][3],
        "seconds": elapsed,
        "peak_mb": peak_mb,
        "csv": str(csv_path),
    }
    print(
        f"  {name:10s} params={params_m:.2f}M loss={summary['loss_last']:.3f} ce={summary['ce_last']:.3f} "
        f"time={elapsed:.1f}s peak={'无CUDA' if peak_mb is None else f'{peak_mb:.0f}MB'}"
    )
    return summary


def plot_curves(csv_dir: Path, png_path: Path, names: list[str]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in names:
        path = csv_dir / f"{name}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            data = list(csv.DictReader(fh))
        steps = [int(r["step"]) for r in data]
        axes[0].plot(steps, [float(r["loss"]) for r in data], label=name)
        axes[1].plot(steps, [float(r["aux"]) for r in data], label=name)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (ce+aux)")
    axes[0].legend()
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("aux loss")
    axes[1].legend()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"曲线已保存：{png_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MoE vs dense 对照（D18）")
    parser.add_argument("--run", nargs="+", default=["dense", "moe"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--data_path", default=PRETRAIN_DATA)
    parser.add_argument("--tokenizer_dir", default=TOKENIZER_DIR)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--aux_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--out_dir", default="out/moe")
    parser.add_argument("--plot", type=int, default=1, choices=[0, 1])
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    unknown = [n for n in args.run if n not in EXPERIMENTS]
    if unknown:
        print(f"未知实验：{unknown}，可选 {list(EXPERIMENTS)}")
        return 2
    out_dir = Path(args.out_dir)
    print(f"device={args.device} | steps={args.steps} | batch={args.batch_size} | aux_coef={args.aux_coef}")
    summaries = [run_one(name, args, out_dir) for name in args.run]

    print()
    print(f"{'name':10s} {'params':>8s} {'loss':>8s} {'ce':>8s} {'aux':>9s} {'time(s)':>8s} {'peak':>8s}")
    for s in summaries:
        peak = "-" if s["peak_mb"] is None else f"{s['peak_mb']:.0f}MB"
        print(
            f"{s['name']:10s} {s['params_m']:>7.2f}M {s['loss_last']:>8.3f} {s['ce_last']:>8.3f} "
            f"{s['aux_last']:>9.5f} {s['seconds']:>8.1f} {peak:>8s}"
        )
    if args.plot:
        plot_curves(out_dir / "exp", out_dir / "curves.png", args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
