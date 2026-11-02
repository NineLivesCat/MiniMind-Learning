"""D11 · SFT 对照实验（LEARNING_TASKS: S5-05 / S5-06）

只改一个变量，比较 SFT 的学习率和数据：
    lr      : 1e-4 / 1e-5 / 1e-6（默认 1e-5）
    control : 无预训练（--from_weight none 的等价实验）
    custom  : 自造的 my_sft_20.jsonl（S5-06，需要你先造好数据）

用法（在 learning\\scratch 下）：
    python -m mymm.sft_experiments --list
    python -m mymm.sft_experiments --group lr
    python -m mymm.sft_experiments --run custom_20
    python -m mymm.sft_experiments --all --steps 50      # 冒烟
    python -m mymm.sft_experiments --plot                # out/exp_sft/curves.png

结果写到 out/exp_sft/<name>.csv（step, loss, lr），并打印汇总表。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .data import SFTDataset  # noqa: E402
from .experiments import plot_curves  # noqa: E402
from .model import CausalLM  # noqa: E402
from .train import get_lr, train_epoch  # noqa: E402

SFT_DATA = r"D:\minimind\learning\data\sft_2000.jsonl"
MY_SFT_DATA = r"D:\minimind\learning\data\my_sft_20.jsonl"
TOKENIZER_DIR = r"D:\minimind\minimind\model"

BASE = dict(
    data_path=SFT_DATA,
    tokenizer_dir=TOKENIZER_DIR,
    save_dir="out",
    pretrain_weight="pretrain",
    pretrain_path="",
    hidden_size=128,
    num_hidden_layers=2,
    max_seq_len=256,
    batch_size=8,
    accumulation_steps=2,
    epochs=1,
    lr=1e-5,
    grad_clip=1.0,
    dtype="bfloat16",
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)

EXPERIMENTS = {
    "lr_1e-4": {**BASE, "lr": 1e-4, "group": "lr"},
    "lr_1e-5": {**BASE, "lr": 1e-5, "group": "lr"},
    "lr_1e-6": {**BASE, "lr": 1e-6, "group": "lr"},
    "no_pretrain": {**BASE, "pretrain_weight": "none", "group": "control"},
    "custom_20": {**BASE, "data_path": MY_SFT_DATA, "group": "custom"},
}

GROUPS = {
    "lr": [n for n, c in EXPERIMENTS.items() if c["group"] == "lr"],
    "control": [n for n, c in EXPERIMENTS.items() if c["group"] == "control"],
    "custom": [n for n, c in EXPERIMENTS.items() if c["group"] == "custom"],
}


def _loss_at(losses: list[float], step: int):
    return losses[step - 1] if 0 < step <= len(losses) else None


def run_one(name: str, cfg: dict | None = None, out_dir: str = "out", max_steps: int = 0) -> dict:
    cfg = dict(cfg or EXPERIMENTS[name])
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])
    dataset = SFTDataset(cfg["data_path"], tokenizer, max_length=cfg["max_seq_len"])
    if max_steps > 0:
        keep = min(len(dataset), max_steps * cfg["batch_size"])
        dataset = Subset(dataset, list(range(keep)))
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=("cuda" in device),
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("数据集太小，凑不出一个 batch")

    config = MiniMindConfig(hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layers"])
    model = CausalLM(config)

    if cfg.get("pretrain_weight", "none") != "none":
        weight_path = Path(cfg["pretrain_path"]) if cfg.get("pretrain_path") else (
            Path(cfg["save_dir"]) / f"{cfg['pretrain_weight']}_{cfg['hidden_size']}.pth"
        )
        if not weight_path.exists():
            raise FileNotFoundError(
                f"找不到预训练权重：{weight_path}\n先跑 D7/D9 的 pretrain，或用 pretrain_weight='none'"
            )
        model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["dtype"] == "float16" and "cuda" in device))
    if "cuda" in device:
        torch.cuda.reset_peak_memory_stats()

    args = SimpleNamespace(
        lr=cfg["lr"],
        epochs=cfg["epochs"],
        accumulation_steps=cfg["accumulation_steps"],
        grad_clip=cfg["grad_clip"],
        log_interval=10**9,
        device=device,
        dtype=cfg["dtype"],
        save_interval=0,
        resume_path=None,
    )
    started = time.time()
    losses = train_epoch(0, loader, model, optimizer, scaler, args)
    elapsed = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if "cuda" in device else None

    exp_dir = Path(out_dir) / "exp_sft"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / f"{name}.csv"
    total_steps = args.epochs * len(loader)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss", "lr"])
        for step, loss in enumerate(losses, start=1):
            writer.writerow([step, f"{loss:.6f}", f"{get_lr(step, total_steps, cfg['lr']):.8e}"])

    summary = {
        "name": name,
        "steps": len(losses),
        "params_m": params,
        "loss_1": _loss_at(losses, 1),
        "loss_50": _loss_at(losses, 50),
        "loss_100": _loss_at(losses, 100),
        "loss_last": losses[-1],
        "loss_best": min(losses),
        "seconds": elapsed,
        "peak_mb": peak_mb,
        "csv": str(csv_path),
    }
    print(
        f"  {name:12s} steps={summary['steps']:<4d} "
        f"loss@1={summary['loss_1']:.3f} "
        f"loss@50={summary['loss_50'] if summary['loss_50'] is not None else '-':>7} "
        f"loss@100={summary['loss_100'] if summary['loss_100'] is not None else '-':>7} "
        f"last={summary['loss_last']:.3f} "
        f"{summary['seconds']:5.1f}s params={params:.2f}M "
        f"peak={f'{peak_mb:.0f}MB' if peak_mb else '-'}"
    )
    return summary


def _print_table(summaries: list[dict]):
    print()
    print(f"{'name':12s} {'steps':>5s} {'loss@1':>8s} {'loss@50':>9s} {'loss@100':>9s} {'last':>8s} {'time(s)':>8s} {'params':>8s}")
    for s in summaries:
        def f(v):
            return f"{v:.3f}" if v is not None else "-"
        print(
            f"{s['name']:12s} {s['steps']:>5d} {f(s['loss_1']):>8s} {f(s['loss_50']):>9s} "
            f"{f(s['loss_100']):>9s} {f(s['loss_last']):>8s} {s['seconds']:>8.1f} {s['params_m']:>7.2f}M"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MiniMind SFT ablations（D11）")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", nargs="+", default=[])
    parser.add_argument("--group", nargs="+", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="只跑前 N 步（冒烟）")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_group", default=None)
    args = parser.parse_args(argv)

    if args.list:
        for name, cfg in EXPERIMENTS.items():
            print(
                f"{name:12s} group={cfg['group']:8s} data={Path(cfg['data_path']).name:20s} "
                f"lr={cfg['lr']:.0e} pretrain={cfg['pretrain_weight']} hidden={cfg['hidden_size']}"
            )
        return 0

    if args.plot:
        names = GROUPS.get(args.plot_group) if args.plot_group else None
        plot_curves(Path(args.out_dir) / "exp_sft", Path(args.out_dir) / "exp_sft" / "curves.png", names)
        return 0

    names = list(args.run)
    for group in args.group:
        if group not in GROUPS:
            print(f"未知 group：{group}，可选 {list(GROUPS)}")
            return 2
        names.extend(GROUPS[group])
    if args.all:
        names.extend(EXPERIMENTS.keys())
    names = list(dict.fromkeys(names))
    if not names:
        parser.print_help()
        return 2

    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"未知实验：{unknown}，用 --list 查看")
        return 2

    print(f"device={BASE['device']} | 运行 {len(names)} 个实验 | 结果目录 {Path(args.out_dir) / 'exp_sft'}")
    summaries = []
    for name in names:
        summaries.append(run_one(name, out_dir=args.out_dir, max_steps=args.steps))
    _print_table(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
