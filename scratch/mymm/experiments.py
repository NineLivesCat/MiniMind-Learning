"""D8 · 预训练对照实验（LEARNING_TASKS: S4-04 / S4-05 / S4-06 / S4-07 / S4-08）

目的不是写新算法，而是**固定其它变量、每次只改一个**，把 loss 曲线跑出来：
    lr   : 1e-3 / 5e-4 / 1e-4
    batch: bs16+accum2 / bs8+accum4 / bs32+accum1
    dtype: bfloat16 / float16
    mid  : 256 维 4 层（接近真实结构）
    big  : 512 维 8 层（可选，为全量 768/8 估时）

用法（在 learning\\scratch 下）：
    python -m mymm.experiments --list
    python -m mymm.experiments --group lr
    python -m mymm.experiments --group batch dtype
    python -m mymm.experiments --run mid_256x4
    python -m mymm.experiments --all
    python -m mymm.experiments --plot                    # 画 out/exp/curves.png
    python -m mymm.experiments --plot --plot_group lr    # 只画 lr 组

每个实验会把逐步 loss/lr 写到 out/exp/<name>.csv，并在终端打印汇总表
（loss@1 / loss@100 / loss@300 / last / 耗时 / 参数 / 显存峰值）。
"""
from __future__ import annotations

import argparse
import csv
import math
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
from .data import PretrainDataset  # noqa: E402
from .model import CausalLM  # noqa: E402
from .train import get_lr, train_epoch  # noqa: E402

DATA = r"D:\minimind\learning\data\pretrain_5000.jsonl"
TOKENIZER_DIR = r"D:\minimind\minimind\model"

BASE = dict(
    data_path=DATA,
    tokenizer_dir=TOKENIZER_DIR,
    hidden_size=128,
    num_hidden_layers=2,
    max_seq_len=128,
    batch_size=16,
    accumulation_steps=2,
    lr=5e-4,
    epochs=1,
    dtype="bfloat16",
    grad_clip=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
)

EXPERIMENTS = {
    "lr_1e-3": {**BASE, "lr": 1e-3, "group": "lr"},
    "lr_5e-4": {**BASE, "lr": 5e-4, "group": "lr"},
    "lr_1e-4": {**BASE, "lr": 1e-4, "group": "lr"},
    "bs16_acc2": {**BASE, "group": "batch"},
    "bs8_acc4": {**BASE, "batch_size": 8, "accumulation_steps": 4, "group": "batch"},
    "bs32_acc1": {**BASE, "batch_size": 32, "accumulation_steps": 1, "group": "batch"},
    "bf16": {**BASE, "dtype": "bfloat16", "group": "dtype"},
    "fp16": {**BASE, "dtype": "float16", "group": "dtype"},
    "mid_256x4": {
        **BASE,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "max_seq_len": 256,
        "batch_size": 8,
        "accumulation_steps": 2,
        "group": "mid",
    },
    "big_512x8": {
        **BASE,
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "max_seq_len": 340,
        "batch_size": 4,
        "accumulation_steps": 4,
        "group": "big",
    },
}

GROUPS = {
    "lr": [n for n, c in EXPERIMENTS.items() if c["group"] == "lr"],
    "batch": [n for n, c in EXPERIMENTS.items() if c["group"] == "batch"],
    "dtype": [n for n, c in EXPERIMENTS.items() if c["group"] == "dtype"],
    "mid": [n for n, c in EXPERIMENTS.items() if c["group"] == "mid"],
    "big": [n for n, c in EXPERIMENTS.items() if c["group"] == "big"],
}


def _loss_at(losses: list[float], step: int):
    return losses[step - 1] if 0 < step <= len(losses) else None


def run_one(name: str, cfg: dict | None = None, out_dir: str = "out", max_steps: int = 0) -> dict:
    """跑一个实验：训练、存 CSV、返回汇总字典。

    参数：
    - name: 实验名（CSV 文件名）
    - cfg: 配置字典；None 时从 EXPERIMENTS[name] 取
    - out_dir: 结果目录，CSV 写到 out_dir/exp/<name>.csv
    - max_steps: >0 时只跑这么多步（快速冒烟；CSV 里也标出来）
    """
    cfg = dict(cfg or EXPERIMENTS[name])
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])
    dataset = PretrainDataset(cfg["data_path"], tokenizer, max_length=cfg["max_seq_len"])
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
    model = CausalLM(config).to(device)
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
    )

    started = time.time()
    losses = train_epoch(0, loader, model, optimizer, scaler, args)
    elapsed = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if "cuda" in device else None

    exp_dir = Path(out_dir) / "exp"
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
        "loss_100": _loss_at(losses, 100),
        "loss_300": _loss_at(losses, 300),
        "loss_last": losses[-1],
        "loss_best": min(losses),
        "seconds": elapsed,
        "peak_mb": peak_mb,
        "csv": str(csv_path),
    }
    print(
        f"  {name:12s} steps={summary['steps']:<4d} "
        f"loss@1={summary['loss_1']:.3f} "
        f"loss@100={summary['loss_100'] if summary['loss_100'] is not None else '-':>7} "
        f"loss@300={summary['loss_300'] if summary['loss_300'] is not None else '-':>7} "
        f"last={summary['loss_last']:.3f} "
        f"{summary['seconds']:5.1f}s params={params:.2f}M "
        f"peak={f'{peak_mb:.0f}MB' if peak_mb else '-'}"
    )
    return summary


def plot_curves(csv_dir: str | Path, out_png: str | Path, names: list[str] | None = None) -> Path:
    """把 CSV 里的 loss 曲线画到一张 PNG（已给好，可直接用）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob("*.csv"))
    if names:
        files = [f for f in files if f.stem in names]
    if not files:
        raise FileNotFoundError(f"{csv_dir} 下没有可画的 CSV，先跑实验")

    plt.figure(figsize=(9, 5.5))
    for path in files:
        steps, losses = [], []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
        plt.plot(steps, losses, linewidth=1.4, label=path.stem)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("MiniMind pretrain ablations")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  曲线已保存：{out_png}")
    return out_png


def _print_table(summaries: list[dict]):
    print()
    print(f"{'name':12s} {'steps':>5s} {'loss@1':>8s} {'loss@100':>9s} {'loss@300':>9s} {'last':>8s} {'time(s)':>8s} {'params':>8s} {'peak':>7s}")
    for s in summaries:
        def f(v):
            return f"{v:.3f}" if v is not None else "-"
        peak = f"{s['peak_mb']:.0f}MB" if s["peak_mb"] else "-"
        print(
            f"{s['name']:12s} {s['steps']:>5d} {f(s['loss_1']):>8s} {f(s['loss_100']):>9s} "
            f"{f(s['loss_300']):>9s} {f(s['loss_last']):>8s} {s['seconds']:>8.1f} "
            f"{s['params_m']:>7.2f}M {peak:>7s}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MiniMind pretrain ablations（D8）")
    parser.add_argument("--list", action="store_true", help="列出所有实验")
    parser.add_argument("--run", nargs="+", default=[], help="跑指定实验名")
    parser.add_argument("--group", nargs="+", default=[], help="跑整组：lr / batch / dtype / mid / big")
    parser.add_argument("--all", action="store_true", help="跑全部")
    parser.add_argument("--steps", type=int, default=0, help="只跑前 N 步（冒烟）")
    parser.add_argument("--out_dir", default="out", help="结果目录（默认 out/exp）")
    parser.add_argument("--plot", action="store_true", help="画曲线")
    parser.add_argument("--plot_group", default=None, help="只画某一组的曲线")
    args = parser.parse_args(argv)

    if args.list:
        for name, cfg in EXPERIMENTS.items():
            print(
                f"{name:12s} group={cfg['group']:5s} hidden={cfg['hidden_size']:<4d} "
                f"layers={cfg['num_hidden_layers']} seq={cfg['max_seq_len']:<4d} "
                f"bs={cfg['batch_size']:<3d} accum={cfg['accumulation_steps']} "
                f"lr={cfg['lr']:.0e} dtype={cfg['dtype']}"
            )
        return 0

    if args.plot:
        names = GROUPS.get(args.plot_group) if args.plot_group else None
        plot_curves(Path(args.out_dir) / "exp", Path(args.out_dir) / "exp" / "curves.png", names)
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

    print(f"device={BASE['device']} | 运行 {len(names)} 个实验 | 结果目录 {Path(args.out_dir) / 'exp'}")
    summaries = []
    for name in names:
        summaries.append(run_one(name, out_dir=args.out_dir, max_steps=args.steps))
    _print_table(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
