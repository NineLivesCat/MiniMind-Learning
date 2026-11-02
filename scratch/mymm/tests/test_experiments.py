"""D8 验收测试：对照实验运行器 + CSV + 曲线

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_experiments.py

覆盖：
    1) 实验配置完整（lr/batch/dtype/mid 四组，只改一个变量）
    2) run_one 能训练、写 CSV、返回汇总
    3) CSV 格式正确（step/loss/lr，loss 有限、lr 递减）
    4) plot_curves 能生成 PNG
"""
from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

from mymm.experiments import EXPERIMENTS, GROUPS, plot_curves, run_one  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REQUIRED_KEYS = {
    "data_path",
    "tokenizer_dir",
    "hidden_size",
    "num_hidden_layers",
    "max_seq_len",
    "batch_size",
    "accumulation_steps",
    "lr",
    "epochs",
    "dtype",
    "grad_clip",
    "device",
    "seed",
}

_STATE: dict = {}


def work_dir() -> Path:
    if "dir" not in _STATE:
        env = os.environ.get("MYMM_TEST_DIR")
        path = Path(env) if env else Path(tempfile.mkdtemp())
        path.mkdir(parents=True, exist_ok=True)
        _STATE["dir"] = path
    return _STATE["dir"]


# ============================================================================
def test_specs():
    if len(GROUPS["lr"]) != 3:
        raise AssertionError(f"lr 组应有 3 个实验，实际 {GROUPS['lr']}")
    if len(GROUPS["batch"]) != 3:
        raise AssertionError(f"batch 组应有 3 个实验，实际 {GROUPS['batch']}")
    if len(GROUPS["dtype"]) != 2:
        raise AssertionError(f"dtype 组应有 2 个实验，实际 {GROUPS['dtype']}")
    if len(GROUPS["mid"]) != 1:
        raise AssertionError(f"mid 组应有 1 个实验，实际 {GROUPS['mid']}")

    for name, cfg in EXPERIMENTS.items():
        missing = REQUIRED_KEYS - set(cfg)
        if missing:
            raise AssertionError(f"{name} 缺少配置项 {sorted(missing)}")

    lr_values = sorted(EXPERIMENTS[n]["lr"] for n in GROUPS["lr"])
    if lr_values != [1e-4, 5e-4, 1e-3]:
        print(f"    lr 组实际学习率={lr_values}")
        raise AssertionError("lr 组学习率应为 1e-4 / 5e-4 / 1e-3")

    # 只改一个变量：lr 组除 lr 外配置应完全相同
    base = {k: v for k, v in EXPERIMENTS["lr_5e-4"].items() if k not in ("lr", "group")}
    for name in GROUPS["lr"]:
        cur = {k: v for k, v in EXPERIMENTS[name].items() if k not in ("lr", "group")}
        if cur != base:
            raise AssertionError(f"{name} 和 lr_5e-4 相比不止改了 lr")

    # batch 组：等效 batch = batch_size * accumulation_steps 应为 32
    for name in GROUPS["batch"]:
        cfg = EXPERIMENTS[name]
        effective = cfg["batch_size"] * cfg["accumulation_steps"]
        if effective != 32:
            print(f"    {name} 等效 batch={effective}")
            raise AssertionError("batch 组的等效 batch 应都是 32（batch_size * accumulation_steps）")


def test_run_one():
    work = work_dir()
    cfg = dict(
        data_path=r"D:\minimind\learning\data\pretrain_5000.jsonl",
        tokenizer_dir=r"D:\minimind\minimind\model",
        hidden_size=128,
        num_hidden_layers=2,
        max_seq_len=64,
        batch_size=8,
        accumulation_steps=1,
        lr=1e-3,
        epochs=1,
        dtype="bfloat16",
        grad_clip=1.0,
        device=DEVICE,
        seed=0,
    )
    t0 = time.time()
    summary = run_one("smoke", cfg, out_dir=str(work), max_steps=16)
    elapsed = time.time() - t0

    if summary["steps"] != 16:
        print(f"    steps={summary['steps']}，期望 16")
        raise AssertionError("max_steps 没有生效")
    if summary["params_m"] <= 0:
        raise AssertionError("参数量应大于 0")
    if not (0.0 < summary["loss_last"] < 100.0):
        print(f"    loss_last={summary['loss_last']}")
        raise AssertionError("loss 异常")
    if not math.isfinite(summary["loss_1"]):
        raise AssertionError("loss@1 不是有限值")
    csv_path = Path(summary["csv"])
    if not csv_path.exists():
        raise AssertionError(f"CSV 未生成：{csv_path}")
    print(f"      smoke: {summary['steps']} 步，{elapsed:.1f}s，loss {summary['loss_1']:.3f} -> {summary['loss_last']:.3f}")
    _STATE["csv"] = csv_path


def test_csv_format():
    csv_path = _STATE.get("csv")
    if csv_path is None:
        test_run_one()
        csv_path = _STATE["csv"]

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise AssertionError("CSV 没有数据行")
    if list(rows[0].keys()) != ["step", "loss", "lr"]:
        print(f"    CSV 表头={list(rows[0].keys())}，期望 ['step', 'loss', 'lr']")
        raise AssertionError("CSV 表头不对")
    if len(rows) != 16:
        raise AssertionError(f"CSV 行数={len(rows)}，期望 16")

    losses = [float(r["loss"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]
    if not all(math.isfinite(x) for x in losses):
        raise AssertionError("CSV 里有 inf/nan")
    for a, b in zip(lrs, lrs[1:]):
        if b > a + 1e-12:
            print(f"    lr 不是递减：{a:.3e} -> {b:.3e}")
            raise AssertionError("余弦学习率应逐步递减")


def test_plot():
    work = work_dir()
    os.environ.setdefault("MPLCONFIGDIR", str(work / "mpl"))
    png = work / "curves.png"
    plot_curves(work / "exp", png, names=["smoke"])
    if not png.exists() or png.stat().st_size <= 0:
        raise AssertionError(f"PNG 未生成：{png}")
    print(f"      曲线：{png} ({png.stat().st_size / 1024:.1f} KB)")


# ============================================================================
TESTS = [test_specs, test_run_one, test_csv_format, test_plot]

HINTS = {
    "test_specs": "四组实验只改一个变量；batch 组等效 batch 都是 32",
    "test_run_one": "run_one 要先建 tokenizer/dataset/model，再调 train_epoch 并写 CSV",
    "test_csv_format": "CSV 表头 step,loss,lr；lr 来自 get_lr",
    "test_plot": "plot_curves 需要 matplotlib（Agg），输出 PNG 到指定路径",
}


def main() -> int:
    filters = [a.lower() for a in sys.argv[1:] if not a.startswith("-")]
    selected = [t for t in TESTS if not filters or any(f in t.__name__.lower() for f in filters)]
    if not selected:
        print(f"没有匹配 {filters} 的测试，可选：{[t.__name__ for t in TESTS]}")
        return 2

    print(f"device = {DEVICE} | torch = {torch.__version__} | 运行 {len(selected)} 项")
    failed = []
    for fn in selected:
        t0 = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:18s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:18s} ({time.time() - t0:.1f}s)")

    print("-" * 60)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D8 的实验运行器可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
