"""D11 验收测试：SFT 学习率对照实验运行器

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_sft_experiments.py

覆盖：
    1) 实验配置：lr 组 3 个（1e-4/1e-5/1e-6），control/custom 组存在
    2) run_one：加载 pretrain、训练、写 CSV、返回汇总
    3) CSV 格式与 lr 递减
    4) plot_curves 能出 PNG
"""
from __future__ import annotations

import csv
import json
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

import mymm.sft_experiments as se  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.experiments import plot_curves  # noqa: E402
from mymm.model import CausalLM  # noqa: E402
from mymm.sft_experiments import EXPERIMENTS, GROUPS, run_one  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_DIR = REPO / "model"
_STATE: dict = {}


def work_dir() -> Path:
    if "dir" not in _STATE:
        env = os.environ.get("MYMM_TEST_DIR")
        path = Path(env) if env else Path(tempfile.mkdtemp())
        path.mkdir(parents=True, exist_ok=True)
        _STATE["dir"] = path
    return _STATE["dir"]


def make_tiny_sft(path: Path, n: int = 8) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            conv = [
                {"role": "user", "content": f"第 {i} 个问题。"},
                {"role": "assistant", "content": f"这是第 {i} 个回答。"},
            ]
            fh.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    return path


def make_pretrain(save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    model = CausalLM(cfg)
    path = save_dir / "pretrain_128.pth"
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, path)
    return path


# ============================================================================
def test_specs():
    if len(GROUPS["lr"]) != 3:
        raise AssertionError(f"lr 组应有 3 个实验，实际 {GROUPS['lr']}")
    lrs = sorted(EXPERIMENTS[n]["lr"] for n in GROUPS["lr"])
    if lrs != [1e-6, 1e-5, 1e-4]:
        print(f"    lr 值={lrs}")
        raise AssertionError("lr 组应为 1e-6 / 1e-5 / 1e-4")
    if not GROUPS["control"] or EXPERIMENTS[GROUPS["control"][0]]["pretrain_weight"] != "none":
        raise AssertionError("control 组应是无预训练对照")
    if not GROUPS["custom"]:
        raise AssertionError("缺少 custom 组")
    if Path(EXPERIMENTS["custom_20"]["data_path"]).name != "my_sft_20.jsonl":
        raise AssertionError("custom 组应指向 my_sft_20.jsonl")

    # lr 组除 lr 外应完全相同
    base = {k: v for k, v in EXPERIMENTS["lr_1e-5"].items() if k not in ("lr", "group")}
    for name in GROUPS["lr"]:
        cur = {k: v for k, v in EXPERIMENTS[name].items() if k not in ("lr", "group")}
        if cur != base:
            raise AssertionError(f"{name} 和 lr_1e-5 相比不止改了 lr")


def test_run_one():
    work = work_dir() / "sft_exp"
    work.mkdir(parents=True, exist_ok=True)
    data_path = make_tiny_sft(work / "tiny_sft.jsonl", n=8)
    save_dir = work / "run"
    make_pretrain(save_dir)

    cfg = dict(se.BASE)
    cfg.update(
        data_path=str(data_path),
        save_dir=str(save_dir),
        device=DEVICE,
        max_seq_len=64,
        batch_size=4,
        accumulation_steps=1,
        epochs=1,
        lr=1e-4,
        seed=0,
    )
    t0 = time.time()
    summary = run_one("smoke", cfg, out_dir=str(work), max_steps=2)
    if summary["steps"] != 2:
        print(f"    steps={summary['steps']}，期望 2")
        raise AssertionError("max_steps 没有生效")
    if not (0.0 < summary["loss_last"] < 100.0):
        print(f"    loss_last={summary['loss_last']}")
        raise AssertionError("loss 异常")
    if summary["params_m"] <= 0:
        raise AssertionError("参数量应大于 0")
    csv_path = Path(summary["csv"])
    if not csv_path.exists():
        raise AssertionError(f"CSV 未生成：{csv_path}")
    print(f"      smoke: {summary['steps']} 步 {time.time() - t0:.1f}s，loss {summary['loss_1']:.3f} -> {summary['loss_last']:.3f}")
    _STATE["csv"] = csv_path
    _STATE["work"] = work


def test_csv_format():
    csv_path = _STATE.get("csv")
    if csv_path is None:
        test_run_one()
        csv_path = _STATE["csv"]

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 2:
        raise AssertionError(f"CSV 行数={len(rows)}，期望 2")
    if list(rows[0].keys()) != ["step", "loss", "lr"]:
        raise AssertionError(f"CSV 表头={list(rows[0].keys())}")
    losses = [float(r["loss"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]
    if not all(math.isfinite(x) for x in losses):
        raise AssertionError("CSV 里有 inf/nan")
    for a, b in zip(lrs, lrs[1:]):
        if b > a + 1e-12:
            raise AssertionError("lr 应逐步递减")


def test_plot():
    work = _STATE.get("work")
    if work is None:
        test_run_one()
        work = _STATE["work"]
    os.environ.setdefault("MPLCONFIGDIR", str(work / "mpl"))
    png = work / "curves.png"
    plot_curves(work / "exp_sft", png, names=["smoke"])
    if not png.exists() or png.stat().st_size <= 0:
        raise AssertionError(f"PNG 未生成：{png}")
    print(f"      曲线：{png} ({png.stat().st_size / 1024:.1f} KB)")


# ============================================================================
TESTS = [test_specs, test_run_one, test_csv_format, test_plot]

HINTS = {
    "test_specs": "lr 组只有 lr 不同；control 无预训练；custom 指向 my_sft_20.jsonl",
    "test_run_one": "run_one 要加载 pretrain_128.pth，用 train_epoch 训练并写 out/exp_sft/name.csv",
    "test_csv_format": "CSV 表头 step,loss,lr；lr 来自 get_lr，逐步递减",
    "test_plot": "plot_curves 读 CSV 画 PNG",
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
    print(f"全部 {len(selected)} 项通过，D11 的 SFT 实验运行器可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
