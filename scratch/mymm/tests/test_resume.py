"""D9 验收测试：断点续训（SkipBatchSampler + checkpoint + resume 接线）

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_resume.py

覆盖：
    1) SkipBatchSampler 跳过前 N 个 batch，顺序不变、len 正确
    2) train_epoch 按 --save_interval 存 resume checkpoint
    3) main --from_resume 1 能从 step=k 继续（跳过 k 个 batch，step 编号延续）
    4) 没有 resume 文件时从 step 0 开始
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import optim
from transformers import AutoTokenizer

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

import mymm.train as tr  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.model import CausalLM  # noqa: E402
from mymm.train import SkipBatchSampler, load_checkpoint, save_checkpoint, train_epoch  # noqa: E402

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


def make_tiny_jsonl(path: Path, n: int = 64) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"text": f"这是第 {i} 条用于断点续训测试的短文本。"}, ensure_ascii=False) + "\n")
    return path


# ============================================================================
def test_skip_sampler():
    indices = list(range(10))
    sampler = SkipBatchSampler(indices, batch_size=3, skip_batches=0)
    batches = list(sampler)
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    if batches != expected:
        print(f"    实际={batches}")
        print(f"    期望={expected}")
        raise AssertionError("SkipBatchSampler 切块/顺序不对")
    if len(sampler) != 4:
        print(f"    len={len(sampler)}，期望 4")
        raise AssertionError("SkipBatchSampler.__len__ 不对")

    skipped = SkipBatchSampler(indices, batch_size=3, skip_batches=2)
    got = list(skipped)
    if got != [[6, 7, 8], [9]]:
        print(f"    skip=2 实际={got}")
        raise AssertionError("SkipBatchSampler 跳过前 2 个 batch 的结果不对")
    if len(skipped) != 2:
        print(f"    skip=2 的 len={len(skipped)}，期望 2")
        raise AssertionError("skip 后 __len__ 不对")

    over = SkipBatchSampler(indices, batch_size=3, skip_batches=10)
    if list(over) != [] or len(over) != 0:
        raise AssertionError("skip 超过总 batch 数时应返回空")


def test_train_epoch_save_interval():
    torch.manual_seed(0)
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    model = CausalLM(cfg).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    ids = torch.randint(0, 6400, (4, 16))
    labels = ids.clone()
    loader = [(ids, labels) for _ in range(8)]

    resume_path = work_dir() / "save_test" / "pretrain_128_resume.pth"
    args = SimpleNamespace(
        lr=1e-3,
        epochs=1,
        accumulation_steps=1,
        grad_clip=1.0,
        log_interval=10**9,
        device=DEVICE,
        dtype="bfloat16",
        save_interval=4,
        resume_path=str(resume_path),
    )
    losses = train_epoch(0, loader, model, optimizer, scaler, args)
    if len(losses) != 8:
        raise AssertionError("train_epoch 应返回 8 个 loss")

    if not resume_path.exists():
        raise AssertionError(f"save_interval 没有生成 checkpoint：{resume_path}")
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    if payload["epoch"] != 0 or payload["step"] != 8:
        print(f"    checkpoint epoch/step = {payload['epoch']}/{payload['step']}，期望 0/8")
        raise AssertionError("save_interval 存下的 step 不对")
    if "optimizer" not in payload:
        raise AssertionError("resume checkpoint 应包含 optimizer 状态")


def test_resume_main():
    work = work_dir() / "resume_main"
    work.mkdir(parents=True, exist_ok=True)
    data_path = make_tiny_jsonl(work / "tiny.jsonl", n=64)
    save_dir = work / "run"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    model = CausalLM(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    resume_path = save_dir / "pretrain_128_resume.pth"
    save_checkpoint(resume_path, model, optimizer, None, epoch=0, step=8)

    calls = []
    real_train_epoch = tr.train_epoch

    def fake_train_epoch(epoch, loader, model, optimizer, scaler, args, start_step=0, iters=None):
        calls.append({"epoch": epoch, "start_step": start_step, "batches": len(loader), "iters": iters})
        return []

    tr.train_epoch = fake_train_epoch
    try:
        argv = [
            "--data_path", str(data_path),
            "--tokenizer_dir", str(TOKENIZER_DIR),
            "--save_dir", str(save_dir),
            "--hidden_size", "128",
            "--num_hidden_layers", "2",
            "--max_seq_len", "32",
            "--batch_size", "4",
            "--accumulation_steps", "1",
            "--epochs", "1",
            "--lr", "5e-4",
            "--log_interval", "1000",
            "--save_interval", "1000",
            "--from_resume", "1",
            "--num_workers", "0",
            "--device", DEVICE,
            "--dtype", "bfloat16",
            "--seed", "42",
        ]
        tr.main(argv)
    finally:
        tr.train_epoch = real_train_epoch

    if len(calls) != 1:
        print(f"    train_epoch 调用次数={len(calls)}，期望 1")
        raise AssertionError("main 应调用一次 train_epoch")
    call = calls[0]
    if call["start_step"] != 8:
        print(f"    start_step={call['start_step']}，期望 8")
        raise AssertionError("main 没有从 checkpoint 的 step 继续")
    if call["batches"] != 8:
        print(f"    loader 剩余 batch 数={call['batches']}，期望 8（16 总数 - 跳过 8）")
        raise AssertionError("续训时 loader 没有跳过已训练的 batch")
    if call["iters"] != 16:
        print(f"    iters={call['iters']}，期望 16（epoch 总 batch 数）")
        raise AssertionError("train_epoch 的 iters 应传总 batch 数")

    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    if payload["epoch"] != 1 or payload["step"] != 0:
        print(f"    epoch 结束后 resume = {payload['epoch']}/{payload['step']}，期望 1/0")
        raise AssertionError("epoch 结束时应存 epoch+1 / step=0")


def test_fresh_start():
    work = work_dir() / "fresh"
    work.mkdir(parents=True, exist_ok=True)
    data_path = make_tiny_jsonl(work / "tiny.jsonl", n=32)
    save_dir = work / "run"

    calls = []
    real_train_epoch = tr.train_epoch

    def fake_train_epoch(epoch, loader, model, optimizer, scaler, args, start_step=0, iters=None):
        calls.append({"start_step": start_step, "batches": len(loader), "iters": iters})
        return []

    tr.train_epoch = fake_train_epoch
    try:
        argv = [
            "--data_path", str(data_path),
            "--tokenizer_dir", str(TOKENIZER_DIR),
            "--save_dir", str(save_dir),
            "--hidden_size", "128",
            "--num_hidden_layers", "2",
            "--max_seq_len", "32",
            "--batch_size", "4",
            "--epochs", "1",
            "--log_interval", "1000",
            "--num_workers", "0",
            "--device", DEVICE,
            "--dtype", "bfloat16",
        ]
        tr.main(argv)
    finally:
        tr.train_epoch = real_train_epoch

    if not calls:
        raise AssertionError("main 没有调用 train_epoch")
    if calls[0]["start_step"] != 0:
        print(f"    无 resume 文件时 start_step={calls[0]['start_step']}，期望 0")
        raise AssertionError("没有 checkpoint 时应从 step 0 开始")
    if calls[0]["batches"] != 8:
        print(f"    32 条 / batch 4 应剩 8 个 batch，实际 {calls[0]['batches']}")
        raise AssertionError("正常启动时 loader 不应跳过 batch")


# ============================================================================
TESTS = [test_skip_sampler, test_train_epoch_save_interval, test_resume_main, test_fresh_start]

HINTS = {
    "test_skip_sampler": "按 batch_size 切块，跳过前 skip_batches 块；最后不足一块也要 yield；len = 总块数 - skip",
    "test_train_epoch_save_interval": "train_epoch 里每 save_interval 步调用 save_checkpoint(epoch, step)",
    "test_resume_main": "main 里 load_checkpoint 得到 (start_epoch, start_step)，用 SkipBatchSampler(skip=start_step) 建 loader，并传 iters=总 batch 数",
    "test_fresh_start": "没有 resume 文件时 start_epoch/start_step 应为 0，不跳过任何 batch",
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
        except NotImplementedError as exc:
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__:30s} 未实现（{exc}）")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:30s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:30s} ({time.time() - t0:.1f}s)")

    print("-" * 64)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D9 可以勾选 S4-09 / S4-11 / S4-12。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
