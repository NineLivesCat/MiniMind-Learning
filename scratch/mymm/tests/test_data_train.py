"""D7 验收测试：预训练数据 + 余弦调度 + 训练循环 + checkpoint

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_data_train.py

只跑某几项：
    python mymm\\tests\\test_data_train.py dataset
    python mymm\\tests\\test_data_train.py lr
    python mymm\\tests\\test_data_train.py train_epoch checkpoint

失败时打印期望/实际数值；实现前全部 FAIL（NotImplementedError）属正常。
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

from mymm.config import MiniMindConfig  # noqa: E402
from mymm.data import PretrainDataset  # noqa: E402
from mymm.model import CausalLM  # noqa: E402
from mymm.train import get_lr, load_checkpoint, save_checkpoint, train_epoch  # noqa: E402

DATA = Path(r"D:\minimind\learning\data\pretrain_5000.jsonl")
TOKENIZER_DIR = REPO / "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ============================================================================
# 诊断工具
# ============================================================================
def fmt(x: float) -> str:
    return f"{x:.6g}"


def show_diff(label: str, got, want, atol: float = 1e-6):
    if not torch.is_tensor(got) or not torch.is_tensor(want):
        raise AssertionError(f"{label}: 期望 Tensor，实际 got={type(got).__name__} want={type(want).__name__}")
    g = got.detach().to(torch.float32).cpu()
    w = want.detach().to(torch.float32).cpu()
    if g.shape != w.shape:
        print(f"    ┌─ {label}")
        print(f"    │  实际 shape={tuple(g.shape)}  期望 shape={tuple(w.shape)}")
        print("    └─")
        raise AssertionError(f"{label}: shape 不一致 {tuple(g.shape)} vs {tuple(w.shape)}")
    diff = (g - w).abs()
    maxd = diff.max().item() if diff.numel() else 0.0
    if maxd <= atol:
        return
    print(f"    ┌─ {label}")
    print(f"    │  max|实际-期望| = {maxd:.3e}   (容差 {atol:.0e})")
    bad = (diff > atol).nonzero()
    print(f"    │  不一致 {int((diff > atol).sum())} 处，前 {min(8, len(bad))} 个：")
    for idx in bad[:8]:
        key = tuple(idx.tolist())
        print(f"    │    位置 {key}: 实际={fmt(g[key].item())}  期望={fmt(w[key].item())}")
    print("    └─")
    raise AssertionError(f"{label}: 数值不一致 (max|diff|={maxd:.3e})")


_STATE: dict = {}


def get_tokenizer():
    if "tok" not in _STATE:
        _STATE["tok"] = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return _STATE["tok"]


def checkpoint_dir() -> Path:
    env = os.environ.get("MYMM_TEST_DIR")
    path = Path(env) if env else Path(tempfile.mkdtemp())
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# 数据（S4-03）
# ============================================================================
def test_dataset():
    tok = get_tokenizer()
    ds = PretrainDataset(str(DATA), tok, max_length=64)
    if len(ds) != 5000:
        print(f"    样本数={len(ds)}，期望 5000")
        raise AssertionError("PretrainDataset 长度不对")

    input_ids, labels = ds[0]
    if tuple(input_ids.shape) != (64,) or tuple(labels.shape) != (64,):
        print(f"    input_ids.shape={tuple(input_ids.shape)} labels.shape={tuple(labels.shape)}")
        raise AssertionError("样本形状应为 (max_length,)")
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise AssertionError("dtype 应为 torch.long")

    if int(input_ids[0]) != tok.bos_token_id:
        print(f"    input_ids[0]={int(input_ids[0])}，期望 bos={tok.bos_token_id}")
        raise AssertionError("开头不是 bos")

    n = int((labels != -100).sum())
    if not (2 <= n <= 64):
        print(f"    label 非 -100 数量={n}")
        raise AssertionError("有效 label 数量不对")
    if int(input_ids[n - 1]) != tok.eos_token_id:
        print(f"    input_ids[{n - 1}]={int(input_ids[n - 1])}，期望 eos={tok.eos_token_id}")
        raise AssertionError("有效 token 结尾不是 eos")
    if not torch.all(input_ids[n:] == tok.pad_token_id):
        raise AssertionError("eos 之后应全部是 pad")
    if not torch.all(labels[n:] == -100):
        raise AssertionError("pad 位置的 label 应为 -100")
    if not torch.equal(labels[:n], input_ids[:n]):
        raise AssertionError("非 pad 位置的 label 应等于 input_ids")


def _expected_item(text: str, tok, max_length: int):
    tokens = tok(text, add_special_tokens=False, max_length=max_length - 2, truncation=True).input_ids
    tokens = [tok.bos_token_id] + tokens + [tok.eos_token_id]
    input_ids = tokens + [tok.pad_token_id] * (max_length - len(tokens))
    ids = torch.tensor(input_ids, dtype=torch.long)
    labels = ids.clone()
    labels[ids == tok.pad_token_id] = -100
    return ids, labels


def test_dataset_parity():
    tok = get_tokenizer()
    ours = PretrainDataset(str(DATA), tok, max_length=128)
    for i in range(3):
        a_ids, a_labels = ours[i]
        b_ids, b_labels = _expected_item(ours.samples[i]["text"], tok, 128)
        show_diff(f"第 {i} 条 input_ids vs 独立实现", a_ids, b_ids, atol=0)
        show_diff(f"第 {i} 条 labels vs 独立实现", a_labels, b_labels, atol=0)


# ============================================================================
# 学习率（S4-02）
# ============================================================================
def test_lr():
    lr = 1e-3
    total = 100
    show_diff("step=0 应等于 base_lr", torch.tensor(get_lr(0, total, lr)), torch.tensor(lr), atol=1e-12)
    show_diff("step=total 应等于 0.1*base_lr", torch.tensor(get_lr(total, total, lr)), torch.tensor(0.1 * lr), atol=1e-12)

    mid = get_lr(total // 2, total, lr)
    expected = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * 0.5)))
    if abs(mid - expected) > 1e-12:
        print(f"    step=50 实际={mid:.8f}，期望={expected:.8f}")
        raise AssertionError("余弦公式不对")

    vals = [get_lr(s, total, lr) for s in range(0, total + 1)]
    for a, b in zip(vals, vals[1:]):
        if b > a + 1e-15:
            print(f"    学习率不是单调递减：{a:.6e} -> {b:.6e}")
            raise AssertionError("学习率应随 step 单调递减")

    if get_lr(0, 0, lr) != lr:
        raise AssertionError("total_steps<=0 时应返回 base_lr（防除零）")


# ============================================================================
# 训练循环（S4-01 / S4-03）
# ============================================================================
def test_train_epoch():
    torch.manual_seed(0)
    tok = get_tokenizer()
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    subset = Subset(PretrainDataset(str(DATA), tok, max_length=64), list(range(512)))
    loader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    model = CausalLM(cfg).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = SimpleNamespace(
        lr=1e-3,
        epochs=1,
        accumulation_steps=2,
        grad_clip=1.0,
        log_interval=1000,
        device=DEVICE,
        dtype="bfloat16",
    )

    t0 = time.time()
    losses = train_epoch(0, loader, model, optimizer, scaler, args)
    elapsed = time.time() - t0

    if len(losses) != len(loader):
        print(f"    返回 {len(losses)} 个 loss，期望 {len(loader)} 个")
        raise AssertionError("train_epoch 应返回每一步的 loss")
    if not all(math.isfinite(x) for x in losses):
        raise AssertionError("loss 出现 inf/nan")

    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    print(f"      loss {first:.3f} -> {last:.3f}（{len(losses)} 步，{elapsed:.1f}s）")
    if last > first - 0.02:
        raise AssertionError("训练后 loss 没有明显下降（检查 backward/step/梯度累积）")


def test_checkpoint():
    torch.manual_seed(1)
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    model = CausalLM(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    path = checkpoint_dir() / "pretrain_128_resume.pth"

    save_checkpoint(path, model, optimizer, None, epoch=1, step=23)
    if not path.exists():
        raise AssertionError(f"checkpoint 未生成：{path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["epoch"] != 1 or payload["step"] != 23:
        print(f"    epoch/step = {payload['epoch']}/{payload['step']}")
        raise AssertionError("checkpoint 里的 epoch/step 不对")
    if next(iter(payload["model"].values())).dtype != torch.float16:
        raise AssertionError("保存的模型权重应为 fp16")

    model2 = CausalLM(cfg)
    epoch, step = load_checkpoint(path, model2)
    if (epoch, step) != (1, 23):
        raise AssertionError("load_checkpoint 返回的 epoch/step 不对")
    for name, tensor in model.state_dict().items():
        show_diff(f"恢复后参数 {name}", model2.state_dict()[name], tensor.half(), atol=0)


# ============================================================================
TESTS = [
    test_dataset,
    test_dataset_parity,
    test_lr,
    test_train_epoch,
    test_checkpoint,
]

HINTS = {
    "test_dataset": "tokens = [bos]+tokenizer(...).input_ids+[eos]；pad 补到 max_length；labels pad 处 -100",
    "test_dataset_parity": "和仓库 PretrainDataset 逐元素一致；注意 add_special_tokens=False 与 max_length-2",
    "test_lr": "lr = base * (0.1 + 0.45 * (1 + cos(pi*step/total)))；total<=0 直接返回 base",
    "test_train_epoch": "loss/accum → backward；每 accum 步 unscale/clip/step/zero；日志 loss 要乘回 accum",
    "test_checkpoint": "save_checkpoint/load_checkpoint 已给；检查是否覆盖了实现",
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
            print(f"FAIL  {fn.__name__:24s} 未实现（{exc}）")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:24s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:24s} ({time.time() - t0:.1f}s)")

    print("-" * 64)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D7 可以勾选 S4-01 / S4-02 / S4-03，G4 的前半达成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
