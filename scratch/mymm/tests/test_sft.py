"""D10 验收测试：SFT 数据 + label mask + 训练接线

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_sft.py

覆盖：
    1) SFTDataset 样本形状 / pad / label 只覆盖 assistant 段
    2) label 与仓库 generate_labels 逐元素一致
    3) SFT 数据能跑 train_epoch（loss 有限）
    4) train_sft.main 能加载 pretrain 权重并保存 full_sft 权重
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
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

import mymm.train_sft as ts  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.data import SFTDataset  # noqa: E402
from mymm.model import CausalLM  # noqa: E402
from mymm.train import train_epoch  # noqa: E402

from dataset.lm_dataset import SFTDataset as RefSFTDataset  # noqa: E402

SFT_DATA = Path(r"D:\minimind\learning\data\sft_2000.jsonl")
TOKENIZER_DIR = REPO / "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STATE: dict = {}


def work_dir() -> Path:
    if "dir" not in _STATE:
        env = os.environ.get("MYMM_TEST_DIR")
        path = Path(env) if env else Path(tempfile.mkdtemp())
        path.mkdir(parents=True, exist_ok=True)
        _STATE["dir"] = path
    return _STATE["dir"]


def get_tokenizer():
    if "tok" not in _STATE:
        _STATE["tok"] = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return _STATE["tok"]


def show_diff(label: str, got, want, atol: float = 1e-6):
    g = got.detach().to(torch.float32).cpu()
    w = want.detach().to(torch.float32).cpu()
    if g.shape != w.shape:
        print(f"    ┌─ {label}")
        print(f"    │  实际 shape={tuple(g.shape)}  期望 shape={tuple(w.shape)}")
        print("    └─")
        raise AssertionError(f"{label}: shape 不一致")
    diff = (g - w).abs()
    maxd = diff.max().item() if diff.numel() else 0.0
    if maxd <= atol:
        return
    print(f"    ┌─ {label}")
    print(f"    │  max|实际-期望| = {maxd:.3e}")
    bad = (diff > atol).nonzero()
    for idx in bad[:8]:
        key = tuple(idx.tolist())
        print(f"    │    位置 {key}: 实际={g[key].item():.6g}  期望={w[key].item():.6g}")
    print("    └─")
    raise AssertionError(f"{label}: 数值不一致 (max|diff|={maxd:.3e})")


def find_sub(seq: list[int], sub: list[int], start: int = 0) -> int:
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return -1


# ============================================================================
def test_sft_dataset():
    tok = get_tokenizer()
    ds = SFTDataset(str(SFT_DATA), tok, max_length=128)
    if len(ds) != 2000:
        print(f"    样本数={len(ds)}，期望 2000")
        raise AssertionError("SFTDataset 长度不对")

    input_ids, labels = ds[0]
    if tuple(input_ids.shape) != (128,) or tuple(labels.shape) != (128,):
        print(f"    input_ids.shape={tuple(input_ids.shape)} labels.shape={tuple(labels.shape)}")
        raise AssertionError("样本形状应为 (max_length,)")
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise AssertionError("dtype 应为 torch.long")

    mask = labels != -100
    if int(mask.sum()) == 0:
        raise AssertionError("一条 label 都没留下：assistant 段没找到")
    show_diff("mask 位置的 label 应等于 input_ids", labels[mask], input_ids[mask], atol=0)

    pad_pos = input_ids == tok.pad_token_id
    if pad_pos.any():
        if not torch.all(labels[pad_pos] == -100):
            raise AssertionError("pad 位置的 label 必须是 -100")

    ids_list = input_ids.tolist()
    bos_pos = find_sub(ids_list, ds.bos_id)
    if bos_pos < 0:
        raise AssertionError("input_ids 里找不到 '<|im_start|>assistant\\n' 标记")
    assistant_start = bos_pos + len(ds.bos_id)
    if not torch.all(labels[:assistant_start] == -100):
        print(f"    assistant 起始位置={assistant_start}，之前的 label 应为 -100")
        raise AssertionError("assistant 之前的 label 应全是 -100")
    if assistant_start < 128 and int(labels[assistant_start]) == -100:
        raise AssertionError("assistant 第一段内容的 label 应该参与 loss")

    print(f"      非 -100 label 数={int(mask.sum())}，assistant 起始={assistant_start}")


def test_sft_label_parity():
    tok = get_tokenizer()
    ds = SFTDataset(str(SFT_DATA), tok, max_length=128)

    ref = object.__new__(RefSFTDataset)  # 只为调用 generate_labels，不触发 datasets 加载
    ref.bos_id = tok(f"{tok.bos_token}assistant\n", add_special_tokens=False).input_ids
    ref.eos_id = tok(f"{tok.eos_token}\n", add_special_tokens=False).input_ids
    ref.max_length = 128

    for i in range(3):
        input_ids, labels = ds[i]
        want = torch.tensor(ref.generate_labels(input_ids.tolist()), dtype=torch.long)
        show_diff(f"第 {i} 条 labels vs 仓库 generate_labels", labels, want, atol=0)


def test_sft_train_epoch():
    torch.manual_seed(0)
    tok = get_tokenizer()
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    model = CausalLM(cfg).to(DEVICE)
    subset = Subset(SFTDataset(str(SFT_DATA), tok, max_length=128), list(range(64)))
    loader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = SimpleNamespace(
        lr=1e-4,
        epochs=1,
        accumulation_steps=1,
        grad_clip=1.0,
        log_interval=1000,
        device=DEVICE,
        dtype="bfloat16",
        save_interval=0,
        resume_path=None,
    )
    t0 = time.time()
    losses = train_epoch(0, loader, model, optimizer, scaler, args)
    if len(losses) != len(loader):
        raise AssertionError(f"返回 {len(losses)} 个 loss，期望 {len(loader)}")
    if not all(torch.isfinite(torch.tensor(x)) for x in losses):
        raise AssertionError("loss 出现 inf/nan")
    print(f"      SFT smoke: {len(losses)} 步，{time.time() - t0:.1f}s，loss {losses[0]:.3f} -> {losses[-1]:.3f}")


def test_sft_main():
    work = work_dir() / "sft_main"
    work.mkdir(parents=True, exist_ok=True)
    save_dir = work / "run"
    save_dir.mkdir(parents=True, exist_ok=True)

    sft_path = work / "tiny_sft.jsonl"
    with open(sft_path, "w", encoding="utf-8") as fh:
        for i in range(8):
            conv = [
                {"role": "user", "content": f"第 {i} 个问题：你好吗？"},
                {"role": "assistant", "content": f"我很好，谢谢关心（{i}）。"},
            ]
            fh.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")

    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    pretrain = CausalLM(cfg)
    with torch.no_grad():
        for p in pretrain.parameters():
            p.fill_(0.01)
    pretrain_path = save_dir / "pretrain_128.pth"
    torch.save({k: v.half().cpu() for k, v in pretrain.state_dict().items()}, pretrain_path)

    captured = {}
    real_train_epoch = ts.train_epoch

    def fake_train_epoch(epoch, loader, model, optimizer, scaler, args, start_step=0, iters=None):
        captured["embed"] = model.model.embed_tokens.weight.detach().cpu().half().clone()
        captured["batches"] = len(loader)
        return []

    ts.train_epoch = fake_train_epoch
    try:
        argv = [
            "--data_path", str(sft_path),
            "--tokenizer_dir", str(TOKENIZER_DIR),
            "--save_dir", str(save_dir),
            "--save_weight", "full_sft",
            "--from_weight", "pretrain",
            "--hidden_size", "128",
            "--num_hidden_layers", "2",
            "--max_seq_len", "128",
            "--batch_size", "4",
            "--accumulation_steps", "1",
            "--epochs", "1",
            "--lr", "1e-5",
            "--log_interval", "1000",
            "--save_interval", "1000",
            "--num_workers", "0",
            "--device", DEVICE,
            "--dtype", "bfloat16",
        ]
        rc = ts.main(argv)
    finally:
        ts.train_epoch = real_train_epoch

    if rc != 0:
        raise AssertionError(f"main 返回 {rc}")
    if "embed" not in captured:
        raise AssertionError("main 没有调用 train_epoch")
    show_diff("加载的 pretrain 权重", captured["embed"], pretrain.state_dict()["model.embed_tokens.weight"].half(), atol=0)

    out_path = save_dir / "full_sft_128.pth"
    if not out_path.exists():
        raise AssertionError(f"没有保存 SFT 权重：{out_path}")
    resume = save_dir / "full_sft_128_resume.pth"
    if not resume.exists():
        raise AssertionError("没有保存 SFT resume checkpoint")
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    if payload["epoch"] != 1 or payload["step"] != 0:
        print(f"    resume epoch/step={payload['epoch']}/{payload['step']}，期望 1/0")
        raise AssertionError("epoch 结束时 resume checkpoint 不对")


# ============================================================================
TESTS = [test_sft_dataset, test_sft_label_parity, test_sft_train_epoch, test_sft_main]

HINTS = {
    "test_sft_dataset": "先 apply_chat_template，再 tokenizer(prompt)，补 pad；labels 只填 assistant 段",
    "test_sft_label_parity": "扫描 '<|im_start|>assistant\\n' 到 '<|im_end|>\\n' 的区间；和仓库 generate_labels 对齐",
    "test_sft_train_epoch": "train_epoch 直接用 SFTDataset 的 (ids, labels) 即可，-100 由 CE 忽略",
    "test_sft_main": "main 里 load_state_dict 加载 pretrain；注意 checkpoint 是 fp16",
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
    print(f"全部 {len(selected)} 项通过，D10 可以勾选 S5-01 / S5-02 / S5-03 / S5-04。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
