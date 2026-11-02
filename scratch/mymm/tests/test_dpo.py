"""D14 验收测试：DPO 数据 + dpo_loss + DPO 训练 + margin 统计

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_dpo.py

覆盖：
    1) DPODataset：x/y/mask 形状、next-token 对齐、mask 只覆盖 assistant
    2) dpo_loss：公式手算；policy==ref 时 loss=ln2、margin=0
    3) sequence_logps：与手动 log_softmax + gather 一致
    4) train_dpo.main：能训练并保存 dpo 权重
    5) evaluate_margins / dpo_metrics.main：同一模型 margin≈0
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

import mymm.dpo_metrics as dm  # noqa: E402
import mymm.train_dpo as td  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.data import DPODataset  # noqa: E402
from mymm.dpo import dpo_loss, sequence_logps  # noqa: E402
from mymm.model import CausalLM  # noqa: E402

DPO_DATA = Path(r"D:\minimind\learning\data\dpo_500.jsonl")
TOKENIZER_DIR = REPO / "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STATE: dict = {}


def show_diff(label: str, got, want, atol: float = 1e-6):
    g = got.detach().to(torch.float32).cpu()
    w = want.detach().to(torch.float32).cpu()
    if g.shape != w.shape:
        print(f"    ┌─ {label}：shape {tuple(g.shape)} vs {tuple(w.shape)}")
        raise AssertionError(f"{label}: shape 不一致")
    diff = (g - w).abs()
    maxd = diff.max().item() if diff.numel() else 0.0
    if maxd <= atol:
        return
    print(f"    ┌─ {label}")
    print(f"    │  max|实际-期望| = {maxd:.3e}")
    bad = (diff > atol).nonzero()
    for idx in bad[:6]:
        key = tuple(idx.tolist())
        print(f"    │    位置 {key}: 实际={g[key].item():.6g}  期望={w[key].item():.6g}")
    print("    └─")
    raise AssertionError(f"{label}: 数值不一致 (max|diff|={maxd:.3e})")


def work_dir() -> Path:
    if "dir" not in _STATE:
        env = os.environ.get("MYMM_TEST_DIR")
        path = Path(env) if env else Path(tempfile.mkdtemp())
        path.mkdir(parents=True, exist_ok=True)
        _STATE["dir"] = path
    return _STATE["dir"]


def get_tokenizer():
    if "tok" not in _STATE:
        from transformers import AutoTokenizer

        _STATE["tok"] = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return _STATE["tok"]


# ============================================================================
def test_dpo_dataset():
    tok = get_tokenizer()
    ds = DPODataset(str(DPO_DATA), tok, max_length=256)
    if len(ds) != 500:
        print(f"    样本数={len(ds)}，期望 500")
        raise AssertionError("DPODataset 长度不对")

    item = ds[0]
    keys = {"x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected"}
    if set(item) != keys:
        print(f"    返回的 key={sorted(item)}")
        raise AssertionError("__getitem__ 返回的 key 不对")

    for name, tensor in item.items():
        if tuple(tensor.shape) != (255,):
            print(f"    {name}.shape={tuple(tensor.shape)}，期望 (255,)")
            raise AssertionError("张量形状应为 (max_length-1,)")
        if tensor.dtype != torch.long:
            raise AssertionError(f"{name} 应为 torch.long")

    for prefix in ("chosen", "rejected"):
        x = item[f"x_{prefix}"]
        y = item[f"y_{prefix}"]
        mask = item[f"mask_{prefix}"]
        show_diff(f"{prefix}: y 应等于 x 左移一位", y[:-1], x[1:], atol=0)
        if not torch.all((mask == 0) | (mask == 1)):
            raise AssertionError("mask 只能是 0/1")
        if int(mask.sum()) <= 0:
            raise AssertionError(f"{prefix} 的 mask 全是 0：没找到 assistant 段")
        if int(mask.sum()) >= 255:
            raise AssertionError(f"{prefix} 的 mask 全 1：user 段不应该算 loss")

    if torch.equal(item["x_chosen"], item["x_rejected"]):
        raise AssertionError("chosen 和 rejected 应该不同")
    print(f"      mask_chosen={int(item['mask_chosen'].sum())} mask_rejected={int(item['mask_rejected'].sum())}")


def test_dpo_loss():
    policy_chosen = torch.tensor([2.0, 1.0])
    policy_rejected = torch.tensor([0.5, 0.2])
    ref_chosen = torch.tensor([0.3, 0.9])
    ref_rejected = torch.tensor([0.1, 0.8])
    beta = 0.1

    out = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    show_diff("dpo loss 公式", out["loss"], -F.logsigmoid(logits).mean(), atol=1e-6)

    chosen_rewards = beta * (policy_chosen - ref_chosen)
    rejected_rewards = beta * (policy_rejected - ref_rejected)
    margin = chosen_rewards - rejected_rewards
    show_diff("reward margin", out["reward_margin"], margin, atol=1e-6)
    show_diff("accuracy", out["accuracy"], (margin > 0).float().mean(), atol=1e-6)

    same = dpo_loss(policy_chosen, policy_rejected, policy_chosen, policy_rejected)
    show_diff("policy==ref 时 loss=ln2", same["loss"], torch.tensor(math.log(2)), atol=1e-6)
    show_diff("policy==ref 时 margin=0", same["reward_margin"], torch.zeros(2), atol=1e-6)


def test_sequence_logps():
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2)).eval().to(DEVICE)
    x = torch.randint(0, 6400, (2, 6), device=DEVICE)
    y = torch.randint(0, 6400, (2, 6), device=DEVICE)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]], device=DEVICE, dtype=torch.long)

    got = sequence_logps(model, x, y, mask)
    with torch.no_grad():
        logits = model(x).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logps = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
        want = (token_logps * mask).sum(dim=-1)
    show_diff("sequence_logps", got, want, atol=1e-5)
    print(f"      logps={got.tolist()}")


def _make_tiny_dpo(path: Path, n: int = 8) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            chosen = [
                {"role": "user", "content": f"感冒了怎么办（{i}）？"},
                {"role": "assistant", "content": "多喝水、多休息，必要时及时就医。"},
            ]
            rejected = [
                {"role": "user", "content": f"感冒了怎么办（{i}）？"},
                {"role": "assistant", "content": "不知道。"},
            ]
            fh.write(json.dumps({"chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
    return path


def test_train_dpo_main():
    work = work_dir() / "train_dpo"
    work.mkdir(parents=True, exist_ok=True)
    base = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2))
    torch.save({k: v.half().cpu() for k, v in base.state_dict().items()}, work / "full_sft_128.pth")
    data_path = _make_tiny_dpo(work / "tiny_dpo.jsonl", n=8)

    argv = [
        "--data_path", str(data_path),
        "--tokenizer_dir", str(TOKENIZER_DIR),
        "--save_dir", str(work),
        "--hidden_size", "128",
        "--num_hidden_layers", "2",
        "--max_seq_len", "64",
        "--batch_size", "2",
        "--accumulation_steps", "2",
        "--epochs", "1",
        "--lr", "4e-8",
        "--beta", "0.1",
        "--log_interval", "1000",
        "--num_workers", "0",
        "--device", DEVICE,
        "--dtype", "bfloat16",
    ]
    t0 = time.time()
    rc = td.main(argv)
    if rc != 0:
        raise AssertionError(f"train_dpo.main 返回 {rc}")
    out_path = work / "dpo_128.pth"
    if not out_path.exists():
        raise AssertionError("没有保存 dpo 权重")
    model = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2))
    model.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True), strict=True)
    print(f"      dpo_128.pth ({out_path.stat().st_size / 1024:.1f} KB) 生成于 {time.time() - t0:.1f}s")


def test_metrics():
    tok = get_tokenizer()
    dataset = DPODataset(str(DPO_DATA), tok, max_length=128)
    subset = Subset(dataset, list(range(8)))
    loader = DataLoader(subset, batch_size=2, shuffle=False)

    model = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2)).eval().to(DEVICE)
    stats = dm.evaluate_margins(model, model, loader, beta=0.1, device=DEVICE)
    if abs(stats["margin"]) > 1e-6:
        print(f"    同一模型 margin={stats['margin']:.3e}，期望 0")
        raise AssertionError("policy==ref 时 margin 应为 0")
    if stats["accuracy"] != 0.0:
        print(f"    同一模型 accuracy={stats['accuracy']}，期望 0")
        raise AssertionError("policy==ref 时 margin 全 0，严格大于 0 的比例应为 0")
    if stats["samples"] != 8:
        raise AssertionError("统计样本数不对")

    work = work_dir() / "metrics"
    work.mkdir(parents=True, exist_ok=True)
    base = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2))
    torch.save({k: v.half().cpu() for k, v in base.state_dict().items()}, work / "full_sft_128.pth")

    argv = [
        "--save_dir", str(work),
        "--before", "full_sft",
        "--after", "full_sft",
        "--ref", "full_sft",
        "--hidden_size", "128",
        "--num_hidden_layers", "2",
        "--max_seq_len", "128",
        "--batch_size", "2",
        "--num_samples", "4",
        "--device", DEVICE,
    ]
    rc = dm.main(argv)
    if rc != 0:
        raise AssertionError(f"dpo_metrics.main 返回 {rc}")
    print(f"      margin={stats['margin']:.2e} acc={stats['accuracy']:.1f}")


# ============================================================================
TESTS = [
    test_dpo_dataset,
    test_dpo_loss,
    test_sequence_logps,
    test_train_dpo_main,
    test_metrics,
]

HINTS = {
    "test_dpo_dataset": "复用 _encode_and_mask 得到 ids/mask，再统一左移一位：x=ids[:-1], y=ids[1:], mask=mask[1:]",
    "test_dpo_loss": "logits = beta*((πc-πr)-(refc-refr))；loss=-logsigmoid(logits).mean()；margin 是逐样本的",
    "test_sequence_logps": "log_softmax 后 gather 目标 token，乘 mask 求和",
    "test_train_dpo_main": "policy 可训练 + ref 冻结；保存 dpo_{hidden}.pth",
    "test_metrics": "同一模型作 policy/ref 时 margin 必须为 0；accuracy 用严格 > 0",
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
            print(f"FAIL  {fn.__name__:22s} 未实现（{exc}）")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            kind = "FAIL " if isinstance(exc, AssertionError) else "ERROR"
            print(f"{kind} {fn.__name__:22s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:22s} ({time.time() - t0:.1f}s)")

    print("-" * 60)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D14 可以勾选 S7-07 / S7-08 / S7-09 / S7-10。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
