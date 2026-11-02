"""D15 验收测试：白盒蒸馏

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_distill.py

覆盖：
    1) distillation_loss = T² * KL(teacher/T || student/T)
    2) distill_loss 的 shift / mask / CE / KD 组合
    3) train_distill.main 能训练并保存 distill 权重
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

import mymm.train_distill as tdist  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.distill import distill_loss, distillation_loss  # noqa: E402
from mymm.model import CausalLM  # noqa: E402

TOKENIZER_DIR = REPO / "model"
SFT_DATA = Path(r"D:\minimind\learning\data\sft_2000.jsonl")
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
    print("    └─")
    raise AssertionError(f"{label}: 数值不一致 (max|diff|={maxd:.3e})")


def work_dir() -> Path:
    if "dir" not in _STATE:
        env = os.environ.get("MYMM_TEST_DIR")
        path = Path(env) if env else Path(tempfile.mkdtemp())
        path.mkdir(parents=True, exist_ok=True)
        _STATE["dir"] = path
    return _STATE["dir"]


# ============================================================================
def test_distillation_loss():
    torch.manual_seed(0)
    student = torch.randn(4, 10)
    teacher = torch.randn(4, 10)
    for temperature in (1.0, 2.0, 4.0):
        got = distillation_loss(student, teacher, temperature=temperature)
        want = temperature ** 2 * F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        show_diff(f"distillation_loss T={temperature}", got, want, atol=1e-6)

    # 3D 输入应等价于 flatten 后计算
    student3 = torch.randn(2, 3, 10)
    teacher3 = torch.randn(2, 3, 10)
    got3 = distillation_loss(student3, teacher3, temperature=2.0)
    want3 = distillation_loss(student3.reshape(-1, 10), teacher3.reshape(-1, 10), temperature=2.0)
    show_diff("3D 输入（flatten 等价）", got3, want3, atol=1e-6)


def test_distill_loss_masking():
    torch.manual_seed(1)
    B, T, vocab = 2, 5, 10
    teacher_vocab = 12
    student_logits = torch.randn(B, T, vocab)
    teacher_logits = torch.randn(B, T, teacher_vocab)
    labels = torch.randint(0, vocab, (B, T))
    labels[:, :2] = -100
    alpha, temperature = 0.3, 2.0

    out = distill_loss(student_logits, teacher_logits, labels, alpha=alpha, temperature=temperature)

    flat_student = student_logits[:, :-1].reshape(-1, vocab)
    flat_teacher = teacher_logits[:, :-1].reshape(-1, teacher_vocab)[..., :vocab]
    shift_labels = labels[:, 1:]
    mask = (shift_labels != -100).reshape(-1)
    want_ce = F.cross_entropy(flat_student, shift_labels.reshape(-1), ignore_index=-100)
    want_kd = distillation_loss(flat_student[mask], flat_teacher[mask], temperature=temperature)

    show_diff("CE（忽略 -100）", out["ce"], want_ce, atol=1e-6)
    show_diff("KD（只在 assistant 段）", out["kd"], want_kd, atol=1e-6)
    show_diff("loss = alpha*CE + (1-alpha)*KD", out["loss"], alpha * want_ce + (1 - alpha) * want_kd, atol=1e-6)


def test_train_distill_main():
    work = work_dir() / "distill"
    work.mkdir(parents=True, exist_ok=True)
    base = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2))
    torch.save({k: v.half().cpu() for k, v in base.state_dict().items()}, work / "full_sft_128.pth")

    argv = [
        "--data_path", str(SFT_DATA),
        "--tokenizer_dir", str(TOKENIZER_DIR),
        "--save_dir", str(work),
        "--student_weight", "full_sft",
        "--teacher_weight", "full_sft",
        "--hidden_size", "128",
        "--num_hidden_layers", "2",
        "--max_seq_len", "64",
        "--batch_size", "4",
        "--accumulation_steps", "1",
        "--epochs", "1",
        "--max_steps", "2",
        "--lr", "1e-5",
        "--alpha", "0.5",
        "--temperature", "2.0",
        "--log_interval", "1000",
        "--num_workers", "0",
        "--device", DEVICE,
        "--dtype", "bfloat16",
    ]
    t0 = time.time()
    rc = tdist.main(argv)
    if rc != 0:
        raise AssertionError(f"train_distill.main 返回 {rc}")
    out_path = work / "distill_128.pth"
    if not out_path.exists():
        raise AssertionError("没有保存 distill 权重")
    model = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2))
    model.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True), strict=True)
    print(f"      distill_128.pth ({out_path.stat().st_size / 1024:.1f} KB) 生成于 {time.time() - t0:.1f}s")


# ============================================================================
TESTS = [test_distillation_loss, test_distill_loss_masking, test_train_distill_main]

HINTS = {
    "test_distillation_loss": "teacher 用 softmax(t/T)，student 用 log_softmax(s/T)，KL 后乘 T²",
    "test_distill_loss_masking": "先 shift 再按 labels!=-100 的 mask 选 token；teacher 词表按学生截断",
    "test_train_distill_main": "student 可训练、teacher eval+no_grad；保存 distill_{hidden}.pth",
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

    print("-" * 60)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D15 蒸馏部分可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
