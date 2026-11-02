"""D18 验收测试：MoE 层 + MoE 模型 + dense 对照实验

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_moe.py

覆盖：
    1) MoEFFN：输出形状、aux loss 公式（负载均衡）
    2) MoEFFN 参数量 = E × dense FFN + gate
    3) apply_moe：把 dense 模型换成 MoE，参数量/forward/aux/梯度正常
    4) moe_compare：dense vs moe 各跑 2 步，CSV + 曲线生成
"""
from __future__ import annotations

import csv
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

import mymm.moe_compare as mc  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.ffn import FeedForward  # noqa: E402
from mymm.model import CausalLM  # noqa: E402
from mymm.moe import MoECausalLM, MoEFFN, apply_moe  # noqa: E402

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
def test_moe_ffn_aux_and_shape():
    cfg = MiniMindConfig(
        hidden_size=128, num_hidden_layers=2, use_moe=True,
        num_experts=4, num_experts_per_tok=2,
    )
    moe = MoEFFN(cfg, num_experts=4, top_k=2).train()
    x = torch.randn(2, 5, 128)
    y = moe(x)
    if tuple(y.shape) != (2, 5, 128):
        print(f"    输出={tuple(y.shape)}，期望 (2,5,128)")
        raise AssertionError("MoE 输出形状不对")
    if moe.aux_loss.numel() != 1:
        raise AssertionError("aux_loss 应为标量")
    if moe.aux_loss.item() <= 0:
        raise AssertionError("训练模式下 aux_loss 应 > 0")

    scores = F.softmax(moe.gate(x.reshape(-1, 128)), dim=-1)
    _, topk_idx = torch.topk(scores, k=2, dim=-1, sorted=False)
    load = F.one_hot(topk_idx, 4).float().mean(0)
    expected = (load * scores.mean(0)).sum() * 4 * cfg.router_aux_loss_coef
    show_diff("aux loss 公式", moe.aux_loss, expected, atol=1e-7)

    moe.eval()
    moe(x)
    if moe.aux_loss.item() != 0.0:
        raise AssertionError("eval 模式下 aux_loss 应为 0")


def test_moe_ffn_params():
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    dense = FeedForward(cfg)
    moe = MoEFFN(cfg, num_experts=4, top_k=2)
    dense_params = sum(p.numel() for p in dense.parameters())
    got = sum(p.numel() for p in moe.parameters())
    want = 4 * dense_params + 128 * 4  # E 个 expert + gate
    if got != want:
        print(f"    MoEFFN 参数={got}，期望 {want}")
        raise AssertionError("MoEFFN 参数量不对")
    print(f"      dense FFN={dense_params}, MoEFFN={got}")


def test_apply_moe_model():
    torch.manual_seed(0)
    cfg = MiniMindConfig(
        hidden_size=128, num_hidden_layers=2, use_moe=True,
        num_experts=4, num_experts_per_tok=2,
    )
    base = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2))
    dense_params = sum(p.numel() for p in base.parameters())

    model = MoECausalLM(cfg)
    apply_moe(model, num_experts=4, top_k=2)
    ffn_params = sum(p.numel() for p in FeedForward(cfg).parameters())
    delta = 2 * ((4 - 1) * ffn_params + 128 * 4)  # 2 层
    got = sum(p.numel() for p in model.parameters())
    if got != dense_params + delta:
        print(f"    MoE 模型参数={got}，期望 {dense_params + delta}")
        raise AssertionError("apply_moe 后参数量不对")

    ids = torch.randint(0, 6400, (1, 8))
    out = model(ids, labels=ids)
    if out.loss is None or not torch.isfinite(out.loss):
        raise AssertionError("MoE loss 应为有限值")
    if out.aux_loss is None or out.aux_loss.item() <= 0:
        raise AssertionError("MoE aux_loss 应 > 0")
    if tuple(out.logits.shape) != (1, 8, 6400):
        raise AssertionError("MoE logits 形状不对")

    total = out.loss + out.aux_loss
    total.backward()
    gate_grads = [p.grad for p in model.model.layers[0].mlp.gate.parameters()]
    if any(g is None for g in gate_grads):
        raise AssertionError("aux loss 应能给 gate 回传梯度")
    expert_grads = [p.grad for p in model.model.layers[0].mlp.experts[0].parameters()]
    print(f"      params={got / 1e6:.2f}M (+{delta / 1e6:.2f}M), loss={out.loss.item():.3f}, aux={out.aux_loss.item():.5f}")
    if not any(g is not None for g in expert_grads):
        raise AssertionError("expert 应该有梯度（若随机路由没选到 expert0，换 seed）")


def test_moe_compare_main():
    work = work_dir() / "moe_cmp"
    work.mkdir(parents=True, exist_ok=True)
    argv = [
        "--run", "dense", "moe",
        "--steps", "2",
        "--batch_size", "4",
        "--hidden_size", "128",
        "--num_hidden_layers", "2",
        "--max_seq_len", "64",
        "--aux_coef", "0.01",
        "--out_dir", str(work),
        "--device", DEVICE,
        "--dtype", "bfloat16",
        "--plot", "1",
    ]
    t0 = time.time()
    rc = mc.main(argv)
    if rc != 0:
        raise AssertionError(f"moe_compare.main 返回 {rc}")
    for name in ("dense", "moe"):
        csv_path = work / "exp" / f"{name}.csv"
        if not csv_path.exists():
            raise AssertionError(f"缺少 {name}.csv")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 2:
            print(f"    {name}.csv 行数={len(rows)}，期望 2")
            raise AssertionError("CSV 行数不对")
        if not all(torch.isfinite(torch.tensor(float(r["loss"]))) for r in rows):
            raise AssertionError("loss 里有 inf/nan")
    png = work / "curves.png"
    if not png.exists() or png.stat().st_size <= 0:
        raise AssertionError("曲线 PNG 未生成")
    print(f"      CSV×2 + curves.png ({png.stat().st_size / 1024:.1f}KB) 用时 {time.time() - t0:.1f}s")


# ============================================================================
TESTS = [test_moe_ffn_aux_and_shape, test_moe_ffn_params, test_apply_moe_model, test_moe_compare_main]

HINTS = {
    "test_moe_ffn_aux_and_shape": "scores=softmax(gate(x))；aux=E*coef*Σ(load*mean_score)；eval 时为 0",
    "test_moe_ffn_params": "E 个 FeedForward + gate(hidden→E)",
    "test_apply_moe_model": "apply_moe 把每层 mlp 换成 MoE；forward 汇总各层 aux_loss",
    "test_moe_compare_main": "moe_compare 用同一数据/步数训练 dense 和 moe，写 CSV+曲线",
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
    print(f"全部 {len(selected)} 项通过，D18 可以勾选 S3-15（选做）/ S10-06。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
