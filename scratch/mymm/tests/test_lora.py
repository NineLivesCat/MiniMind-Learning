"""D13 验收测试：LoRA 模块 + 挂载 + 冻结 + 保存/加载/合并 + 训练接线

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_lora.py

覆盖：
    1) LoRA：A 高斯初始化、B 全 0、forward = B(A(x))
    2) apply_lora：只挂 square Linear（128 维模型 4 个），挂完输出不变
    3) freeze：只有 lora 参数可训练（rank=16 时 16,384 个）
    4) save/load/merge：加载后输出一致；W_new = W_old + B@A；合并后无 lora
    5) lora_merge 工具：合并文件能被普通 CausalLM 直接加载
    6) train_lora.main：冻结 base、训练、保存 lora 权重
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

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

import mymm.train_lora as tl  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.lora import (  # noqa: E402
    LoRA,
    apply_lora,
    count_trainable,
    freeze_base_model,
    load_lora,
    merge_lora,
    save_lora,
)
from mymm.lora_merge import merge_files  # noqa: E402
from mymm.model import CausalLM  # noqa: E402

TOKENIZER_DIR = REPO / "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STATE: dict = {}


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


def make_model(hidden_size=128, layers=2) -> CausalLM:
    cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=layers)
    return CausalLM(cfg).eval()


def square_linears(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m.in_features == m.out_features]


# ============================================================================
def test_lora_module():
    torch.manual_seed(0)
    lora = LoRA(16, 16, rank=8)
    if tuple(lora.A.weight.shape) != (8, 16) or tuple(lora.B.weight.shape) != (16, 8):
        print(f"    A={tuple(lora.A.weight.shape)} B={tuple(lora.B.weight.shape)}")
        raise AssertionError("A/B 形状应为 (rank, in) / (out, rank)")
    std = lora.A.weight.std().item()
    if not (0.01 < std < 0.03):
        print(f"    A 的 std={std:.4f}，期望 ~0.02")
        raise AssertionError("A 应做 N(0, 0.02) 高斯初始化")
    show_diff("B 应全 0", lora.B.weight, torch.zeros_like(lora.B.weight), atol=0)

    x = torch.randn(2, 5, 16)
    y = lora(x)
    show_diff("forward = B(A(x))", y, lora.B(lora.A(x)), atol=1e-6)


def test_apply_lora_identity():
    torch.manual_seed(1)
    model = make_model()
    ids = torch.randint(0, 6400, (1, 6))
    with torch.no_grad():
        before = model(ids).logits.clone()

    n = apply_lora(model, rank=16)
    expected = len(square_linears(model))
    if n != expected:
        print(f"    挂载数量={n}，期望 {expected}")
        raise AssertionError("apply_lora 挂载数量不对")
    if len([m for m in model.modules() if hasattr(m, "lora")]) != expected:
        raise AssertionError("带有 lora 子模块的 Linear 数量不对")

    with torch.no_grad():
        after = model(ids).logits
    show_diff("B=0 时挂 LoRA 后输出应不变", after, before, atol=1e-6)
    print(f"      attached={n} modules")


def test_freeze_and_count():
    model = make_model()
    apply_lora(model, rank=16)
    freeze_base_model(model)
    got = count_trainable(model)
    expected = 4 * (16 * 128 + 128 * 16)  # q/o × 2 层 × (A+B)
    if got != expected:
        print(f"    可训练参数={got}，期望 {expected}")
        raise AssertionError("LoRA 可训练参数量不对")
    for name, param in model.named_parameters():
        if param.requires_grad and ".lora." not in name:
            print(f"    这个参数不该可训练：{name}")
            raise AssertionError("freeze_base_model 没冻住 base 参数")
    print(f"      trainable={got} ({got / 1e6:.3f}M)")


def test_save_load_merge():
    torch.manual_seed(2)
    work = work_dir() / "lora_merge"
    work.mkdir(parents=True, exist_ok=True)
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    ids = torch.randint(0, 6400, (1, 6))

    model = CausalLM(cfg).eval()
    apply_lora(model, rank=4)
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            torch.nn.init.normal_(module.lora.A.weight, std=0.02)
            torch.nn.init.normal_(module.lora.B.weight, std=0.02)
    with torch.no_grad():
        out_before = model(ids).logits.clone()

    lora_path = work / "lora_rank4.pth"
    save_lora(model, lora_path)
    if not lora_path.exists() or lora_path.stat().st_size > 1024 * 1024:
        raise AssertionError("LoRA 权重文件异常（应该很小）")

    fresh = CausalLM(cfg).eval()
    fresh.load_state_dict(model.state_dict(), strict=False)  # base 权重一致
    apply_lora(fresh, rank=4)
    loaded = load_lora(fresh, lora_path)
    if loaded != 4:
        print(f"    load_lora 加载模块数={loaded}，期望 4")
        raise AssertionError("load_lora 加载数量不对")
    with torch.no_grad():
        out_loaded = fresh(ids).logits
    show_diff("加载 LoRA 后输出一致", out_loaded, out_before, atol=1e-3)

    module = fresh.model.layers[0].self_attn.q_proj
    w_before = module.weight.detach().clone()
    delta = module.lora.B.weight.detach() @ module.lora.A.weight.detach()
    merge_lora(fresh)
    show_diff("merge 后 W = W + B@A", module.weight, w_before + delta, atol=1e-6)
    if hasattr(module, "lora"):
        raise AssertionError("merge 后不应再有 lora 子模块")
    with torch.no_grad():
        out_merged = fresh(ids).logits
    show_diff("merge 前后输出一致", out_merged, out_loaded, atol=1e-5)


def test_merge_files():
    torch.manual_seed(3)
    work = work_dir() / "merge_tool"
    work.mkdir(parents=True, exist_ok=True)
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    base = CausalLM(cfg).eval()
    base_path = work / "full_sft_128.pth"
    torch.save({k: v.half().cpu() for k, v in base.state_dict().items()}, base_path)

    lora_model = CausalLM(cfg).eval()
    lora_model.load_state_dict(base.state_dict(), strict=True)
    apply_lora(lora_model, rank=4)
    for name, module in lora_model.named_modules():
        if hasattr(module, "lora"):
            torch.nn.init.normal_(module.lora.B.weight, std=0.02)
    lora_path = work / "lora_rank4.pth"
    save_lora(lora_model, lora_path)

    ids = torch.randint(0, 6400, (1, 5))
    with torch.no_grad():
        expect = lora_model(ids).logits.clone()

    merged_path = work / "merged.pth"
    merge_files(str(base_path), str(lora_path), str(merged_path), 128, 2)
    plain = CausalLM(cfg).eval()
    plain.load_state_dict(torch.load(merged_path, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        got = plain(ids).logits
    show_diff("merge_files 产物输出一致", got, expect, atol=1e-3)
    if plain.state_dict().keys() != base.state_dict().keys():
        raise AssertionError("合并后 state_dict 应和普通模型一致（无 lora key）")


def test_train_lora_main():
    work = work_dir() / "train_lora"
    work.mkdir(parents=True, exist_ok=True)
    base = make_model()
    torch.save({k: v.half().cpu() for k, v in base.state_dict().items()}, work / "full_sft_128.pth")

    data_path = work / "tiny_lora.jsonl"
    with open(data_path, "w", encoding="utf-8") as fh:
        for i in range(8):
            conv = [
                {"role": "user", "content": f"医学问题 {i}：感冒了怎么办？"},
                {"role": "assistant", "content": f"多喝水、休息，必要时就医（{i}）。"},
            ]
            fh.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")

    captured = {}
    real_train_epoch = tl.train_epoch

    def fake_train_epoch(epoch, loader, model, optimizer, scaler, args, start_step=0, iters=None):
        captured["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        captured["trainable_names"] = [n for n, p in model.named_parameters() if p.requires_grad]
        captured["batches"] = len(loader)
        return []

    tl.train_epoch = fake_train_epoch
    try:
        argv = [
            "--data_path", str(data_path),
            "--tokenizer_dir", str(TOKENIZER_DIR),
            "--save_dir", str(work),
            "--base_weight", "full_sft",
            "--save_name", "lora_tiny",
            "--rank", "4",
            "--hidden_size", "128",
            "--num_hidden_layers", "2",
            "--max_seq_len", "64",
            "--batch_size", "4",
            "--accumulation_steps", "1",
            "--epochs", "1",
            "--lr", "1e-3",
            "--log_interval", "1000",
            "--num_workers", "0",
            "--device", DEVICE,
            "--dtype", "bfloat16",
        ]
        rc = tl.main(argv)
    finally:
        tl.train_epoch = real_train_epoch

    if rc != 0:
        raise AssertionError(f"train_lora.main 返回 {rc}")
    expected = 4 * (4 * 128 + 128 * 4)
    if captured.get("trainable") != expected:
        print(f"    可训练参数={captured.get('trainable')}，期望 {expected}")
        raise AssertionError("训练时只应训练 LoRA 参数")
    if any(".lora." not in n for n in captured["trainable_names"]):
        raise AssertionError("存在非 LoRA 的可训练参数（base 没冻住）")
    if captured.get("batches") != 2:
        print(f"    batches={captured.get('batches')}，期望 2")
        raise AssertionError("8 条数据 / batch 4 应为 2 步")

    lora_path = work / "lora_tiny_128.pth"
    if not lora_path.exists():
        raise AssertionError("没有保存 LoRA 权重")
    state = torch.load(lora_path, map_location="cpu", weights_only=True)
    if not state or not all(k.endswith((".lora.A.weight", ".lora.B.weight")) for k in state):
        raise AssertionError("保存的 LoRA state_dict key 不对")

    fresh = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2)).eval()
    apply_lora(fresh, rank=4)
    load_lora(fresh, lora_path)
    with torch.no_grad():
        fresh(torch.randint(0, 6400, (1, 4)))
    print(f"      saved {lora_path.name} ({lora_path.stat().st_size / 1024:.1f} KB), trainable={captured['trainable']}")


def test_chat_lora():
    import mymm.chat as chat

    work = work_dir() / "chat_lora"
    work.mkdir(parents=True, exist_ok=True)
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    torch.manual_seed(4)
    base = CausalLM(cfg).eval()
    torch.save({k: v.half().cpu() for k, v in base.state_dict().items()}, work / "full_sft_128.pth")
    ids = torch.randint(0, 6400, (1, 5))

    lora_model = CausalLM(cfg).eval()
    lora_model.load_state_dict(base.state_dict(), strict=True)
    apply_lora(lora_model, rank=4)
    for name, module in lora_model.named_modules():
        if hasattr(module, "lora"):
            torch.nn.init.normal_(module.lora.B.weight, std=0.02)
    lora_path = work / "lora.pth"
    save_lora(lora_model, lora_path)
    with torch.no_grad():
        want = lora_model(ids).logits.clone()
        base_out = base(ids).logits.clone()

    args = SimpleNamespace(
        tokenizer_dir=str(TOKENIZER_DIR),
        save_dir=str(work),
        weight="full_sft",
        weight_path="",
        hidden_size=128,
        num_hidden_layers=2,
        device="cpu",
        lora=str(lora_path),
        lora_rank=4,
    )
    model, _ = chat.build_model(args)
    with torch.no_grad():
        got = model(ids).logits
    show_diff("chat 加载 LoRA 后输出", got, want, atol=2e-3)
    if torch.allclose(got, base_out, atol=1e-3):
        raise AssertionError("LoRA 没生效（输出和 base 一样）")


# ============================================================================
TESTS = [
    test_lora_module,
    test_apply_lora_identity,
    test_freeze_and_count,
    test_save_load_merge,
    test_merge_files,
    test_train_lora_main,
    test_chat_lora,
]

HINTS = {
    "test_lora_module": "A = nn.Linear(in, rank)，B = nn.Linear(rank, out)；A ~ N(0,0.02)，B = 0",
    "test_apply_lora_identity": "只挂 in==out 的 Linear；B=0 所以输出必须和原来一致",
    "test_freeze_and_count": "freeze 后只有名字含 .lora. 的参数 requires_grad=True",
    "test_save_load_merge": "merge：W += B@A，然后 del module.forward / del module.lora",
    "test_merge_files": "merge_files 用 base + lora 生成普通 checkpoint",
    "test_train_lora_main": "main 里 apply_lora → freeze → 只优化 LoRA 参数 → save_lora",
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
    print(f"全部 {len(selected)} 项通过，D13 可以勾选 S7-01 / S7-02 / S7-03 / S7-04 / S7-05 / S7-06。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
