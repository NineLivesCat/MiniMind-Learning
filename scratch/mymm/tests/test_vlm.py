"""D16 验收测试：VLM（视觉编码器 / projector / 注入 / forward / 冻结策略 / 数据集 / 训练）

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_vlm.py

覆盖：
    1) SigLIP2 加载：冻结、eval、256x256 → 64 个 768 维 patch
    2) projector：768 → hidden、参数量
    3) count_vision_proj：连续 image_pad 用对应图片特征替换（多段/多图）
    4) forward：注入后 logits/loss 正常，梯度只到 vision_proj 不到 vision_encoder
    5) freeze_vlm 三种策略的可训练参数集合
    6) VLMDataset：<image> → 64 个 pad、labels 只含 assistant、pixel_values 形状
    7) train_vlm.main 两阶段能跑通并保存（不含 vision_encoder）
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

REPO = Path(r"D:\minimind\minimind")
SCRATCH = Path(__file__).resolve().parents[2]  # learning\scratch
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

import mymm.train_vlm as tvlm  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.model import CausalLMOutput  # noqa: E402
from mymm.vlm import (  # noqa: E402
    MMVisionProjector,
    MiniMindVLM,
    freeze_vlm,
    run_transformer_layers,
)
from mymm.vlm_data import VLMDataset  # noqa: E402

TOKENIZER_DIR = REPO / "model"
VISION_PATH = Path(r"D:\minimind\minimind-v\model\siglip2-base-p32-256-ve")
DATA_1000 = Path(r"D:\minimind\learning\data\vlm_1000.parquet")
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


def get_tokenizer():
    if "tok" not in _STATE:
        from transformers import AutoTokenizer

        _STATE["tok"] = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return _STATE["tok"]


def get_vlm(hidden_size=128, layers=2):
    key = f"vlm_{hidden_size}_{layers}"
    if key not in _STATE:
        cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=layers)
        _STATE[key] = MiniMindVLM(cfg, vision_model_path=str(VISION_PATH)).eval()
    return _STATE[key]


def get_first_image():
    if "img" not in _STATE:
        from PIL import Image
        import io

        table = pq.read_table(DATA_1000)
        image_bytes = table.slice(0, 1).to_pylist()[0]["image_bytes"]
        _STATE["img"] = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _STATE["img"]


# ============================================================================
def test_vision_encoder():
    model = get_vlm()
    encoder, processor = model.vision_encoder, model.processor
    if encoder is None or processor is None:
        raise AssertionError("SigLIP2 没有加载成功")
    if encoder.training:
        raise AssertionError("vision encoder 应处于 eval")
    if any(p.requires_grad for p in encoder.parameters()):
        raise AssertionError("vision encoder 参数应全部冻结")

    inputs = MiniMindVLM.image2tensor(get_first_image(), processor)
    pixel_values = inputs["pixel_values"]
    if tuple(pixel_values.shape) != (1, 3, 256, 256):
        print(f"    pixel_values={tuple(pixel_values.shape)}，期望 (1,3,256,256)")
        raise AssertionError("图像预处理尺寸不对")
    feats = MiniMindVLM.get_image_embeddings(pixel_values, encoder)
    if tuple(feats.shape) != (1, 64, 768):
        print(f"    vision features={tuple(feats.shape)}，期望 (1,64,768)")
        raise AssertionError("patch 数量/维度不对")
    print(f"      vision params={sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")


def test_projector():
    proj = MMVisionProjector(768, 128)
    x = torch.randn(2, 64, 768)
    y = proj(x)
    if tuple(y.shape) != (2, 64, 128):
        print(f"    projector 输出={tuple(y.shape)}，期望 (2,64,128)")
        raise AssertionError("projector 形状不对")
    want = 768 * 2 + (768 * 128 + 128) + (128 * 128 + 128)
    got = sum(p.numel() for p in proj.parameters())
    if got != want:
        print(f"    projector 参数={got}，期望 {want}")
        raise AssertionError("projector 参数量不对")


def test_count_vision_proj():
    vlm = get_vlm()
    B, T, H = 1, 10, 128
    tokens = torch.full((B, T), 3, dtype=torch.long)  # 3 是普通 token
    tokens[0, 2:4] = 12  # 第一段：2 个 image_pad
    tokens[0, 6:8] = 12  # 第二段：2 个 image_pad
    h = torch.randn(B, T, H)
    vision = torch.randn(B, 2, 64, H)  # 2 张图
    got = vlm.count_vision_proj(tokens, h.clone(), vision, seqlen=T)
    want = h.clone()
    want[0, 2:4] = vision[0, 0, :2]
    want[0, 6:8] = vision[0, 1, :2]
    show_diff("count_vision_proj 替换两段", got, want, atol=1e-6)

    # 3 维输入视作单图：只有第一段 image_pad 会被替换，第二段保持原 embedding
    got1 = vlm.count_vision_proj(tokens, h.clone(), vision[:, 0], seqlen=T)
    want1 = h.clone()
    want1[0, 2:4] = vision[0, 0, :2]
    show_diff("单图 3 维输入", got1, want1, atol=1e-6)


def test_forward_injection():
    vlm = get_vlm().to(DEVICE)
    tok = get_tokenizer()
    image_inputs = MiniMindVLM.image2tensor(get_first_image(), vlm.processor)
    pixel_values = image_inputs["pixel_values"].to(DEVICE)

    ids = [tok.bos_token_id] + [12] * 64 + tok("描述这张图。", add_special_tokens=False).input_ids
    input_ids = torch.tensor([ids], device=DEVICE)
    labels = input_ids.clone()
    out = vlm(input_ids, labels=labels, pixel_values=pixel_values)
    if out.loss is None or not torch.isfinite(out.loss):
        raise AssertionError("带图片的 loss 应为有限值")
    if tuple(out.logits.shape) != (1, len(ids), 6400):
        print(f"    logits={tuple(out.logits.shape)}")
        raise AssertionError("logits 形状不对")

    out_no_img = vlm(input_ids, labels=labels, pixel_values=None)
    if torch.allclose(out.logits, out_no_img.logits):
        raise AssertionError("注入视觉特征后 logits 应与纯文本不同")

    vlm.zero_grad(set_to_none=True)
    out.loss.backward()
    proj_grad = sum(p.grad.abs().sum().item() for p in vlm.vision_proj.parameters() if p.grad is not None)
    if proj_grad <= 0:
        raise AssertionError("loss 没有回传到 vision_proj")
    if any(p.grad is not None for p in vlm.vision_encoder.parameters()):
        raise AssertionError("vision_encoder 不该有梯度")
    vlm.zero_grad(set_to_none=True)
    print(f"      loss={out.loss.item():.3f}, proj grad sum={proj_grad:.3e}")


def test_freeze_vlm():
    vlm = get_vlm()
    last = vlm.config.num_hidden_layers - 1

    freeze_vlm(vlm, 2)
    names2 = {n for n, p in vlm.named_parameters() if p.requires_grad}
    if not names2 or any("vision_proj" not in n for n in names2):
        raise AssertionError("freeze_llm=2 时应只有 vision_proj 可训练")

    freeze_vlm(vlm, 1)
    names1 = {n for n, p in vlm.named_parameters() if p.requires_grad}
    want1 = {"vision_proj"} | {n for n in names1 if "vision_proj" in n}
    llm1 = {n for n in names1 if "vision_proj" not in n}
    if not llm1:
        raise AssertionError("freeze_llm=1 时应放开首尾层")
    allowed = ("layers.0.", f"layers.{last}.")
    if any(not n.startswith("model.") or not any(a in n for a in allowed) for n in llm1):
        print(f"    意外可训练参数：{sorted(llm1)[:5]}")
        raise AssertionError("freeze_llm=1 只应放开 layers.0 和最后一层")

    freeze_vlm(vlm, 0)
    names0 = {n for n, p in vlm.named_parameters() if p.requires_grad}
    if any("vision_encoder" in n for n in names0):
        raise AssertionError("freeze_llm=0 也不该训练 vision_encoder")
    if not any("lm_head" in n or "embed_tokens" in n for n in names0):
        raise AssertionError("freeze_llm=0 时 LLM 应全部可训练")
    print(f"      freeze2={len(names2)} freeze1={len(names1)} freeze0={len(names0)}")


def test_vlm_dataset():
    tok = get_tokenizer()
    vlm = get_vlm()
    ds = VLMDataset(str(DATA_1000), tok, vlm.processor, max_length=256)
    if len(ds) != 1000:
        print(f"    数据集长度={len(ds)}，期望 1000")
        raise AssertionError("VLMDataset 长度不对")
    item = ds[0]
    for key in ("input_ids", "labels", "pixel_values"):
        if key not in item:
            raise AssertionError(f"缺少字段 {key}")
    ids = item["input_ids"]
    labels = item["labels"]
    pv = item["pixel_values"]
    if tuple(ids.shape) != (256,) or tuple(labels.shape) != (256,):
        raise AssertionError("input_ids/labels 形状应为 (max_length,)")
    if ids.dtype != torch.long or labels.dtype != torch.long:
        raise AssertionError("input_ids/labels 应为 long")
    if tuple(pv.shape) != (1, 3, 256, 256):
        print(f"    pixel_values={tuple(pv.shape)}，期望 (1,3,256,256)")
        raise AssertionError("pixel_values 形状不对")
    if int((ids == 12).sum()) != 64:
        print(f"    image_pad 数={int((ids == 12).sum())}，期望 64")
        raise AssertionError("<image> 应替换成恰好 64 个 image_pad")
    if int((labels != -100).sum()) <= 0:
        raise AssertionError("labels 里没有 assistant 段")
    print(f"      ids={tuple(ids.shape)} pv={tuple(pv.shape)} label_tokens={int((labels != -100).sum())}")


def test_train_vlm_main():
    work = work_dir() / "vlm_train"
    work.mkdir(parents=True, exist_ok=True)
    common = [
        "--tokenizer_dir", str(TOKENIZER_DIR),
        "--vision_model_path", str(VISION_PATH),
        "--save_dir", str(work),
        "--hidden_size", "128",
        "--num_hidden_layers", "2",
        "--max_seq_len", "128",
        "--batch_size", "1",
        "--accumulation_steps", "1",
        "--epochs", "1",
        "--max_steps", "1",
        "--lr", "1e-4",
        "--log_interval", "1000",
        "--num_workers", "0",
        "--device", DEVICE,
        "--dtype", "bfloat16",
        "--data_path", str(DATA_1000),
    ]
    t0 = time.time()
    rc = tvlm.main(["--stage", "pretrain", "--init_weight", "none", "--freeze_llm", "2"] + common)
    if rc != 0:
        raise AssertionError(f"pretrain 阶段返回 {rc}")
    pretrain_path = work / "pretrain_vlm_128.pth"
    if not pretrain_path.exists():
        raise AssertionError("没有保存 pretrain_vlm_128.pth")
    state = torch.load(pretrain_path, map_location="cpu", weights_only=True)
    if any(k.startswith("vision_encoder.") for k in state):
        raise AssertionError("保存的 ckpt 不应包含冻结的 vision_encoder")
    if not any("vision_proj" in k for k in state):
        raise AssertionError("ckpt 里缺少 vision_proj")

    rc = tvlm.main(["--stage", "sft", "--freeze_llm", "1"] + common)
    if rc != 0:
        raise AssertionError(f"sft 阶段返回 {rc}")
    sft_path = work / "sft_vlm_128.pth"
    if not sft_path.exists():
        raise AssertionError("没有保存 sft_vlm_128.pth")
    print(f"      pretrain={pretrain_path.stat().st_size / 1024 ** 2:.1f}MB "
        f"sft={sft_path.stat().st_size / 1024 ** 2:.1f}MB ({time.time() - t0:.1f}s)")


# ============================================================================
TESTS = [
    test_vision_encoder,
    test_projector,
    test_count_vision_proj,
    test_forward_injection,
    test_freeze_vlm,
    test_vlm_dataset,
    test_train_vlm_main,
]

HINTS = {
    "test_vision_encoder": "SiglipVisionModel.from_pretrained + 冻结；256x256 → 64 个 patch（768 维）",
    "test_projector": "LayerNorm(768) → Linear(768,hidden) → GELU → Linear(hidden,hidden)",
    "test_count_vision_proj": "扫描连续 image_pad 段，第 k 张图用 vision_tensors[b][k][:n] 替换",
    "test_forward_injection": "先 embed → 注入视觉特征 → 跑各层 → lm_head → shift loss",
    "test_freeze_vlm": "freeze=2 只 proj；=1 再加 layers.0/最后一层；=0 除 vision_encoder 全开",
    "test_vlm_dataset": "<image> 替换成 64 个 pad；图像过 SiglipImageProcessor",
    "test_train_vlm_main": "两阶段训练入口：pretrain 用 freeze_llm=2，sft 用 freeze_llm=1 并加载 pretrain_vlm",
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
    print(f"全部 {len(selected)} 项通过，D16 VLM 部分可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
