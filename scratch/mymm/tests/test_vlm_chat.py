"""D17 验收测试：VLM 推理/对话 + 评测报告

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_vlm_chat.py

覆盖：
    1) load_vlm 能加载不含 vision_encoder 的 VLM ckpt
    2) vlm_generate：带图生成、返回文本/token 数/耗时；prompt 自动补 <image>
    3) 评测脚本：生成 report.md / report.csv
"""
from __future__ import annotations

import csv
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

import mymm.vlm_eval as ve  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.vlm import MiniMindVLM  # noqa: E402
from mymm.vlm_chat import EVAL_IMAGES, load_vlm, vlm_generate  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STATE: dict = {}


def work_dir() -> Path:
    if "dir" not in _STATE:
        env = os.environ.get("MYMM_TEST_DIR")
        path = Path(env) if env else Path(tempfile.mkdtemp())
        path.mkdir(parents=True, exist_ok=True)
        _STATE["dir"] = path
    return _STATE["dir"]


def make_ckpt() -> Path:
    if "ckpt" not in _STATE:
        work = work_dir()
        model = MiniMindVLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2))
        state = {k: v.half().cpu() for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")}
        path = work / "sft_vlm_128.pth"
        torch.save(state, path)
        _STATE["ckpt"] = path
    return _STATE["ckpt"]


def get_images() -> list[Path]:
    return [p for p in sorted(Path(EVAL_IMAGES).iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]


# ============================================================================
def test_load_vlm():
    ckpt = make_ckpt()
    model, tokenizer, path = load_vlm(
        save_dir=str(ckpt.parent), weight="sft_vlm", hidden_size=128, num_hidden_layers=2,
        device=DEVICE,
    )
    if path != ckpt:
        raise AssertionError("load_vlm 没加载到预期文件")
    if model.training:
        raise AssertionError("推理模型应处于 eval")
    if model.vision_encoder is None:
        raise AssertionError("vision encoder 没加载")
    if DEVICE == "cuda" and next(model.parameters()).dtype != torch.float16:
        raise AssertionError("cuda 下 VLM 应使用 fp16 推理")
    print(f"      params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, ckpt={path.name}")


def test_vlm_generate():
    model, tokenizer, _ = load_vlm(
        save_dir=str(make_ckpt().parent), weight="sft_vlm", hidden_size=128,
        num_hidden_layers=2, device=DEVICE,
    )
    image_path = get_images()[0]
    from PIL import Image

    image = Image.open(image_path)
    t0 = time.time()
    reply, n, dt = vlm_generate(
        model, tokenizer, image, "请描述这张图。",
        max_new_tokens=4, temperature=0.7, top_p=0.85, top_k=50, device=DEVICE,
    )
    if not isinstance(reply, str):
        raise AssertionError("回答应为 str")
    if n <= 0 or n > 4:
        print(f"    生成 token 数={n}")
        raise AssertionError("生成 token 数异常")
    if dt <= 0:
        raise AssertionError("耗时应为正")
    print(f"      生成 {n} tok / {dt:.2f}s | {reply[:40]!r}")

    # prompt 不含 <image> 时应自动补上，而不是报错
    reply2, n2, _ = vlm_generate(
        model, tokenizer, image, "图中有什么？",
        max_new_tokens=2, device=DEVICE,
    )
    if not isinstance(reply2, str) or n2 <= 0:
        raise AssertionError("prompt 自动补 <image> 后应能生成")


def test_eval_report():
    work = work_dir() / "eval"
    work.mkdir(parents=True, exist_ok=True)
    argv = [
        "--save_dir", str(make_ckpt().parent),
        "--weight", "sft_vlm",
        "--hidden_size", "128",
        "--num_hidden_layers", "2",
        "--images_dir", str(EVAL_IMAGES),
        "--max_images", "1",
        "--questions", "描述这张图。",
        "--max_new_tokens", "4",
        "--out_dir", str(work),
        "--device", DEVICE,
    ]
    rc = ve.main(argv)
    if rc != 0:
        raise AssertionError(f"vlm_eval.main 返回 {rc}")
    report = work / "report.md"
    csv_path = work / "report.csv"
    if not report.exists() or not csv_path.exists():
        raise AssertionError("report.md / report.csv 未生成")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        print(f"    CSV 行数={len(rows)}，期望 1")
        raise AssertionError("评测行数不对")
    if get_images()[0].name not in report.read_text(encoding="utf-8"):
        raise AssertionError("报告里没有图片名")
    print(f"      {report.name} + {csv_path.name}, {len(rows)} 行")


# ============================================================================
TESTS = [test_load_vlm, test_vlm_generate, test_eval_report]

HINTS = {
    "test_load_vlm": "VLM ckpt 不含 vision_encoder；strict=False 加载后检查 vision_proj 存在",
    "test_vlm_generate": "prompt 里 <image> 要展开成 64 个 image_pad；pixel_values 传进 sampling.generate",
    "test_eval_report": "vlm_eval 对固定图片×问题生成，输出 md/csv",
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
            print(f"{kind} {fn.__name__:22s} {type(exc).__name__ if kind == 'ERROR' else ''}{exc}")
            print(f"      提示：{HINTS.get(fn.__name__, '')}")
        else:
            print(f"PASS  {fn.__name__:22s} ({time.time() - t0:.1f}s)")

    print("-" * 60)
    if failed:
        print(f"{len(selected) - len(failed)}/{len(selected)} 通过，未过：{', '.join(failed)}")
        return 1
    print(f"全部 {len(selected)} 项通过，D17 VLM 推理/评测可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
