"""D12 验收测试：滑动窗口 PPL + 生成评测 + 吞吐/显存 + 报告

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_eval.py

覆盖：
    1) 均匀分布模型 PPL ≈ vocab_size（验证 exp(平均 NLL) 公式）
    2) 完美预测模型 PPL ≈ 1，且滑动窗口只统计新覆盖 token
    3) generate_texts 能出固定 prompt 的样本
    4) benchmark 返回 tokens/s（和显存峰值）
    5) write_report 生成 report.md / report.csv / generations_*.md
"""
from __future__ import annotations

import csv
import math
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

from mymm.config import MiniMindConfig  # noqa: E402
from mymm.eval import (  # noqa: E402
    PROMPTS,
    benchmark,
    build_ids,
    compute_ppl,
    generate_texts,
    write_report,
)
from mymm.model import CausalLM  # noqa: E402

TOKENIZER_DIR = REPO / "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 6400
_STATE: dict = {}


class UniformModel:
    """所有 token 等概率 → PPL 应恰好等于 vocab_size。"""

    def eval(self):
        return self

    def __call__(self, input_ids):
        length = input_ids.shape[1]
        logits = torch.zeros(1, length, VOCAB, device=input_ids.device)
        return SimpleNamespace(logits=logits)


class PerfectModel:
    """下一 token 永远给正确项极高 logit → PPL ≈ 1。"""

    def __init__(self, ids):
        self.ids = ids

    def eval(self):
        return self

    def __call__(self, input_ids):
        length = input_ids.shape[1]
        row = input_ids[0].tolist()
        start = 0
        for i in range(len(self.ids) - length + 1):
            if self.ids[i : i + length] == row:
                start = i
                break
        logits = torch.full((1, length, VOCAB), -30.0, device=input_ids.device)
        for j in range(length - 1):
            logits[0, j, self.ids[start + j + 1]] = 30.0
        return SimpleNamespace(logits=logits)


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
def test_ppl_uniform():
    tok = get_tokenizer()
    texts = ["你好，这是一个用来测试困惑度的小文本。", "第二段文本，确保窗口逻辑被走到。"]
    result = compute_ppl(
        UniformModel(), tok, texts,
        max_length=64, stride=32, device="cpu", max_tokens=256,
    )
    ppl = result["ppl"]
    if abs(ppl - VOCAB) > 1.0:
        print(f"    均匀分布模型 PPL={ppl:.2f}，期望 {VOCAB}")
        raise AssertionError("PPL 公式不对（均匀分布应等于 vocab_size）")
    if result["tokens"] <= 0 or result["windows"] <= 0:
        raise AssertionError("统计的 token/window 数应为正")
    print(f"      uniform PPL={ppl:.1f} tokens={result['tokens']} windows={result['windows']}")


def test_ppl_perfect():
    tok = get_tokenizer()
    texts = ["短文本测试滑动窗口困惑度。"]
    ids = build_ids(tok, texts, max_tokens=128)
    result = compute_ppl(
        PerfectModel(ids), tok, texts,
        max_length=128, stride=64, device="cpu", max_tokens=128,
    )
    if result["ppl"] > 1.01:
        print(f"    完美模型 PPL={result['ppl']:.4f}，期望 ≈1")
        raise AssertionError("完美预测模型的 PPL 应接近 1")
    if result["tokens"] != len(ids) - 1:
        print(f"    统计 token 数={result['tokens']}，期望 {len(ids) - 1}（单窗口全覆盖）")
        raise AssertionError("单窗口时应统计全部 len-1 个 token")


def test_generate_texts():
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2)).eval().to(DEVICE)
    gens = generate_texts(
        model, tok, PROMPTS[:2],
        device=DEVICE, max_new_tokens=4, samples=1,
    )
    if len(gens) != 2:
        print(f"    生成条数={len(gens)}，期望 2")
        raise AssertionError("每个 prompt 应各生成 1 条")
    for item in gens:
        if not isinstance(item["text"], str):
            raise AssertionError("生成文本应为 str")
        if item["prompt"] not in PROMPTS[:2]:
            raise AssertionError("返回的 prompt 不对")


def test_benchmark():
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = CausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2)).eval().to(DEVICE)
    result = benchmark(model, tok, device=DEVICE, max_new_tokens=8)
    if result["tokens_per_s"] <= 0:
        raise AssertionError("tokens/s 应为正")
    if result["new_tokens"] <= 0:
        raise AssertionError("生成的 token 数应为正")
    if "cuda" in DEVICE and (result["peak_mb"] is None or result["peak_mb"] <= 0):
        raise AssertionError("cuda 下应记录显存峰值")
    print(f"      benchmark: {result['tokens_per_s']:.1f} tok/s, peak={result['peak_mb']}")


def test_report():
    work = work_dir() / "eval_report"
    rows = [
        {"weight": "pretrain_128", "ppl": 12.34, "tokens_per_s": 100.0, "peak_mb": 200.0,
        "samples": [{"prompt": PROMPTS[0], "sample": 0, "text": "回答 A"}]},
        {"weight": "full_sft_128", "ppl": 11.11, "tokens_per_s": 110.0, "peak_mb": 210.0,
        "samples": [{"prompt": PROMPTS[0], "sample": 0, "text": "回答 B"}]},
    ]
    generations = {
        "pretrain_128": [{"prompt": PROMPTS[0], "sample": 0, "text": "回答 A"}],
        "full_sft_128": [{"prompt": PROMPTS[0], "sample": 0, "text": "回答 B"}],
    }
    csv_path, report_md = write_report(rows, work, generations)
    if not csv_path.exists() or not report_md.exists():
        raise AssertionError("report.csv / report.md 未生成")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    if header != ["weight", "ppl", "tokens_per_s", "peak_mb", "sample"]:
        print(f"    CSV 表头={header}")
        raise AssertionError("report.csv 表头不对")
    gen_md = work / "generations_full_sft_128.md"
    if not gen_md.exists() or PROMPTS[0] not in gen_md.read_text(encoding="utf-8"):
        raise AssertionError("generations_*.md 内容不对")
    print(f"      报告：{report_md.name}, {csv_path.name}, {gen_md.name}")


# ============================================================================
TESTS = [test_ppl_uniform, test_ppl_perfect, test_generate_texts, test_benchmark, test_report]

HINTS = {
    "test_ppl_uniform": "PPL = exp(总 NLL / token 数)；均匀分布时每个 token NLL=ln(vocab)",
    "test_ppl_perfect": "滑动窗口重叠部分不能重复计入；第一个窗口统计全部，之后只统计新覆盖的 stride 个",
    "test_generate_texts": "用 apply_chat_template + 你的 sampling.generate",
    "test_benchmark": "tokens/s = 新 token 数 / 耗时；cuda 下记录 max_memory_allocated",
    "test_report": "write_report 输出 csv + md + 每个权重的 generations_*.md",
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
    print(f"全部 {len(selected)} 项通过，D12 的评估脚本可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
