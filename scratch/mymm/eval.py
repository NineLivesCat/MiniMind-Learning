"""D12 · 评估（LEARNING_TASKS: S6-01 / S6-02 / S6-03 / S6-04 / S6-07）

三件事：
    1) 手写滑动窗口 PPL（exp(平均交叉熵)）
    2) 固定 20 个 prompt 的生成评测（每个权重可出 3 个样本，人工打分）
    3) 吞吐（tokens/s）与显存峰值，最后汇总成 PPL / 主观质量 / 速度 / 显存 四列表

用法（在 learning\\scratch 下）：
    # 对比 pretrain 和 full_sft 两个权重（默认 200 条文本、窗口 512/步长 256）
    python -m mymm.eval --weights pretrain full_sft

    # 每个 prompt 生成 3 个样本，方便人工打分（S6-03）
    python -m mymm.eval --weights full_sft --samples 3

    # 对比 768 权重（weights 名对应 out/<weight>_768.pth）
    python -m mymm.eval --hidden_size 768 --num_hidden_layers 8 --weights full_sft

    # 只跑 PPL，不出生成（快）
    python -m mymm.eval --weights pretrain full_sft --skip_gen 1

结果写到 out/eval/：report.md、report.csv、generations_<weight>.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .model import CausalLM  # noqa: E402
from .sampling import generate as my_generate  # noqa: E402

PROMPTS = [
    "你有什么特长？",
    "为什么天空是蓝色的？",
    "请用 Python 写一个计算斐波那契数列的函数。",
    "解释一下什么是机器学习。",
    "如果明天下雨，我应该如何出门？",
    "推荐一些中国的美食。",
    "用一句话介绍你自己。",
    "比较一下猫和狗作为宠物的优缺点。",
    "解释一下什么是光合作用。",
    "给我讲一个关于友谊的小故事。",
    "如何提高睡眠质量？",
    "写一首关于秋天的五言绝句。",
    "解释一下什么是复利。",
    "中国有多少个省级行政区？",
    "如何学习一门新的外语？",
    "解释一下什么是区块链。",
    "推荐三本值得读的书。",
    "为什么运动对健康有益？",
    "如何做一份简单的番茄炒蛋？",
    "用三句话总结一下人工智能的利弊。",
]

PPL_DATA = r"D:\minimind\learning\data\pretrain_5000.jsonl"
TOKENIZER_DIR = r"D:\minimind\minimind\model"

__all__ = [
    "PROMPTS",
    "load_texts",
    "build_ids",
    "compute_ppl",
    "generate_texts",
    "benchmark",
    "load_model",
    "write_report",
    "main",
]


def load_texts(data_path: str, num_texts: int = 200) -> list[str]:
    texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(str(json.loads(line)["text"]))
            if len(texts) >= num_texts:
                break
    return texts


def build_ids(tokenizer, texts: list[str], max_tokens: int = 8192) -> list[int]:
    ids = []
    for text in texts:
        tokens = tokenizer(str(text), add_special_tokens=False).input_ids
        ids.extend(tokens)
        ids.append(tokenizer.eos_token_id)
        if len(ids) >= max_tokens:
            break
    return ids[:max_tokens]


def compute_ppl(model, tokenizer, texts: list[str], *, max_length: int = 512,
                stride: int = 256, device: str = "cuda", max_tokens: int = 8192) -> dict:
    """滑动窗口 PPL。

    做法：
    - 把 texts 拼成一条长 token 序列（段落间加 eos）
    - 每次取 length=max_length 的窗口，步长 stride；窗口之间会有 stride 个 token 的重叠
    - 只统计「新覆盖」的 token 的 NLL：第一个窗口全部统计，之后跳过前 len(chunk)-stride 个位置
    - PPL = exp(总 NLL / 统计 token 数)

    返回 {"ppl": float, "nll": float, "tokens": int, "windows": int}
    """
    if hasattr(model, "eval"):
        model.eval()
    ids = build_ids(tokenizer, texts, max_tokens=max_tokens)
    if len(ids) < 2:
        raise ValueError("文本太短，无法计算 PPL")

    total_nll = 0.0
    scored = 0
    windows = 0
    start = 0
    while start < len(ids):
        chunk = ids[start : start + max_length]
        if len(chunk) < 2:
            break
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[0, :-1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, targets[:, None]).squeeze(1)

        skip = 0 if start == 0 else max(0, len(chunk) - stride)
        kept = nll[skip:]
        total_nll += kept.sum().item()
        scored += kept.numel()
        windows += 1

        if start + max_length >= len(ids):
            break
        start += stride

    mean_nll = total_nll / max(scored, 1)
    return {"ppl": math.exp(mean_nll), "nll": mean_nll, "tokens": scored, "windows": windows}


def generate_texts(
    model, tokenizer, prompts: list[str], *, device: str = "cuda",
    max_new_tokens: int = 64, temperature: float = 0.7, top_p: float = 0.85,
    top_k: int = 50, samples: int = 1,
) -> list[dict]:
    """对每个 prompt 生成 samples 个回复，返回 [{prompt, sample, text}, ...]。"""
    if hasattr(model, "eval"):
        model.eval()
    results = []
    for prompt in prompts:
        prompt_ids = tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            return_tensors="pt",
        ).input_ids.to(device)
        for k in range(samples):
            out = my_generate(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.0,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
            results.append({"prompt": prompt, "sample": k, "text": text})
    return results


def benchmark(model, tokenizer, *, device: str = "cuda", max_new_tokens: int = 64) -> dict:
    """测一次生成的 tokens/s 与显存峰值（单位 MB）。"""
    if hasattr(model, "eval"):
        model.eval()
    prompt = "请用一句话介绍你自己。"
    input_ids = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        ),
        return_tensors="pt",
    ).input_ids.to(device)
    if "cuda" in device:
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    out = my_generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    elapsed = time.time() - started
    new_tokens = out.shape[1] - input_ids.shape[1]
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if "cuda" in device else None
    return {"tokens_per_s": new_tokens / max(elapsed, 1e-6), "peak_mb": peak_mb, "new_tokens": new_tokens}


def load_model(
    save_dir: str, weight: str, hidden_size: int, num_hidden_layers: int,
    *, device: str = "cuda", weight_path: str = "",
):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    model = CausalLM(config)
    if weight_path:
        path = Path(weight_path)
    elif weight.endswith(f"_{hidden_size}"):
        path = Path(save_dir) / f"{weight}.pth"
    else:
        path = Path(save_dir) / f"{weight}_{hidden_size}.pth"
    if not path.exists():
        raise FileNotFoundError(f"找不到权重：{path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.eval()
    if "cuda" in device:
        model = model.half()
    return model.to(device), tokenizer, path


def write_report(rows: list[dict], out_dir: Path, generations: dict[str, list[dict]]):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["weight", "ppl", "tokens_per_s", "peak_mb", "sample"])
        for row in rows:
            sample = row["samples"][0]["text"].replace("\n", " ")[:60] if row["samples"] else ""
            ppl = "" if row["ppl"] is None else f"{row['ppl']:.4f}"
            peak = "" if row["peak_mb"] is None else f"{row['peak_mb']:.0f}"
            writer.writerow([row["weight"], ppl, f"{row['tokens_per_s']:.1f}", peak, sample])

    lines = [
        "# 评估报告",
        "",
        "| 权重 | PPL | tokens/s | 显存峰值(MB) | 生成示例 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        sample = row["samples"][0]["text"].replace("\n", " ")[:40] if row["samples"] else ""
        ppl = "-" if row["ppl"] is None else f"{row['ppl']:.2f}"
        peak = "-" if row["peak_mb"] is None else f"{row['peak_mb']:.0f}"
        lines.append(f"| {row['weight']} | {ppl} | {row['tokens_per_s']:.1f} | {peak} | {sample} |")
    lines += ["", "主观质量：打开 generations_<weight>.md 按 1-5 分人工打分后填进来。"]
    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for weight, items in generations.items():
        gen_md = out_dir / f"generations_{weight}.md"
        md = [f"# {weight} 生成评测", ""]
        for item in items:
            md.append(f"## {item['prompt']}")
            md.append(f"- sample {item['sample']}: {item['text']!r}")
            md.append(f"- 分数(1-5): ")
            md.append("")
        gen_md.write_text("\n".join(md), encoding="utf-8")
    return csv_path, report_md


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind eval（D12）")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weights", nargs="+", default=["pretrain", "full_sft"])
    parser.add_argument("--weight_path", default="", help="只评估单个权重时直接指定路径")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_texts", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--skip_gen", type=int, default=0, choices=[0, 1])
    parser.add_argument("--out_dir", default="out/eval")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    texts = load_texts(PPL_DATA, num_texts=args.num_texts)
    weights = args.weights
    if args.weight_path:
        weights = [Path(args.weight_path).stem]

    rows = []
    generations = {}
    for weight in weights:
        path = args.weight_path if args.weight_path else ""
        model, tokenizer, weight_path = load_model(
            args.save_dir, weight, args.hidden_size, args.num_hidden_layers,
            device=args.device, weight_path=path,
        )
        print(f"\n=== {weight} ({weight_path.name}) ===")
        ppl = compute_ppl(
            model, tokenizer, texts,
            max_length=args.max_length, stride=args.stride,
            device=args.device, max_tokens=args.max_tokens,
        )
        print(f"  PPL: {ppl['ppl']:.2f} | tokens={ppl['tokens']} | windows={ppl['windows']}")

        bench = benchmark(model, tokenizer, device=args.device, max_new_tokens=args.max_new_tokens)
        peak = "无 CUDA" if bench["peak_mb"] is None else f"{bench['peak_mb']:.0f} MB"
        print(f"  speed: {bench['tokens_per_s']:.1f} tok/s | peak: {peak}")

        gens = []
        if not args.skip_gen:
            gens = generate_texts(
                model, tokenizer, PROMPTS,
                device=args.device, max_new_tokens=args.max_new_tokens,
                samples=args.samples,
            )
            generations[weight] = gens
        rows.append({
            "weight": weight,
            "ppl": ppl["ppl"],
            "tokens_per_s": bench["tokens_per_s"],
            "peak_mb": bench["peak_mb"],
            "samples": gens,
        })

    csv_path, report_md = write_report(rows, Path(args.out_dir), generations)
    print("\n四列表：")
    print(f"{'weight':16s} {'PPL':>8s} {'tok/s':>8s} {'peak(MB)':>9s}")
    for row in rows:
        peak = "-" if row["peak_mb"] is None else f"{row['peak_mb']:.0f}"
        print(f"{row['weight']:16s} {row['ppl']:>8.2f} {row['tokens_per_s']:>8.1f} {peak:>9s}")
    print(f"\n报告：{report_md}")
    print(f"CSV ：{csv_path}")
    if generations:
        for weight in generations:
            print(f"生成评测（人工打分）：{Path(args.out_dir) / f'generations_{weight}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
