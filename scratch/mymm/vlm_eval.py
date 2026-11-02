"""D17 · VLM 评测（LEARNING_TASKS: S9-10 / S9-11 / S9-12 / S9-13 / S9-15）

固定图片 × 固定问题，生成回答并保存成可人工打分的报告：
    out/vlm_eval/report.md / report.csv

用法（在 learning\\scratch 下）：
    # 128 冒烟：6 张图 × 3 个问题，生成长度 64
    python -m mymm.vlm_eval --hidden_size 128 --num_hidden_layers 2 --max_new_tokens 64

    # 768 正式评测
    python -m mymm.vlm_eval --hidden_size 768 --num_hidden_layers 8 --max_new_tokens 128

报告列：图像 / 问题 / 回答 / 人工评分(1-5) / 备注，最后按 S9-15 汇总。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .vlm_chat import EVAL_IMAGES, VISION_MODEL_PATH, load_vlm, vlm_generate  # noqa: E402

QUESTIONS = [
    "请描述这张图中的主要物体和场景。",
    "图中有哪些颜色？",
    "这张图可能是在什么时间或天气下拍摄的？",
]

__all__ = ["QUESTIONS", "evaluate", "write_report", "parse_args", "main"]


def evaluate(model, tokenizer, images: list[Path], questions: list[str], *, args) -> list[dict]:
    rows = []
    for image_path in images:
        image = Image.open(image_path)
        for question in questions:
            reply, n, dt = vlm_generate(
                model, tokenizer, image, question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                open_thinking=bool(args.open_thinking),
                device=args.device,
            )
            rows.append({
                "image": image_path.name,
                "question": question,
                "answer": reply.strip(),
                "tokens": n,
                "seconds": dt,
            })
            print(f"[{image_path.name}] {question[:20]}... -> {reply.strip()[:50]!r} ({n} tok, {dt:.2f}s)")
    return rows


def write_report(
    rows: list[dict], out_dir: Path, *, weight_path: Path, params_m: float,
    device: str, peak_mb: float | None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = sum(r["tokens"] for r in rows)
    total_seconds = sum(r["seconds"] for r in rows)
    speed = total_tokens / max(total_seconds, 1e-6)

    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "question", "answer", "tokens", "seconds", "score_1_5", "note"])
        for row in rows:
            writer.writerow([
                row["image"], row["question"], row["answer"], row["tokens"],
                f"{row['seconds']:.3f}", "", "",
            ])

    lines = [
        "# VLM 评测报告",
        "",
        f"- 权重：`{weight_path.name}`",
        f"- 参数量：{params_m:.2f}M | 设备：{device}",
        f"- 生成：{speed:.1f} tokens/s | 显存峰值：{'无 CUDA' if peak_mb is None else f'{peak_mb:.0f} MB'}",
        f"- 样本数：{len(rows)}（{len({r['image'] for r in rows})} 张图 × {len({r['question'] for r in rows})} 个问题）",
        "",
        "| 图像 | 问题 | 回答 | 人工评分(1-5) | 备注 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        answer = row["answer"].replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(f"| {row['image']} | {row['question']} | {answer} |  |  |")
    lines += [
        "",
        "## 汇总（按 S9-15 填）",
        "",
        "- 正确/合理条数： / " + str(len(rows)),
        "- 典型失败案例（S9-12）：",
        "- 每类问题的平均分：",
    ]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind VLM 评测（D17）")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="sft_vlm")
    parser.add_argument("--weight_path", default="")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--vision_model_path", default=VISION_MODEL_PATH)
    parser.add_argument("--images_dir", default=EVAL_IMAGES)
    parser.add_argument("--max_images", type=int, default=6)
    parser.add_argument("--questions", nargs="*", default=QUESTIONS)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--open_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--out_dir", default="out/vlm_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model, tokenizer, weight_path = load_vlm(
        args.save_dir, args.weight, args.hidden_size, args.num_hidden_layers,
        vision_model_path=args.vision_model_path, device=args.device, weight_path=args.weight_path,
    )
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {weight_path} | params={params_m:.2f}M")

    images = [
        p for p in sorted(Path(args.images_dir).iterdir())
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ][: args.max_images]
    if not images:
        print(f"{args.images_dir} 下没有图片")
        return 2
    questions = list(args.questions) if args.questions else QUESTIONS

    if "cuda" in args.device:
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    rows = evaluate(model, tokenizer, images, questions, args=args)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if "cuda" in args.device else None

    report, csv_path = write_report(
        rows, Path(args.out_dir), weight_path=weight_path, params_m=params_m,
        device=args.device, peak_mb=peak_mb,
    )
    print(f"\n完成 {len(rows)} 条，用时 {time.time() - started:.1f}s")
    print(f"报告：{report}")
    print(f"CSV ：{csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
