#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create small dataset subsets for fast experiments while learning MiniMind.

The full datasets are 1-14 GB and one epoch takes hours; while learning it is
much better to iterate on a few thousand samples (seconds per experiment).

Usage (from D:\\minimind):
    .venv\\Scripts\\python.exe learning\\make_subsets.py
    .venv\\Scripts\\python.exe learning\\make_subsets.py --pretrain 5000 --sft 2000 --vlm 1000

Outputs go to D:\\minimind\\learning\\data\\:
    pretrain_5k.jsonl, sft_2k.jsonl, dpo_500.jsonl, rlaif_500.jsonl,
    agent_rl_500.jsonl, lora_medical_500.jsonl, vlm_1k.parquet
"""
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"

TEXT_SOURCES = {
    "pretrain": ROOT / "minimind_dataset" / "pretrain_t2t_mini.jsonl",
    "sft": ROOT / "minimind_dataset" / "sft_t2t_mini.jsonl",
    "dpo": ROOT / "minimind_dataset" / "dpo.jsonl",
    "rlaif": ROOT / "minimind_dataset" / "rlaif.jsonl",
    "agent_rl": ROOT / "minimind_dataset" / "agent_rl.jsonl",
    "lora_medical": ROOT / "minimind_dataset" / "lora_medical.jsonl",
    "lora_identity": ROOT / "minimind_dataset" / "lora_identity.jsonl",
}

PARQUET_SOURCES = {
    "vlm": ROOT / "minimind-v_dataset" / "pretrain_i2t.parquet",
    "vlm_sft": ROOT / "minimind-v_dataset" / "sft_i2t.parquet",
}


def head_jsonl(src: Path, dst: Path, n: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(src, "r", encoding="utf-8", errors="ignore") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if written >= n:
                break
            if not line.strip():
                continue
            fo.write(line if line.endswith("\n") else line + "\n")
            written += 1
    print(f"  {dst.relative_to(ROOT)}: {written} rows")
    return written


def head_parquet(src: Path, dst: Path, n: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(src)
    parts, got = [], 0
    for rg in range(pf.metadata.num_row_groups):
        if got >= n:
            break
        table = pf.read_row_group(rg)
        take = table.slice(0, n - got)
        parts.append(take)
        got += take.num_rows
    if not parts:
        raise RuntimeError(f"no rows read from {src}")
    out = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
    pq.write_table(out, dst)
    print(f"  {dst.relative_to(ROOT)}: {got} rows")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrain", type=int, default=5000, help="rows for pretrain subset")
    ap.add_argument("--sft", type=int, default=2000, help="rows for SFT subset")
    ap.add_argument("--dpo", type=int, default=500, help="rows for DPO subset")
    ap.add_argument("--rlaif", type=int, default=500, help="rows for RLAIF/PPO/GRPO subset")
    ap.add_argument("--agent", type=int, default=500, help="rows for Agent-RL subset")
    ap.add_argument("--lora-medical", dest="lora_medical", type=int, default=500, help="rows for LoRA medical subset")
    ap.add_argument("--vlm", type=int, default=1000, help="rows for VLM pretrain subset (contains images)")
    ap.add_argument("--vlm-sft", dest="vlm_sft", type=int, default=500, help="rows for VLM SFT subset")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing subsets to {OUT}")

    jobs = [
        ("pretrain", args.pretrain), ("sft", args.sft), ("dpo", args.dpo),
        ("rlaif", args.rlaif), ("agent_rl", args.agent), ("lora_medical", args.lora_medical),
    ]
    for key, n in jobs:
        if n <= 0:
            continue
        src = TEXT_SOURCES[key]
        if not src.exists():
            print(f"  skip {key}: missing {src}")
            continue
        head_jsonl(src, OUT / f"{key}_{n}.jsonl", n)

    for key, n in (("vlm", args.vlm), ("vlm_sft", args.vlm_sft)):
        if n <= 0:
            continue
        src = PARQUET_SOURCES[key]
        if not src.exists():
            print(f"  skip {key}: missing {src}")
            continue
        head_parquet(src, OUT / f"{key}_{n}.parquet", n)


if __name__ == "__main__":
    main()
