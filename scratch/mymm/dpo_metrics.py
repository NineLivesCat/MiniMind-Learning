"""D14 · DPO margin 统计（LEARNING_TASKS: S7-09）

对比「DPO 前（full_sft）」和「DPO 后（dpo）」在偏好数据上的
    chosen logp / rejected logp / reward margin / accuracy

用法（在 learning\\scratch 下）：
    python -m mymm.dpo_metrics --hidden_size 128 --num_hidden_layers 2 --num_samples 100

margin = β · [ (logπ(y_w) - logπ(y_l)) - (logπ_ref(y_w) - logπ_ref(y_l)) ]
        = chosen_reward - rejected_reward
DPO 训练理想情况下：chosen logp 上升、rejected logp 下降、margin 均值变大。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .data import DPODataset  # noqa: E402
from .dpo import sequence_logps  # noqa: E402
from .model import CausalLM  # noqa: E402

DPO_DATA = r"D:\minimind\learning\data\dpo_500.jsonl"
TOKENIZER_DIR = r"D:\minimind\minimind\model"

__all__ = ["load_policy", "evaluate_margins", "parse_args", "main"]


def load_policy(save_dir, weight: str, hidden_size: int, num_hidden_layers: int, device: str):
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    model = CausalLM(config)
    path = Path(save_dir) / f"{weight}_{hidden_size}.pth"
    if not path.exists():
        raise FileNotFoundError(f"找不到权重：{path}")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device), path


def evaluate_margins(policy, ref, loader, *, beta: float = 0.1, device: str = "cuda") -> dict:
    """遍历 loader，返回平均 chosen/rejected logp、margin、accuracy。"""
    policy.eval()
    ref.eval()
    totals = {"policy_chosen": 0.0, "policy_rejected": 0.0, "ref_chosen": 0.0,
        "ref_rejected": 0.0, "margin": 0.0, "correct": 0.0, "n": 0}
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            pc = sequence_logps(policy, batch["x_chosen"], batch["y_chosen"], batch["mask_chosen"])
            pr = sequence_logps(policy, batch["x_rejected"], batch["y_rejected"], batch["mask_rejected"])
            rc = sequence_logps(ref, batch["x_chosen"], batch["y_chosen"], batch["mask_chosen"])
            rr = sequence_logps(ref, batch["x_rejected"], batch["y_rejected"], batch["mask_rejected"])
        margin = beta * ((pc - pr) - (rc - rr))
        totals["policy_chosen"] += pc.sum().item()
        totals["policy_rejected"] += pr.sum().item()
        totals["ref_chosen"] += rc.sum().item()
        totals["ref_rejected"] += rr.sum().item()
        totals["margin"] += margin.sum().item()
        totals["correct"] += (margin > 0).float().sum().item()
        totals["n"] += pc.numel()
    n = max(totals["n"], 1)
    return {
        "policy_chosen": totals["policy_chosen"] / n,
        "policy_rejected": totals["policy_rejected"] / n,
        "ref_chosen": totals["ref_chosen"] / n,
        "ref_rejected": totals["ref_rejected"] / n,
        "margin": totals["margin"] / n,
        "accuracy": totals["correct"] / n,
        "samples": totals["n"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DPO margin 统计（D14）")
    parser.add_argument("--data_path", default=DPO_DATA)
    parser.add_argument("--tokenizer_dir", default=TOKENIZER_DIR)
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--before", default="full_sft", help="DPO 前的权重前缀")
    parser.add_argument("--after", default="dpo", help="DPO 后的权重前缀")
    parser.add_argument("--ref", default="full_sft", help="reference 权重前缀")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    subset = Subset(dataset, list(range(min(args.num_samples, len(dataset)))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ref, ref_path = load_policy(args.save_dir, args.ref, args.hidden_size, args.num_hidden_layers, args.device)
    print(f"ref: {ref_path.name}")
    rows = []
    for stage, weight in (("before(DPO 前)", args.before), ("after(DPO 后)", args.after)):
        policy, path = load_policy(args.save_dir, weight, args.hidden_size, args.num_hidden_layers, args.device)
        stats = evaluate_margins(policy, ref, loader, beta=args.beta, device=args.device)
        rows.append((stage, path.name, stats))
        print(
            f"{stage:16s} policy_chosen={stats['policy_chosen']:+.3f} "
            f"policy_rejected={stats['policy_rejected']:+.3f} "
            f"margin={stats['margin']:+.4f} acc={stats['accuracy']:.3f}"
        )

    print()
    print(f"{'stage':16s} {'file':24s} {'chosen':>9s} {'rejected':>9s} {'margin':>9s} {'acc':>6s}")
    for stage, name, stats in rows:
        print(
            f"{stage:16s} {name:24s} {stats['policy_chosen']:>9.3f} {stats['policy_rejected']:>9.3f} "
            f"{stats['margin']:>9.4f} {stats['accuracy']:>6.3f}"
        )
    print(f"\nsamples={rows[-1][2]['samples']} | beta={args.beta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
