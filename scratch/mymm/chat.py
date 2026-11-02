"""D11 · 对话/评测脚本（LEARNING_TASKS: S5-07 / S5-08）

用你自己的模型 + sampling.generate 做推理/对话，不依赖仓库 eval_llm.py。

用法（在 learning\\scratch 下）：
    # 交互式对话（默认加载 out/full_sft_{hidden}.pth）
    python -m mymm.chat --hidden_size 128 --num_hidden_layers 2

    # 自动跑几条预设问题（方便对比权重）
    python -m mymm.chat --auto 1 --hidden_size 128 --num_hidden_layers 2

    # 单条问题 + 调采样参数
    python -m mymm.chat --prompt "你好，你是谁？" --temperature 0.85 --top_p 0.95 --max_new_tokens 128

    # 加载 LoRA 推理（S7-04）
    python -m mymm.chat --hidden_size 128 --num_hidden_layers 2 --lora out/lora_medical_tiny_128.pth

    # 官方 768 权重（S5-08）：指向 minimind/out
    python -m mymm.chat --save_dir D:\\minimind\\minimind\\out --weight full_sft `
        --hidden_size 768 --num_hidden_layers 8 --auto 1

常用参数：
    --historys 4        带最近 4 条历史（0=不带）
    --open_thinking 1   允许模型输出 <think> 推理块
    --show_speed 1      打印 tokens/s
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .lora import apply_lora, load_lora  # noqa: E402
from .model import CausalLM  # noqa: E402
from .sampling import generate as my_generate  # noqa: E402

__all__ = ["build_model", "generate_reply", "PROMPTS", "parse_args", "main"]

PROMPTS = [
    "你有什么特长？",
    "为什么天空是蓝色的？",
    "请用 Python 写一个计算斐波那契数列的函数。",
    "解释一下什么是机器学习。",
    "如果明天下雨，我应该如何出门？",
    "推荐一些中国的美食。",
    "用一句话介绍你自己。",
    "比较一下猫和狗作为宠物的优缺点。",
]


def build_model(args):
    """加载 tokenizer + CausalLM + checkpoint，返回 (model, tokenizer)。"""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model = CausalLM(config)

    weight_path = Path(args.weight_path) if args.weight_path else (
        Path(args.save_dir) / f"{args.weight}_{args.hidden_size}.pth"
    )
    if not weight_path.exists():
        raise FileNotFoundError(
            f"找不到权重：{weight_path}\n"
            f"先训练出 {args.weight}_{args.hidden_size}.pth，或用 --weight_path 指定绝对路径"
        )
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    if getattr(args, "lora", ""):
        applied = apply_lora(model, rank=getattr(args, "lora_rank", 16))
        loaded = load_lora(model, args.lora)
        print(f"LoRA: {loaded}/{applied} modules from {args.lora}")

    model = model.eval()
    if "cuda" in args.device:
        model = model.half()
    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded {weight_path} | Model Params: {total / 1e6:.2f}M")
    return model.to(args.device), tokenizer


def generate_reply(
    model, tokenizer, messages, *, historys=0, max_new_tokens=256,
    temperature=0.85, top_p=0.85, top_k=50, open_thinking=False,
):
    """给定完整对话（最后一条是当前 user），返回 (回复文本, 生成 token 数, 耗时秒)。"""
    current = messages[-1]
    history = messages[:-1]
    if historys > 0:
        history = history[-historys:]
    else:
        history = []
    conversation = history + [current]

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=bool(open_thinking),
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    started = time.time()
    out = my_generate(
        model,
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.0,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    elapsed = time.time() - started
    new_tokens = out.shape[1] - inputs.input_ids.shape[1]
    text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text, new_tokens, elapsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind chat / eval（D11）")
    parser.add_argument("--tokenizer_dir", default=r"D:\minimind\minimind\model")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--weight_path", default="", help="直接指定权重路径（优先于 save_dir/weight_hidden）")
    parser.add_argument("--lora", default="", help="LoRA 权重路径（可选；加载后自动 apply_lora + load_lora）")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--historys", type=int, default=0, help="携带最近几条历史消息（0=不带）")
    parser.add_argument("--open_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--auto", type=int, default=0, help="1=跑内置的 8 个问题")
    parser.add_argument("--prompt", default="", help="只回答这一条问题后退出")
    parser.add_argument("--show_speed", type=int, default=1, choices=[0, 1])
    return parser.parse_args(argv)


def _print_reply(text, new_tokens, elapsed, show_speed):
    print(f"🤖: {text}")
    if show_speed:
        speed = new_tokens / max(elapsed, 1e-6)
        print(f"    [{new_tokens} tokens, {speed:.1f} tok/s]")


def main(argv=None) -> int:
    args = parse_args(argv)
    model, tokenizer = build_model(args)
    kwargs = dict(
        historys=args.historys,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        open_thinking=bool(args.open_thinking),
    )

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        text, n, dt = generate_reply(model, tokenizer, messages, **kwargs)
        _print_reply(text, n, dt, args.show_speed)
        return 0

    if args.auto:
        for prompt in PROMPTS:
            messages = [{"role": "user", "content": prompt}]
            text, n, dt = generate_reply(model, tokenizer, messages, **kwargs)
            print(f"💬: {prompt}")
            _print_reply(text, n, dt, args.show_speed)
            print()
        return 0

    conversation = []
    print("进入对话模式，输入 exit 退出。")
    while True:
        try:
            user = input("💬: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("exit", "quit", ""):
            break
        conversation.append({"role": "user", "content": user})
        text, n, dt = generate_reply(model, tokenizer, conversation, **kwargs)
        _print_reply(text, n, dt, args.show_speed)
        conversation.append({"role": "assistant", "content": text})
    return 0


if __name__ == "__main__":
    sys.exit(main())
