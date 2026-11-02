"""命令行入口（给普通用户使用的 exe 版本）。

两种模式：
    minimind                         # 交互式文字对话
    minimind "你好"                  # 单次问答
    minimind --image 图片.jpg        # 交互式看图问答（多轮）
    minimind --image 图片.jpg "描述这张图"   # 单次看图问答

模型文件放在 exe 同级的 model/ 目录：
    model/
      full_sft_768.pth        # 文字模型（必需）
      sft_vlm_768.pth         # 看图模型（可选）
      tokenizer/              # tokenizer.json + tokenizer_config.json
      siglip2/                # 视觉编码器（看图模式需要）

开发时也可以：
    python -m mymm.desktop --model-dir out/save --tokenizer-dir D:\\minimind\\minimind\\model
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

__all__ = ["main", "app_dir", "default_model_dir"]

APP_NAME = "MiniMind"
VERSION = "1.0"
DEFAULT_HIDDEN = 768
DEFAULT_LAYERS = 8


def app_dir() -> Path:
    """exe 所在目录（打包后）或项目 learning\\scratch 目录（源码运行时）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def default_model_dir() -> Path:
    if getattr(sys, "frozen", False):
        return app_dir() / "model"
    return app_dir() / "out" / "save"


def _setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _die(message: str, code: int = 2):
    print(f"\n[错误] {message}\n")
    raise SystemExit(code)


def _pick_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        print("[提示] 没有检测到 CUDA，自动改用 CPU 运行。")
        return "cpu"
    return choice


def _tokenizer_dir(args) -> Path:
    if args.tokenizer_dir:
        return Path(args.tokenizer_dir)
    local = Path(args.model_dir) / "tokenizer"
    if (local / "tokenizer.json").exists():
        return local
    # 源码开发环境：复用仓库里的 tokenizer
    fallback = Path(r"D:\minimind\minimind\model")
    if fallback.exists():
        return fallback
    _die(f"找不到 tokenizer：{local}\n请把 tokenizer.json / tokenizer_config.json 放进 {local}，"
         "或用 --tokenizer-dir 指定目录")


class LLMEngine:
    """文字对话引擎（懒加载，第一次用时才占内存）。"""

    def __init__(self, args):
        from transformers import AutoTokenizer

        from .config import MiniMindConfig
        from .model import CausalLM

        self.device = _pick_device(args.device)
        weight = Path(args.weight_path) if args.weight_path else Path(args.model_dir) / f"{args.weight}_{args.hidden_size}.pth"
        if not weight.exists():
            _die(f"找不到文字模型：{weight}\n请把 full_sft_768.pth 放到 model/ 目录，或用 --weight-path 指定")
        tokenizer_dir = _tokenizer_dir(args)
        print(f"正在加载文字模型：{weight.name} ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.model = CausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers))
        state = torch.load(weight, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        print(f"加载完成（{time.time() - t0:.1f}s，设备：{self.device.upper()}）\n", flush=True)

    @torch.no_grad()
    def reply(self, messages, *, max_new_tokens=256, temperature=0.85, top_p=0.85, top_k=50, open_thinking=False):
        from .sampling import generate

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            open_thinking=bool(open_thinking),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        started = time.time()
        out = generate(
            self.model,
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.0,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        elapsed = time.time() - started
        text = self.tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return text, out.shape[1] - inputs.input_ids.shape[1], elapsed


class VLMEngine:
    """看图对话引擎（懒加载）。"""

    def __init__(self, args):
        from transformers import AutoTokenizer

        from .config import MiniMindConfig
        from .vlm import MiniMindVLM

        self.device = _pick_device(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(_tokenizer_dir(args)))

        weight = Path(args.vlm_weight_path) if args.vlm_weight_path else Path(args.model_dir) / f"sft_vlm_{args.hidden_size}.pth"
        if not weight.exists():
            _die(f"找不到看图模型：{weight}\n请把 sft_vlm_768.pth 放到 model/ 目录，或用 --vlm-weight-path 指定")
        vision_dir = Path(args.vision_dir) if args.vision_dir else Path(args.model_dir) / "siglip2"
        if not (vision_dir / "config.json").exists() and not (vision_dir / "model.safetensors").exists():
            fallback = Path(r"D:\minimind\minimind-v\model\siglip2-base-p32-256-ve")
            if fallback.exists():
                vision_dir = fallback
            else:
                _die(f"找不到视觉编码器：{vision_dir}\n请把 siglip2 目录放到 model/ 下，或用 --vision-dir 指定")

        print(f"正在加载看图模型：{weight.name} ...", flush=True)
        t0 = time.time()
        self.model = MiniMindVLM(
            MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers),
            vision_model_path=str(vision_dir),
        )
        state = torch.load(weight, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if any("vision_proj" in key for key in missing):
            _die("这个权重里没有 vision_proj，不是 VLM 权重")
        self.model = self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        print(f"加载完成（{time.time() - t0:.1f}s，设备：{self.device.upper()}）\n", flush=True)

    def reply(self, image, prompt, *, max_new_tokens=128, temperature=0.7, top_p=0.85, top_k=50):
        from .vlm_chat import vlm_generate

        return vlm_generate(
            self.model, self.tokenizer, image, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, device=self.device,
        )


def _chat_interactive(engine: LLMEngine, args):
    print("进入对话模式：直接输入问题，输入 /clear 清空上下文，/exit 退出。\n")
    messages = []
    while True:
        try:
            user = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0
        if not user:
            continue
        if user in ("/exit", "/quit", "exit", "quit"):
            print("再见！")
            return 0
        if user == "/clear":
            messages.clear()
            print("（上下文已清空）\n")
            continue
        messages.append({"role": "user", "content": user})
        history = messages[-(args.history * 2 + 1):] if args.history > 0 else messages[-1:]
        reply, n, dt = engine.reply(history, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
        print(f"{APP_NAME}：{reply.strip()}\n")
        if args.show_speed:
            print(f"  [{n} tokens, {n / max(dt, 1e-6):.1f} tok/s]\n")
        messages.append({"role": "assistant", "content": reply})


def _vlm_interactive(engine: VLMEngine, image, args):
    print("进入看图对话：输入问题，/exit 退出。\n")
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0
        if not question:
            continue
        if question in ("/exit", "/quit", "exit", "quit"):
            print("再见！")
            return 0
        reply, n, dt = engine.reply(image, question, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
        print(f"{APP_NAME}：{reply.strip()}\n")
        if args.show_speed:
            print(f"  [{n} tokens, {n / max(dt, 1e-6):.1f} tok/s]\n")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="minimind",
        description=f"{APP_NAME} 命令行版（文字对话 / 看图问答）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", default="", help="一次性问题；不填则进入交互模式")
    parser.add_argument("--image", default="", help="图片路径（填了就是看图模式）")
    parser.add_argument("--model-dir", default=str(default_model_dir()), help="模型目录（含 .pth 和 tokenizer/）")
    parser.add_argument("--tokenizer-dir", default="", help="tokenizer 目录（默认 model/tokenizer）")
    parser.add_argument("--weight", default="full_sft", help="文字权重前缀（model/<weight>_<hidden>.pth）")
    parser.add_argument("--weight-path", default="", help="直接指定文字权重路径")
    parser.add_argument("--vlm-weight-path", default="", help="直接指定看图权重路径")
    parser.add_argument("--vision-dir", default="", help="SigLIP2 目录（默认 model/siglip2）")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--num-hidden-layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--history", type=int, default=0, help="携带最近 N 轮历史")
    parser.add_argument("--show-speed", type=int, default=0, choices=[0, 1])
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {VERSION}")
    return parser


def main(argv=None) -> int:
    _setup_console()
    args = _build_parser().parse_args(argv)
    args.model_dir = str(Path(args.model_dir).expanduser().resolve())

    if args.image:
        from PIL import Image

        image_path = Path(args.image).expanduser()
        if not image_path.exists():
            _die(f"找不到图片：{image_path}")
        engine = VLMEngine(args)
        try:
            image = Image.open(image_path)
        except Exception as exc:
            _die(f"图片打不开：{exc}")
        if args.prompt:
            reply, n, dt = engine.reply(image, args.prompt,
                                        max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature,
                                        top_p=args.top_p, top_k=args.top_k)
            print(f"🖼  {image_path.name}")
            print(f"💬 {args.prompt}")
            print(f"{APP_NAME}：{reply.strip()}")
            if args.show_speed:
                print(f"  [{n} tokens, {n / max(dt, 1e-6):.1f} tok/s]")
            return 0
        return _vlm_interactive(engine, image, args)

    engine = LLMEngine(args)
    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        reply, n, dt = engine.reply(messages, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p, top_k=args.top_k)
        print(f"{APP_NAME}：{reply.strip()}")
        if args.show_speed:
            print(f"  [{n} tokens, {n / max(dt, 1e-6):.1f} tok/s]")
        return 0
    return _chat_interactive(engine, args)


if __name__ == "__main__":
    sys.exit(main())
