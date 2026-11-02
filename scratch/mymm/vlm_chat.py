"""D17 · VLM 推理/对话（LEARNING_TASKS: S9-10 / S9-13）

加载 sft_vlm_{hidden}.pth + SigLIP2，对单张图片生成回答：
    <image> 展开成 64 个 image_pad → VLM.forward 注入视觉特征 → 自回归生成

用法（在 learning\\scratch 下）：
    # 对一张本地图片提问
    python -m mymm.vlm_chat --hidden_size 128 --num_hidden_layers 2 `
        --weight sft_vlm --image D:\\minimind\\minimind-v\\dataset\\eval_images\\image-01-golden-dog-balloons.jpg `
        --prompt "请描述这张图中的主要物体和场景。"

    # 自动跑 eval_images 下所有图片（默认）
    python -m mymm.vlm_chat --hidden_size 128 --num_hidden_layers 2 --auto 1

    # 768 权重
    python -m mymm.vlm_chat --hidden_size 768 --num_hidden_layers 8 --weight sft_vlm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mymm"

from .config import MiniMindConfig  # noqa: E402
from .sampling import generate as my_generate  # noqa: E402
from .vlm import VISION_MODEL_PATH, MiniMindVLM  # noqa: E402

TOKENIZER_DIR = r"D:\minimind\minimind\model"
EVAL_IMAGES = r"D:\minimind\minimind-v\dataset\eval_images"
DEFAULT_PROMPT = "请描述这张图中的主要物体和场景。"

__all__ = ["load_vlm", "vlm_generate", "parse_args", "main"]


def load_vlm(
    save_dir: str = "out", weight: str = "sft_vlm", hidden_size: int = 128,
    num_hidden_layers: int = 2, vision_model_path: str = VISION_MODEL_PATH,
    device: str = "cuda", weight_path: str = "",
):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    model = MiniMindVLM(config, vision_model_path=vision_model_path)
    path = Path(weight_path) if weight_path else Path(save_dir) / f"{weight}_{hidden_size}.pth"
    if not path.exists():
        raise FileNotFoundError(f"找不到 VLM 权重：{path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any("vision_proj" in key for key in missing):
        raise ValueError(f"{path} 里没有 vision_proj，确认是 VLM 权重而不是纯 LLM 权重")
    model = model.eval()
    if "cuda" in device:
        model = model.half()
    return model.to(device), tokenizer, path


def vlm_generate(
    model, tokenizer, image: Image.Image, prompt: str, *, max_new_tokens: int = 128,
    temperature: float = 0.7, top_p: float = 0.85, top_k: int = 50,
    open_thinking: bool = False, device: str = "cuda",
):
    """返回 (回答文本, 生成 token 数, 耗时秒)。"""
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    marker = model.config.image_special_token * model.config.image_token_len
    content = prompt.replace("<image>", marker)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=bool(open_thinking),
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    image = image.convert("RGB") if image.mode != "RGB" else image
    pixel_values = MiniMindVLM.image2tensor(image, model.processor)["pixel_values"]
    pixel_values = pixel_values.to(device=device, dtype=next(model.parameters()).dtype)

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
        pixel_values=pixel_values,
    )
    elapsed = time.time() - started
    new_tokens = out.shape[1] - inputs.input_ids.shape[1]
    reply = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return reply, new_tokens, elapsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind VLM chat（D17）")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="sft_vlm")
    parser.add_argument("--weight_path", default="")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--vision_model_path", default=VISION_MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image", default="", help="单张图片路径；为空时跑 --images_dir 下全部图片")
    parser.add_argument("--images_dir", default=EVAL_IMAGES)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--auto", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--open_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--show_speed", type=int, default=1, choices=[0, 1])
    return parser.parse_args(argv)


def _print_reply(reply, new_tokens, elapsed, show_speed):
    print(f"🤖: {reply}")
    if show_speed:
        print(f"    [{new_tokens} tokens, {new_tokens / max(elapsed, 1e-6):.1f} tok/s]")


def main(argv=None) -> int:
    args = parse_args(argv)
    model, tokenizer, path = load_vlm(
        args.save_dir, args.weight, args.hidden_size, args.num_hidden_layers,
        vision_model_path=args.vision_model_path, device=args.device, weight_path=args.weight_path,
    )
    print(f"Loaded {path} | params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if args.image:
        images = [Path(args.image)]
    else:
        images = [
            p for p in sorted(Path(args.images_dir).iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        if not images:
            print(f"{args.images_dir} 下没有图片")
            return 2

    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        open_thinking=bool(args.open_thinking),
        device=args.device,
    )
    for image_path in images:
        image = Image.open(image_path)
        print(f"[图像]: {image_path.name}")
        print(f"💬: {args.prompt}")
        reply, n, dt = vlm_generate(model, tokenizer, image, args.prompt, **kwargs)
        _print_reply(reply, n, dt, args.show_speed)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
