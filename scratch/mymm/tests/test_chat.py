"""D11 验收测试：对话/评测脚本

运行：
    . D:\\minimind\\activate_env.ps1
    cd D:\\minimind\\learning\\scratch
    python mymm\\tests\\test_chat.py

覆盖：
    1) build_model 能加载 checkpoint（strict）并进入 eval
    2) generate_reply 返回文本/tokens/耗时，historys 截断不报错
    3) --prompt 和 --auto 两种非交互模式能跑通
"""
from __future__ import annotations

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

import mymm.chat as chat  # noqa: E402
from mymm.config import MiniMindConfig  # noqa: E402
from mymm.model import CausalLM  # noqa: E402

TOKENIZER_DIR = REPO / "model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STATE: dict = {}


def work_dir() -> Path:
    if "dir" not in _STATE:
        env = os.environ.get("MYMM_TEST_DIR")
        path = Path(env) if env else Path(tempfile.mkdtemp())
        path.mkdir(parents=True, exist_ok=True)
        _STATE["dir"] = path
    return _STATE["dir"]


def make_checkpoint() -> Path:
    if "ckpt" in _STATE:
        return _STATE["ckpt"]
    save_dir = work_dir() / "chat_run"
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
    model = CausalLM(cfg)
    path = save_dir / "full_sft_128.pth"
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, path)
    _STATE["ckpt"] = path
    _STATE["save_dir"] = save_dir
    return path


def make_args():
    ckpt = make_checkpoint()
    return SimpleNamespace(
        tokenizer_dir=str(TOKENIZER_DIR),
        save_dir=str(ckpt.parent),
        weight="full_sft",
        weight_path="",
        hidden_size=128,
        num_hidden_layers=2,
        device=DEVICE,
    )


# ============================================================================
def test_build_model():
    args = make_args()
    model, tokenizer = chat.build_model(args)
    if model.training:
        raise AssertionError("推理模型应处于 eval 模式")
    if tokenizer.vocab_size != 6400:
        print(f"    tokenizer vocab={tokenizer.vocab_size}")
        raise AssertionError("tokenizer 加载不对")
    if DEVICE == "cuda" and next(model.parameters()).dtype != torch.float16:
        print(f"    dtype={next(model.parameters()).dtype}，期望 float16")
        raise AssertionError("cuda 下应使用 fp16 推理")
    _STATE["model"], _STATE["tok"] = model, tokenizer


def test_generate_reply():
    if "model" not in _STATE:
        test_build_model()
    model, tokenizer = _STATE["model"], _STATE["tok"]

    started = time.time()
    text, new_tokens, elapsed = chat.generate_reply(
        model, tokenizer, [{"role": "user", "content": "你好，你是谁？"}], max_new_tokens=8
    )
    if not isinstance(text, str):
        raise AssertionError("回复应为 str")
    if new_tokens <= 0 or new_tokens > 8:
        print(f"    生成 token 数={new_tokens}，期望 1~8")
        raise AssertionError("生成 token 数异常")
    if elapsed <= 0:
        raise AssertionError("耗时应为正数")

    # historys 截断：5 条历史 + 当前问题，不应报错
    messages = []
    for i in range(5):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"历史 {i}"})
    messages.append({"role": "user", "content": "继续"})
    text2, n2, _ = chat.generate_reply(model, tokenizer, messages, historys=4, max_new_tokens=4)
    if not isinstance(text2, str):
        raise AssertionError("带历史时回复应为 str")
    print(f"      生成 {new_tokens} tok ({elapsed * 1000:.0f}ms)；带历史 {n2} tok")


def test_main_prompt():
    args = make_args()
    argv = [
        "--tokenizer_dir", args.tokenizer_dir,
        "--save_dir", args.save_dir,
        "--weight", "full_sft",
        "--hidden_size", "128",
        "--num_hidden_layers", "2",
        "--device", DEVICE,
        "--prompt", "你好",
        "--max_new_tokens", "8",
        "--show_speed", "0",
    ]
    rc = chat.main(argv)
    if rc != 0:
        raise AssertionError(f"--prompt 模式返回 {rc}")


def test_main_auto():
    args = make_args()
    argv = [
        "--tokenizer_dir", args.tokenizer_dir,
        "--save_dir", args.save_dir,
        "--weight", "full_sft",
        "--hidden_size", "128",
        "--num_hidden_layers", "2",
        "--device", DEVICE,
        "--auto", "1",
        "--max_new_tokens", "4",
        "--show_speed", "0",
    ]
    started = time.time()
    rc = chat.main(argv)
    if rc != 0:
        raise AssertionError(f"--auto 模式返回 {rc}")
    print(f"      8 条自动问题耗时 {time.time() - started:.1f}s")


# ============================================================================
TESTS = [test_build_model, test_generate_reply, test_main_prompt, test_main_auto]

HINTS = {
    "test_build_model": "checkpoint 是 fp16；cuda 下模型转 half，加载用 strict=True",
    "test_generate_reply": "historys 截断后要用 apply_chat_template + 你的 sampling.generate",
    "test_main_prompt": "main 里 --prompt 模式应单轮回答后 return 0",
    "test_main_auto": "--auto 模式遍历内置问题，不报错即可",
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
    print(f"全部 {len(selected)} 项通过，D11 的 chat 脚本可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
