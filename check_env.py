#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Environment self-check for the shared MiniMind / MiniMind-V uv environment.

Usage (from the environment root D:\minimind):
    .venv\\Scripts\\python.exe check_env.py
    .venv\\Scripts\\python.exe check_env.py --skip-gpu

Exit code 0 = everything that could be tested passed.
"""
import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOS = {"minimind": ROOT / "minimind", "minimind-v": ROOT / "minimind-v"}

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_failures = []


def report(status, name, detail=""):
    line = f"[{status:<4}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if status == FAIL:
        _failures.append(name)


def section(title):
    print(f"\n=== {title} ===")


def pkg_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# 1) Global environment checks
# ---------------------------------------------------------------------------
def check_runtime():
    section("Runtime")
    print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform    : {platform.platform()}")
    print(f"cwd         : {os.getcwd()}")
    expect = (3, 11)
    ok = sys.version_info[:2] == expect
    report(PASS if ok else FAIL, "Python 3.11.x", f"got {sys.version_info[:2]}" if not ok else "")


def check_packages():
    section("Package versions")
    expected = {
        "torch": None,                 # printed with local version (cu128) below
        "transformers": "4.57.6",
        "tokenizers": "0.22.2",
        "datasets": "3.6.0",
        "pyarrow": "23.0.0",
        "numpy": "1.26.4",
        "pydantic": "2.11.5",
        "gradio": "5.49.1",
        "streamlit": "1.50.0",
        "modelscope": "1.37.0",
        "swanlab": "0.9.8",
        "wandb": "0.22.3",
        "fastapi": "0.120.1",
        "uvicorn": "0.38.0",
        "openai": "1.59.6",
        "scipy": "1.17.1",
        "matplotlib": "3.10.9",
        "pillow": "11.3.0",
    }
    for name, want in expected.items():
        got = pkg_version(name)
        if got is None:
            report(FAIL, f"{name}", "not installed")
        elif want is None or got == want:
            report(PASS, f"{name}", got)
        else:
            report(FAIL, f"{name}", f"expected {want}, got {got}")

    # torch prints its CUDA suffix (2.9.1+cu128)
    try:
        import torch
        report(PASS, "torch", f"{torch.__version__} (cuda runtime {torch.version.cuda})")
        if "+cu128" not in torch.__version__:
            report(FAIL, "torch cu128 build", f"got {torch.__version__}, need +cu128 for RTX 50 series")
    except Exception as exc:  # pragma: no cover
        report(FAIL, "torch import", repr(exc))


def check_windows_dll_order():
    """Repos import `datasets` before `torch` to dodge a pyarrow/torch DLL clash."""
    section("Windows DLL order (datasets -> pyarrow -> torch)")
    try:
        import datasets  # noqa: F401
        import pyarrow  # noqa: F401
        import torch  # noqa: F401
        report(PASS, "import datasets, pyarrow, torch in that order")
    except Exception as exc:
        report(FAIL, "import datasets, pyarrow, torch in that order", repr(exc))


def check_gpu():
    section("GPU / CUDA")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        report(FAIL, "torch import", repr(exc))
        return

    if not torch.cuda.is_available():
        report(FAIL, "torch.cuda.is_available()", "CUDA not available")
        return

    report(PASS, "torch.cuda.is_available()")
    count = torch.cuda.device_count()
    report(PASS, "CUDA device count", str(count))
    for i in range(count):
        cap = torch.cuda.get_device_capability(i)
        props = torch.cuda.get_device_properties(i)
        report(PASS, f"device {i}", f"{props.name}, sm_{cap[0]}{cap[1]}, {props.total_memory / 1024**3:.1f} GiB")

    arch = torch.cuda.get_arch_list()
    print(f"torch built for archs: {', '.join(arch)}")
    # RTX 50xx (Blackwell) is sm_120; wheels must include it or kernels fall back/JIT-fail.
    if any(a in ("sm_120", "compute_120") for a in arch) or count == 0:
        report(PASS, "sm_120 (Blackwell) kernels present")
    else:
        report(FAIL, "sm_120 (Blackwell) kernels present", f"arch list: {arch}")

    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        c = a @ b
        torch.cuda.synchronize()
        report(PASS, "fp16 1024x1024 matmul on GPU", f"checksum {c.float().abs().mean().item():.4f}")

        import torch.nn.functional as F
        q = torch.randn(2, 8, 32, 32, device="cuda", dtype=torch.float16)
        F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize()
        report(PASS, "scaled_dot_product_attention on GPU (MiniMind attention path)")
    except Exception as exc:
        report(FAIL, "GPU compute smoke test", repr(exc))

    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")


def check_tooling():
    section("Tooling")
    scripts = Path(sys.executable).parent
    for exe in ("torchrun.exe", "python.exe", "pip.exe"):
        p = scripts / exe
        report(PASS if p.exists() else FAIL, f"venv script {exe}", str(p))
    try:
        enc = sys.stdout.encoding
        report(PASS, "stdout encoding", enc or "unknown")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 2) Per-repo import + forward smoke tests (run in a child process, cwd=repo)
# ---------------------------------------------------------------------------
REPO_SMOKE = r'''
import json, os, sys, tempfile, warnings
warnings.filterwarnings("ignore")

repo = os.getcwd()
sys.path.insert(0, repo)
import datasets  # noqa: F401  (Windows DLL order, same as the repo's train scripts)
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

which = sys.argv[1]

config = MiniMindConfig(hidden_size=64, num_hidden_layers=1, vocab_size=6400,
                        num_attention_heads=8, num_key_value_heads=4,
                        max_position_embeddings=128)
model = MiniMindForCausalLM(config).eval()
ids = torch.randint(0, 6400, (2, 16))
with torch.no_grad():
    out = model(ids)
assert out.logits.shape == (2, 16, 6400), out.logits.shape
print(f"  model forward OK on CPU: logits {tuple(out.logits.shape)}")

if torch.cuda.is_available():
    gpu_model = MiniMindForCausalLM(config).eval().half().cuda()
    with torch.no_grad():
        out = gpu_model(ids.cuda())
    torch.cuda.synchronize()
    print(f"  model forward OK on GPU: logits {tuple(out.logits.shape)}")
    del gpu_model
    torch.cuda.empty_cache()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(repo, "model"))
text = tok.decode(tok("你好，MiniMind").input_ids)
print(f"  tokenizer OK: {text!r}")

from trainer.trainer_utils import get_model_params, setup_seed  # noqa: F401
print("  trainer.trainer_utils import OK")

if which == "minimind":
    from model.model_lora import LoRA, apply_lora, save_lora, merge_lora  # noqa: F401
    from dataset.lm_dataset import PretrainDataset
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "MiniMind 是一个从零训练的小语言模型。"}, ensure_ascii=False) + "\n")
        tmp = fh.name
    ds = PretrainDataset(tmp, tok, max_length=64)
    input_ids, labels = ds[0]
    assert input_ids.shape == (64,) and labels.shape == (64,)
    print(f"  dataset.PretrainDataset OK: input_ids {tuple(input_ids.shape)}, labels {tuple(labels.shape)}")
    os.unlink(tmp)
else:
    from model.model_vlm import VLMConfig, MiniMindVLM  # noqa: F401
    vcfg = VLMConfig(hidden_size=64, num_hidden_layers=1, vocab_size=6400)
    print(f"  model_vlm OK: VLMConfig(hidden_size={vcfg.hidden_size})")
    from dataset.lm_dataset import VLMDataset  # noqa: F401
    print("  dataset.VLMDataset import OK")
    vision = os.path.join(repo, "model", "siglip2-base-p32-256-ve")
    if os.path.isdir(vision):
        vlm = MiniMindVLM(vcfg, vision_model_path=vision).eval()
        from PIL import Image
        img = Image.new("RGB", (256, 256), (128, 128, 128))
        tensor = vlm.processor(images=img, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            feat = vlm.vision_encoder(pixel_values=tensor.half().cuda() if torch.cuda.is_available() else tensor.half())
        print(f"  vision encoder OK: {tuple(feat.last_hidden_state.shape)}")
    else:
        print(f"  vision encoder SKIP (missing {vision})")
print("SMOKE_OK")
'''


def run_repo_smoke():
    section("Repository smoke tests")
    for name, path in REPOS.items():
        if not path.is_dir():
            report(FAIL, f"{name} repo found", str(path))
            continue
        print(f"\n--- {name} ({path}) ---")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(REPO_SMOKE)
            script = fh.name
        try:
            env = dict(os.environ)
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", script, name],
                cwd=str(path), env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=900,
            )
            for line in (proc.stdout or "").splitlines():
                print(f"  {line}")
            if proc.returncode == 0 and "SMOKE_OK" in (proc.stdout or ""):
                report(PASS, f"{name}: import + tiny forward pass")
            else:
                if proc.stderr:
                    print("  --- stderr (tail) ---")
                    for line in proc.stderr.strip().splitlines()[-25:]:
                        print(f"  {line}")
                report(FAIL, f"{name}: import + tiny forward pass", f"exit {proc.returncode}")
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass


def check_torchrun():
    """torchrun on Windows needs the libuv patch applied to the venv torch."""
    section("torchrun rendezvous")
    if sys.platform != "win32":
        report(SKIP, "torchrun smoke", "not Windows")
        return
    torchrun = Path(sys.executable).parent / "torchrun.exe"
    if not torchrun.exists():
        report(FAIL, "torchrun.exe present")
        return
    probe = Path(tempfile.gettempdir()) / "_minimind_torchrun_probe.py"
    probe.write_text("import os\nprint('TORCHRUN_PROBE_OK', os.environ.get('RANK'))\n", encoding="utf-8")
    env = dict(os.environ)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "USE_LIBUV": "0"})
    try:
        proc = subprocess.run(
            [str(torchrun), "--nproc_per_node", "1", str(probe)],
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        ok = proc.returncode == 0 and "TORCHRUN_PROBE_OK" in ((proc.stdout or "") + (proc.stderr or ""))
        detail = "" if ok else "run fix_torchrun_windows.ps1 (libuv patch missing after torch upgrade?)"
        report(PASS if ok else FAIL, "torchrun --nproc_per_node 1", detail)
    except Exception as exc:
        report(FAIL, "torchrun --nproc_per_node 1", repr(exc))
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-gpu", action="store_true", help="skip CUDA checks")
    ap.add_argument("--skip-repos", action="store_true", help="skip per-repo smoke tests")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("MiniMind shared environment self-check")
    print(f"environment root: {ROOT}")
    check_runtime()
    check_windows_dll_order()
    check_packages()
    if not args.skip_gpu:
        check_gpu()
    check_tooling()
    check_torchrun()
    if not args.skip_repos:
        run_repo_smoke()

    print("\n" + "=" * 60)
    if _failures:
        print(f"RESULT: {len(_failures)} check(s) FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
