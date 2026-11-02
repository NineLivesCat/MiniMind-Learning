# MiniMind 命令行 exe 打包说明

把 `learning\scratch\out\save` 里训练好的权重，封装成一个普通用户双击/命令行即可使用的 Windows exe。
**无 GUI，纯命令行**，分为「文字对话」和「看图问答」两种用法。

## 1. 最终产物

```
packaging\dist\minimind\
  minimind.exe            ← 用户运行的入口
  _internal\              ← PyInstaller 运行时（torch/transformers 等）
  model\
    full_sft_768.pth      ← 文字对话模型（131MB）
    sft_vlm_768.pth       ← 看图模型（可选，134MB）
    tokenizer\            ← tokenizer.json / tokenizer_config.json
    siglip2\              ← 视觉编码器（可选，190MB）
  使用说明.txt
packaging\dist\MiniMind-CLI-Windows.zip   ← 发给别人的压缩包
```

> 采用 `--onedir`（文件夹模式）而不是 `--onefile`：
> onefile 每次启动要把几百 MB 解压到临时目录，启动慢、杀毒软件更容易误报；
> onedir 启动快，模型文件在外部也方便替换/更新。

## 2. 构建环境（只需一次）

当前 `.venv` 是 CUDA 版 torch，直接打包会让 exe 带上 2~3GB 的 CUDA 运行库。
普通用户大多没有 N 卡，推荐单独建一个 **CPU 构建环境**：

```powershell
# 1) 建一个只装 CPU torch 的构建环境（用 uv；uv 不在 PATH 时用 C:\Users\skyland\.local\bin\uv.exe）
D:\minimind\.venv\Scripts\python.exe -m venv D:\minimind\.build_env --without-pip
$env:UV_CACHE_DIR = "D:\minimind\.cache\uv"

# 2) torch 只装包本身（--no-deps 避免去 pytorch 源拉依赖时卡住）
uv pip install --python D:\minimind\.build_env\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.9.1

# 3) 其余依赖从 PyPI 装（含 torch 缺的 jinja2/sympy 等）
uv pip install --python D:\minimind\.build_env\Scripts\python.exe `
  jinja2 sympy networkx filelock fsspec typing_extensions

# 4) 打包工具 + 推理依赖
uv pip install --python D:\minimind\.build_env\Scripts\python.exe `
  "transformers==4.57.6" pillow pyinstaller
```

校验：

```powershell
D:\minimind\.build_env\Scripts\python.exe -X utf8 -c "import torch, transformers, PyInstaller; print(torch.__version__, torch.cuda.is_available())"
```

## 3. 构建

```powershell
cd D:\minimind\learning\scratch
powershell -ExecutionPolicy Bypass -File .\packaging\build_cpu.ps1
```

只要文字对话、不带看图模型（压缩包更小）：

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\build_cpu.ps1 -TextOnly
```

如果当前账户对工程目录没有写权限（例如在受限沙箱里构建），可把输出改到可写目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\build_cpu.ps1 `
  -OutRoot D:\minimind\.cache\minimind_dist -WorkRoot D:\minimind\.cache\pyi_build
```

脚本会：PyInstaller 打包 → 复制权重/tokenizer/视觉编码器 → 写使用说明 → 生成 zip。
第一次约 3~6 分钟，之后 1 分钟左右。本次已经按上面命令构建并验证过，产物在
`D:\minimind\.cache\minimind_dist\`（zip 约 596MB，解压后约 942MB）。

## 4. 构建后自测

```powershell
cd D:\minimind\learning\scratch\packaging\dist\minimind
.\minimind.exe "你好，介绍一下你自己"
.\minimind.exe --image D:\minimind\minimind-v\dataset\eval_images\image-01-golden-dog-balloons.jpg "描述这张图"
```

预期：先打印「正在加载…」，然后输出回答；CPU 下 768 模型约 **40~60 tok/s**，看图中等。

## 5. 用户怎么用

把 zip 解压到任意目录（路径尽量不含中文/空格），双击 `minimind.exe` 进入交互对话；
或在 cmd/PowerShell 里：

```bat
minimind.exe
minimind.exe "推荐一些中国的美食"
minimind.exe --image photo.jpg "这张图里有什么？"
```

首次运行 Windows 可能弹 SmartScreen，点「更多信息 → 仍要运行」。

## 6. 常见问题

- **提示找不到模型**：确认 `model\` 和 exe 在同一目录；或 `--model-dir` 指向模型目录。
- **中文乱码**：程序内部已设 UTF-8；若旧终端仍乱码，先执行 `chcp 65001`。
- **没有 NVIDIA 显卡**：默认 `--device auto` 会用 CPU，无需任何驱动；只是生成慢一些。
- **想用 GPU**：用 CUDA 版 torch 重新构建即可（体积约 2~3GB）：
  在 `.build_env` 里换成 `pip install torch==2.9.1`（默认 CUDA 轮子），其余步骤不变；
  有 N 卡的机器运行 `--device cuda`（auto 会自动选）。
- **杀毒误报**：PyInstaller 打包程序常见误报，可加入信任或用文件夹版分发；不要用 onefile。
- **更新模型**：直接替换 `model\*.pth`，exe 不用重新打包。

## 7. 体积参考（CPU 版）

| 内容 | 大小 |
|---|---|
| PyInstaller + CPU torch + transformers | 约 600~900MB（解压后） |
| full_sft_768.pth | 131MB |
| sft_vlm_768.pth + siglip2 | 134MB + 190MB |
| zip 压缩后 | 约 400~700MB（视是否含看图） |

> 如果还嫌大：文字版可只保留 `full_sft_768.pth` + tokenizer（`-TextOnly`），
> 或后续改用量化/ONNX/GGUF 方案（64M 模型可压到几十 MB，但需要额外适配）。
