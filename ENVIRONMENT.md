# MiniMind 双仓库共享环境（uv）

> 适用于 `D:\minimind\minimind` 与 `D:\minimind\minimind-v` 两个仓库的**同一个** Python 虚拟环境。
> 本机：Windows 10.0.26200 / RTX 5070 Ti 16GB / NVIDIA 驱动 591.86 / uv 0.10.6。
> 最近验证：2026-10-20，`check_env.py` 全部通过。
>
> 📚 配套学习路径见 [LEARNING_PATH.md](LEARNING_PATH.md)（从零手写模型 → 预训练 → SFT → 后训练 → VLM）。

---

## 一、结论速览

| 项目 | 结果 |
|---|---|
| 是否需要安装/升级显卡驱动 | **不需要**。驱动 591.86（CUDA 13.1）已支持 RTX 5070 Ti（Blackwell, `sm_120`） |
| 是否需要安装 CUDA Toolkit | **不需要**。PyTorch wheel 自带 CUDA 12.8 运行时 |
| 环境位置 | `D:\minimind\.venv`（Python 3.11.14，由 uv 管理） |
| 核心版本 | `torch 2.9.1+cu128` / `transformers 4.57.6` / `datasets 3.6.0` / `numpy 1.26.4` |
| 显存识别 | 15.9 GiB，`torch.cuda.is_available() == True`，torch 编译包含 `sm_120` |
| 两个仓库的代码 | 均可 import 并完成前向/反向；真实小训练、CLI 入口、真实数据集均已跑通 |
| 唯一额外修复 | Windows 版 torch 缺 libuv，已用 `fix_torchrun_windows.ps1` 修复 `torchrun` |

安装体积参考：`.venv` 约 **5.3 GB**，uv 缓存约 **5.5 GB**（在 `C:\Users\skyland\AppData\Local\uv\cache`，可用 `uv cache clean` 清理，下次安装会重新下载）。

---

## 二、目录结构

```text
D:\minimind\
├── .venv\                      # ★ 共享虚拟环境（两个仓库都用它）
├── .python-version             # 3.11
├── pyproject.toml              # ★ 环境定义：依赖清单 + 国内镜像配置
├── uv.lock                     # ★ 锁定文件（165 个包，可精确复现）
├── setup_env.ps1               # 一键创建/修复/验证环境
├── activate_env.ps1            # 在当前 PowerShell 窗口激活环境
├── run.ps1                     # 便捷运行器（自动选 cwd + 解释器）
├── check_env.py                # 环境自检（GPU / 版本 / 两仓库冒烟）
├── fix_torchrun_windows.ps1    # Windows torchrun 修复（幂等，可 -Revert）
├── link_datasets.ps1           # 把已下载的硬链接到两个仓库的 dataset/ 目录
├── LEARNING_PATH.md            # ★ 从零学习路径（配合本环境使用）
├── learning\                   # 学习用小数据子集与生成脚本
│   ├── make_subsets.py         # 从全量数据切 5k/2k/500 条快速实验子集
│   └── data\                   # 已生成的小子集（约 18MB）
├── ENVIRONMENT.md              # 本文件
├── minimind\                   # 仓库 1（语言模型）
├── minimind-v\                 # 仓库 2（多模态 VLM）
├── minimind_dataset\           # 已下载数据集（.jsonl）
└── minimind-v_dataset\         # 已下载数据集（.parquet）
```

---

## 三、日常使用

### 1. 激活环境（每个新开的 PowerShell 窗口执行一次）

```powershell
. D:\minimind\activate_env.ps1
```

激活后 `python` / `pip` / `torchrun` / `modelscope` 都指向共享环境，并自动设置
`PYTHONUTF8=1`（避免仓库打印中文时乱码）。

> 也可以不激活，直接用绝对路径：`D:\minimind\.venv\Scripts\python.exe`。

### 2. 跑 minimind（语言模型）

```powershell
cd D:\minimind\minimind\trainer
python train_pretrain.py                 # 单卡直接 python 即可
python train_full_sft.py
python train_lora.py
python train_dpo.py

cd D:\minimind\minimind
python eval_llm.py --load_from .\minimind-3      # 需要先下载 transformers 格式权重
cd scripts
streamlit run web_demo.py                         # 需要模型目录在 scripts\ 下
```

### 3. 跑 minimind-v（多模态）

```powershell
cd D:\minimind\minimind-v\trainer
python train_pretrain_vlm.py
python train_sft_vlm.py

cd D:\minimind\minimind-v
python eval_vlm.py --load_from .\minimind-3v
cd scripts
python web_demo_vlm.py                             # Gradio
```

### 4. 用 `run.ps1` 免激活运行（自动处理工作目录）

```powershell
# trainer 脚本 -> cwd 自动设为 <repo>\trainer（仓库默认参数用 ../dataset、../out）
.\run.ps1 minimind trainer\train_pretrain.py --from_resume 1

# eval/其它脚本 -> cwd 自动设为仓库根目录
.\run.ps1 minimind eval_llm.py --load_from .\minimind-3

# 需要多卡时（见“已知限制”，本仓库在 Windows 上不可用）
.\run.ps1 minimind trainer\train_pretrain.py --epochs 2
```

### 5. 用 `uv run`（另一种方式，uv 会自动发现上层的 pyproject.toml）

```powershell
cd D:\minimind\minimind\trainer
uv run --no-sync python train_pretrain.py
```

---

## 四、环境规格

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.11.14 | uv 托管（`uv python install 3.11`）；不用系统 3.14 |
| torch | **2.9.1+cu128** | 必须 cu128+，RTX 50 系 `sm_120` 内核 |
| torchvision | 未安装 | 两仓库代码均未使用（requirements 里也是注释） |
| transformers | 4.57.6 | 两仓库一致 |
| tokenizers | 0.22.2 | 由 transformers 约束 |
| datasets | 3.6.0 | 两仓库一致 |
| pyarrow | 23.0.0 | 两仓库一致 |
| numpy | 1.26.4 | 两仓库一致 |
| scipy | 1.17.1 | 自动解析（兼容 numpy 1.26） |
| pandas | 2.3.3 | 自动解析 |
| pillow | 11.3.0 | gradio 要求 `<12` |
| pydantic | 2.11.5 | 取两仓库中较新 |
| gradio / streamlit | 5.49.1 / 1.50.0 | VLM WebUI / LM WebUI |
| fastapi / uvicorn | 0.120.1 / 0.38.0 | OpenAI 兼容 API |
| wandb / swanlab | 0.22.3 / 0.9.8 | 训练可视化（脚本默认 swanlab） |
| modelscope | 1.37.0 | 权重/数据集下载（CLI：`modelscope`） |
| pip | 26.2.1 | uv venv 默认不带 pip，已显式加入便于排查 |

完整列表见 `uv.lock`。

---

## 五、与两份 requirements.txt 的差异（为什么不是完全照搬）

两个仓库的 requirements 有冲突，且部分旧 pin 在 Python 3.11 / Windows 上装不上。合并规则：**能装且兼容的一律保持原版本；冲突取较新；只有装不上/不兼容时才上移**。

| 包 | minimind | minimind-v | 本环境 | 原因 |
|---|---|---|---|---|
| torch | `#2.6.0`（注释） | `#2.6.0`（注释） | **2.9.1+cu128** | 2.6 无 `sm_120`，RTX 5070 Ti 无法跑 GPU 内核 |
| pydantic | 2.11.5 | 2.8.2 | 2.11.5 | 两仓库冲突，取新 |
| wandb | 0.22.3 | 0.18.3 | 0.22.3 | 同上 |
| swanlab | 0.9.8 | 0.6.12 | 0.9.8 | 同上 |
| matplotlib | `#3.10.0` | 3.5.1 | 3.10.9 | 3.5.1 无 cp311 wheel；仓库仅在个别脚本用到 |
| peft | `#0.7.1` | 0.7.1 | 0.19.1 | 0.7.1 与 transformers 4.57 不兼容；仓库代码未直接 `import peft` |
| ujson | 5.1.0 | 5.1.0 | 6.0.0 | 5.1.0 无 cp311 wheel，需本地编译 |
| pip | — | — | 26.2.1 | 新增：uv venv 默认无 pip，方便 `pip list` 排查 |

其余 `transformers==4.57.6`、`datasets==3.6.0`、`numpy==1.26.4`、`trl==0.13.0`、`sentence_transformers==2.3.1`、`scikit_learn==1.5.1` 等均保持原 pin。所有包都做了 import 测试（含 `trl`、`peft`、`sentence_transformers`、`swanlab`、`modelscope`），全部可导入。

torchvision 未安装：两个仓库的代码没有 import 它（requirements 里也是注释状态）。

---

## 六、数据与模型资源

### 1. 数据集（已就位）

`link_datasets.ps1` 已把已下载的数据集以**硬链接**方式放进仓库要求的目录（不额外占磁盘，删除链接不影响原文件）：

```text
D:\minimind\minimind_dataset\*.jsonl   ->  D:\minimind\minimind\dataset\*.jsonl      (11 个)
D:\minimind\minimind-v_dataset\*.parquet ->  D:\minimind\minimind-v\dataset\*.parquet (2 个)
```

如需重新执行 / 撤销：

```powershell
powershell -ExecutionPolicy Bypass -File D:\minimind\link_datasets.ps1
# 撤销：直接删除上面两个 dataset 目录里新出现的 *.jsonl / *.parquet 即可
```

> 注意：这些数据文件会以「未跟踪文件」出现在两个 git 仓库的 `git status` 里。
> 这是正常的（数据本来就不该进 git）；如果不想看到，可在各自仓库的 `.gitignore` 追加 `dataset/*.jsonl` / `dataset/*.parquet`。

数据集内容已验证：`PretrainDataset`/`SFTDataset` 能正确解析真实 jsonl，`pretrain_i2t.parquet` 为 `conversations(string) + image_bytes(binary)`、1,274,698 行。

### 2. 还缺的模型权重（需要时自行下载，约几百 MB ~ 数 GB）

这些是**模型资源**而非环境依赖，`check_env.py` 会显示 SKIP：

```powershell
. D:\minimind\activate_env.ps1

# minimind-v 的视觉编码器（VLM 训练/推理必需）
modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir D:\minimind\minimind-v\model\siglip2-base-p32-256-ve

# minimind-v 训练用的基座语言模型权重
modelscope download --model gongjy/minimind-3v-pytorch llm_768.pth --local_dir D:\minimind\minimind-v\out

# 推理用：minimind 的 transformers 格式权重
modelscope download --model gongjy/minimind-3 --local_dir D:\minimind\minimind\minimind-3

# 推理用：minimind-v 的 transformers 格式权重
modelscope download --model gongjy/minimind-3v --local_dir D:\minimind\minimind-v\minimind-3v
```

---

## 七、Windows 专项修复：torchrun 的 libuv 问题

**现象**：`torchrun --nproc_per_node 1 xxx.py` 直接报

```text
torch.distributed.DistStoreError: use_libuv was requested but PyTorch was built without libuv support
```

**原因**：Windows 版 PyTorch wheel 未编译 libuv，但 `torch.distributed.elastic` 的几处
`TCPStore(...)` 使用默认 `use_libuv=True`。上游到 torch 2.11 仍未修复（PR #175316 / #196855 未合入），
且 `USE_LIBUV=0` 环境变量只被其中一部分调用点识别。

**修复**：`fix_torchrun_windows.ps1` 向 venv 内 torch 的 4 处 `TCPStore` 调用补上
`use_libuv=False` / `use_libuv=use_libuv`（幂等，可重复执行）：

```powershell
powershell -ExecutionPolicy Bypass -File D:\minimind\fix_torchrun_windows.ps1
# 还原：
powershell -ExecutionPolicy Bypass -File D:\minimind\fix_torchrun_windows.ps1 -Revert
```

`setup_env.ps1` 在每次 `uv sync` 后都会自动重新应用（torch 升级/重装后补丁会丢失，脚本会补回来）。
修复后已验证：

- `torchrun --nproc_per_node 1` 正常拉起 worker
- 两个仓库的 `eval_*.py --help`、`train_*.py --help` 全部正常

---

## 八、已知限制与注意事项

1. **多卡 DDP 在 Windows 上不可用（仓库代码限制，不是环境问题）**
   两个仓库的 `trainer/trainer_utils.py` 都写死了 `dist.init_process_group(backend="nccl")`，
   Windows 没有 NCCL。且 Windows 的 gloo 后端受本机 `hosts`（Docker 的 `kubernetes.docker.internal`）
   与 IPv6 地址影响会报 `makeDeviceForHostname(): unsupported gloo device`。
   **本机只有 1 张 GPU，直接用 `python train_*.py` 即可**，`torchrun` 仅在需要多卡时才有意义。
   如果以后要在 Windows 多卡，需要把仓库里 `nccl` 改成 `gloo` 并设置 `GLOO_SOCKET_IFNAME`。

2. **数据加载顺序**：仓库训练脚本在 `import torch` 之前先 `import datasets`（注释写明是规避
   Windows pyarrow/torch DLL 冲突）。自己写新脚本时请保持同样的顺序。

3. **显存**：默认配置（hidden_size 768、batch_size 32、seq_len 340、accumulation 8）在 16GB 卡上
   一般可行，但若 OOM 请降 `--batch_size` 或 `--max_seq_len`。显卡当前被其它程序占用约 2.6GB，
   训练前可用 `nvidia-smi` 确认。

4. **中文输出乱码**：`activate_env.ps1` / `run.ps1` 已设置 `PYTHONUTF8=1`。若手工执行，请用
   `python -X utf8` 或先 `chcp 65001`。

5. **uv 硬链接警告**：uv 缓存在 C:，工程在 D:，跨卷无法硬链接，会提示
   “Failed to hardlink files; falling back to full copy”，只是提示，不影响使用。

6. **镜像配置**：`pyproject.toml` 中默认 PyPI 使用阿里云镜像，torch 使用上海交大
   `pytorch-wheels/cu128` 镜像（官方源本机只有约 0.9 MB/s）。若在海外/有代理，可改成官方源：

   ```toml
   [[tool.uv.index]]
   name = "pytorch-cu128"
   url = "https://download.pytorch.org/whl/cu128"
   explicit = true
   ```

7. **`minimind-v` 的 WebUI 与训练**需要先下载第六节的 SigLIP2 视觉编码器与 `llm_768.pth`，
   否则 VLM 构造阶段会因缺少视觉权重失败（环境本身没有问题）。

---

## 九、验证记录（2026-10-20）

运行 `D:\minimind\.venv\Scripts\python.exe D:\minimind\check_env.py`，结果：

```text
[PASS] Python 3.11.x
[PASS] import datasets, pyarrow, torch in that order
[PASS] torch 2.9.1+cu128 / transformers 4.57.6 / numpy 1.26.4 / datasets 3.6.0 ...
[PASS] torch.cuda.is_available()
[PASS] device 0 — NVIDIA GeForce RTX 5070 Ti, sm_120, 15.9 GiB
torch built for archs: sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
[PASS] sm_120 (Blackwell) kernels present
[PASS] fp16 1024x1024 matmul on GPU
[PASS] scaled_dot_product_attention on GPU (MiniMind attention path)
[PASS] torchrun --nproc_per_node 1
--- minimind ---   模型 CPU/GPU 前向 (2,16,6400)、tokenizer、trainer_utils、PretrainDataset 全部 OK
--- minimind-v --- 模型 CPU/GPU 前向、tokenizer、VLMConfig、VLMDataset 全部 OK
RESULT: all checks passed
```

额外人工验证：

| 验证项 | 结果 |
|---|---|
| `python train_pretrain.py`（小模型、8 条样本、1 epoch） | 完成前向+反向+保存 `out/*.pth` 与 `checkpoints/*_resume.pth` |
| `torchrun --nproc_per_node 1 train_pretrain.py` | 能拉起 worker（随后因仓库写死 `nccl` 而退出，属预期） |
| `eval_llm.py --help`、`eval_vlm.py --help`、`train_pretrain.py --help`、`train_lora.py --help`、`train_pretrain_vlm.py --help`、`train_sft_vlm.py --help` | 全部 exit 0 |
| 真实数据：`PretrainDataset` / `SFTDataset` 读取 `minimind_dataset/*.jsonl` | 正确返回 `(input_ids, labels)` |
| `pyarrow` 读取 `pretrain_i2t.parquet` | schema `conversations:string, image_bytes:binary`，1,274,698 行 |
| `modelscope` / `pip` / `torchrun` CLI | 均在 `.venv\Scripts` 下 |

---

## 十、维护

```powershell
# 重新创建/修复环境（幂等；会重装缺失包、重新打 torchrun 补丁、跑自检）
powershell -ExecutionPolicy Bypass -File D:\minimind\setup_env.ps1

# 新增依赖（推荐 uv add，会同步 pyproject.toml 与 uv.lock）
cd D:\minimind
uv add some-package==1.2.3

# 按 lock 精确同步（不重新解析）
uv sync --frozen

# 自检
.\.venv\Scripts\python.exe check_env.py

# 清理 uv 下载缓存（约 5.5GB，可随时重建）
uv cache clean

# 完全重建
Remove-Item D:\minimind\.venv -Recurse -Force
powershell -ExecutionPolicy Bypass -File D:\minimind\setup_env.ps1
```

升级 torch 时的注意点：

- 必须选 `+cu128`（或更新的 cu13x）CUDA 构建，否则 RTX 50 系会报 “no kernel image is available”；
- 升级后重跑 `fix_torchrun_windows.ps1`（`setup_env.ps1` 已包含）；
- 若 `transformers` 也要升级，建议先跑 `check_env.py` 确认两仓库模型仍可 import。
