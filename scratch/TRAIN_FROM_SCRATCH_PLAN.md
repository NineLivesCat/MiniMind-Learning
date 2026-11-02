# 从零训练到复现官方效果 · 完整执行计划

> 目标模型：`minimind-3` 64M dense（768 隐藏 / 8 层 / 8 头 / 4 KV 头），词表 6400。
> 数据：本地已下载完整数据集（无需再下载）：
> - `D:\minimind\minimind\dataset\pretrain_t2t_mini.jsonl`（1.24GB）
> - `D:\minimind\minimind\dataset\sft_t2t_mini.jsonl`（1.74GB）
> - `D:\minimind\minimind\dataset\dpo.jsonl`、`lora_medical.jsonl`、`rlaif.jsonl`
> - VLM：`D:\minimind\minimind-v\dataset\pretrain_i2t.parquet`（4.3GB）、`sft_i2t.parquet`（4.9GB）
>
> 说明：本计划同时给两组时间——「官方 README 的 3090 参考值」和「你在 RTX 5070 Ti 上用 mymm 代码实测值」。
> 你的实现与官方源码在数值上对齐（test_align 8/8），但逐 bit 相同不可能；目标是**指标/效果级复现**。

## 1. 数据集参考（本地实测统计）

| 数据集 | 条数 | 平均 tokens/条 | 文件大小 | 用途 |
|---|---:|---:|---:|---|
| `pretrain_t2t_mini` | 1,270,238 | 266.5 | 1.24 GB | 快速预训练（Zero 路线） |
| `sft_t2t_mini` | 905,718 | 492.9 | 1.74 GB | 快速 SFT（Zero 路线） |
| `pretrain_t2t`（完整） | 8,468,827 | 266.0 | 8.28 GB | 主线预训练（可选，更接近发布权重） |
| `sft_t2t`（完整） | 5,109,432 | 886.9 | 14.1 GB | 主线 SFT（可选） |
| `dpo.jsonl` | 17,166 对 | 932.9（chosen+rejected） | 53.7 MB | DPO |
| `lora_medical.jsonl` | 25,276 | 375.5 | 34 MB | LoRA 领域实验 |
| `lora_identity.jsonl` | 91 | 59.3 | 0.0 MB | 自我认知 LoRA |
| `rlaif.jsonl` | 19,502 | 326.8 | 24 MB | PPO/GRPO（选做） |
| `pretrain_i2t.parquet` | 1,274,698 | — | 4.3 GB | VLM 投影对齐 |
| `sft_i2t.parquet` | 2,904,511 | — | 4.9 GB | VLM SFT（已含 pretrain 子集） |

## 2. 官方训练超参（直接取自官方 trainer 默认值）

| 阶段 | epochs | batch | accum | 有效 batch | lr | max_seq_len | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| Pretrain | 2 | 32 | 8 | 256 | 5e-4 | 340 | AdamW + cosine（1.0→0.1） |
| Full SFT | 2 | 16 | 1 | 16 | 1e-5 | 768 | 从 pretrain 权重继续 |
| LoRA | 10 | 32 | 1 | 32 | 1e-4 | 340 | rank 16，base=full_sft |
| DPO | 1 | 4 | 1 | 4 | 4e-8 | 1024 | beta 0.15，base=full_sft |
| Distill（选做） | 6 | 32 | 1 | 32 | 5e-6 | 340 | alpha 0.5，T 1.5 |
| PPO（选做） | 1 | 2 | 1 | 2 | 3e-7（critic 5e-7） | 768+gen1024 | 需 reward model |
| GRPO（选做） | 1 | 2 | 1 | 2 | 3e-7 | 768+gen1024 | gen=6，需 reward model |
| VLM Pretrain | 2 | 16 | 1 | 16 | 4e-4 | 450 | freeze_llm=2（只训 projector） |
| VLM SFT | 2 | 4 | 1 | 4 | 5e-6 | 768 | freeze_llm=1（首尾层+projector） |

官方 README 参考耗时（单卡 3090）：
- Pretrain mini：1 epoch ≈ **1.21h**，2 epochs ≈ 2.42h；
- SFT mini：1 epoch ≈ **1.10h**，2 epochs ≈ 2.20h；
- Zero（2+2 epochs）合计 ≈ **4.6h**；
- VLM SFT：1 epoch ≈ 2h；VLM Pretrain 数据量约为 SFT 的 45%，1 epoch ≈ 0.9h。

## 3. 逐步展开：步数 / token / 预计耗时

### 3.1 预训练 mini（768/8，bs32×accum8，seq340）

- micro steps/epoch = ⌊1,270,238 / 32⌋ = **39,694**
- 参数更新/epoch = 39,694 / 8 ≈ **4,962**
- 每 epoch 计算 token（padding 后）= 39,694 × 32 × 340 = **431.9M**
- 2 epochs：**79,388 micro steps / 9,924 次更新 / 863.7M tokens**
- 实测（你的 5070 Ti，mymm 代码，bs32）：0.125 s/step，87.2k tok/s，峰值 6.0GB
  → **1.38h/epoch，2 epochs ≈ 2.75h**（含 eval/保存按 ~3.0h 预算）
- token 预算：864M / 63.9M 参数 = **13.5 tokens/param**（低于 Chinchilla 的 20，属于“能出对话效果”的轻量路线）

### 3.2 SFT mini（768/8，bs16，seq768）

- steps/epoch = ⌊905,718 / 16⌋ = **56,607**
- token/epoch（padding 后）= 56,607 × 16 × 768 = **695.6M**
- 2 epochs：**113,214 steps / 1,391M tokens**
- 实测：0.157 s/step，78.5k tok/s，峰值 7.0GB → **2.46h/epoch，2 epochs ≈ 4.92h**（按 5.5h 预算）

### 3.3 Zero 总预算

| 项 | 3090 官方参考 | 你的 5070 Ti + mymm |
|---|---:|---:|
| Pretrain 2 epochs | 2.42h | **2.75h** |
| SFT 2 epochs | 2.20h | **4.92h** |
| 合计（纯计算） | 4.62h | **7.7h** |
| 加 15% 评估/保存/重启 | ≈5.3h | **≈9h** |

### 3.4 可选后训练

| 阶段 | 规模 | 步数 | 实测/估算耗时 |
|---|---|---|---|
| DPO 1 epoch | 17,166 对，bs4，seq1024 | 4,291 | **0.5~1h**（双模型前向） |
| LoRA 10 epochs | 25,276 条，bs32，seq340 | 7,890 | **20~40min** |
| Distill 1 epoch | 905,718 条，bs32，seq340 | 28,303 | **≈2.5h/epoch**（6 epochs ≈15h，建议先跑 1 epoch） |
| GRPO 1 epoch（选做） | 19,502 prompt，bs2×gen6 | 9,751 | 16GB 显存紧张，建议先 200~500 步（小时级） |

### 3.5 VLM（可选，768/8）

| 阶段 | 数据条数 | batch | steps/epoch | 实测耗时/epoch | 官方参考 |
|---|---:|---:|---:|---:|---:|
| Pretrain VLM（freeze 2） | 1,274,698 | 16 | 79,668 | ≈1.5~2h | ≈0.9h |
| SFT VLM（freeze 1） | 2,904,511 | 16 | 181,532 | ≈7~8h | ≈2h |
| 2+2 epochs 合计 | — | — | — | **≈17~20h** | ≈5.8h |

> VLM SFT 数据已包含 pretrain 子集，官方推荐**可跳过 pretrain 直接 SFT**；只做 SFT 2 epochs ≈ **14~16h**。
> 想省时间：先用 50k 子集跑通 2 epochs（≈15min），验证指标后再全量。

## 4. 参考锚点（本地官方权重实测）

| 指标 | 数值 | 测法 |
|---|---:|---|
| 参数量 | **63,912,192** | `test_model` / `sum(p.numel())` |
| 官方 768 权重 PPL | **66.28** | `mymm.eval`：100 段 `pretrain_5000`，window 512 / stride 256 |
| 官方权重在 pretrain 子集上的 loss | **2.44** | 768 模型，seq340，8 个 batch 平均 |
| 官方权重在 sft_mini 子集上的 loss | **0.89** | 同上，seq768（说明发布权重在训练分布上拟合很深） |
| 随机初始化 loss | ≈8.76（ln6400） | 起点参考 |

> 注意：`llm_768.pth`、`full_sft_768.pth`、HF `minimind-3` 本地这三个文件权重**相同**（MD5 一致），是官方发布的主线基座。
> 你 2 epochs mini 的模型 loss/PPL **不会**降到 2.44/0.89 这么低，这是数据量与训练轮次差距，属正常；以“PPL 尽量接近 66、对话可用”为目标。
> 若想更接近发布权重：改用完整数据（pretrain_t2t + sft_t2t）训练，见第 7 节。

## 5. 执行命令（mymm 手写实现，单卡 16GB）

```powershell
. D:\minimind\activate_env.ps1
cd D:\minimind\learning\scratch

# 0) 对齐自检（应 8/8 通过）
python mymm\tests\test_align.py

# 1) 预训练 2 epochs（后台跑，约 3h）
python -m mymm.train --data_path D:\minimind\minimind\dataset\pretrain_t2t_mini.jsonl `
  --save_dir out --save_weight pretrain --hidden_size 768 --num_hidden_layers 8 `
  --max_seq_len 340 --batch_size 32 --accumulation_steps 8 --epochs 2 --lr 5e-4 `
  --save_interval 1000 --log_interval 100 --num_workers 4

# 2) 评估 pretrain（对比 66.28）
python -m mymm.eval --weight_path out/pretrain_768.pth --hidden_size 768 --num_hidden_layers 8 `
  --num_texts 100 --max_length 512 --stride 256 --skip_gen 1

# 3) SFT 2 epochs（约 5h）
python -m mymm.train_sft --data_path D:\minimind\minimind\dataset\sft_t2t_mini.jsonl `
  --from_weight pretrain --save_weight full_sft --hidden_size 768 --num_hidden_layers 8 `
  --max_seq_len 768 --batch_size 16 --accumulation_steps 1 --epochs 2 --lr 1e-5 `
  --save_interval 1000 --log_interval 100 --num_workers 4

# 4) 对话自测（20 个固定问题）
python -m mymm.chat --hidden_size 768 --num_hidden_layers 8 --weight full_sft --auto 1

# 5) DPO（可选，约 0.5~1h）
python -m mymm.train_dpo --data_path D:\minimind\minimind\dataset\dpo.jsonl `
  --base_weight full_sft --save_weight dpo --hidden_size 768 --num_hidden_layers 8 `
  --max_seq_len 1024 --batch_size 4 --accumulation_steps 1 --epochs 1 --lr 4e-8 --beta 0.15
python -m mymm.dpo_metrics --hidden_size 768 --num_hidden_layers 8 --num_samples 100

# 6) LoRA（可选，约 30min）
python -m mymm.train_lora --data_path D:\minimind\minimind\dataset\lora_medical.jsonl `
  --base_weight full_sft --save_name lora_medical --hidden_size 768 --num_hidden_layers 8 `
  --max_seq_len 340 --batch_size 32 --epochs 10 --lr 1e-4 --rank 16
```

VLM（可选）：

```powershell
# VLM SFT 2 epochs（官方推荐直接做这一步，约 14~16h）
python -m mymm.train_vlm --stage sft --hidden_size 768 --num_hidden_layers 8 `
  --init_weight D:\minimind\minimind-v\out\llm_768.pth `
  --data_path D:\minimind\minimind-v\dataset\sft_i2t.parquet `
  --max_seq_len 768 --batch_size 16 --accumulation_steps 1 --epochs 2 --lr 5e-6 --freeze_llm 1

# 若想先做投影对齐（约 3~4h），再跑上面的 SFT 并指定 --init_weight out/pretrain_vlm_768.pth
python -m mymm.train_vlm --stage pretrain --hidden_size 768 --num_hidden_layers 8 `
  --init_weight D:\minimind\minimind-v\out\llm_768.pth `
  --data_path D:\minimind\minimind-v\dataset\pretrain_i2t.parquet `
  --max_seq_len 450 --batch_size 16 --accumulation_steps 1 --epochs 2 --lr 4e-4 --freeze_llm 2

# 评测（6 图 × 3 问）
python -m mymm.vlm_eval --hidden_size 768 --num_hidden_layers 8 --max_new_tokens 128
```

## 6. 每个阶段的验收门槛

| 阶段 | 必过项 | 参考数值 |
|---|---|---|
| 训练前 | `test_align` 8/8；参数量 63,912,192 | 已通过 |
| Pretrain | loss 曲线平滑下降；PPL 评测 | loss@500 < 7.5；2 epochs 后 PPL 尽量 < ~120（官方 66.28） |
| SFT | loss 明显下降；20 问对话通顺 | SFT loss < 3；自我介绍为 MiniMind；中文问答不胡言 |
| DPO | margin 由 ~0 变正 | `dpo_metrics` 前后对比 |
| LoRA | 医疗问答风格变化；参数量 ~0.8MB rank16 | 前后对话对比 |
| VLM | 6 张图描述基本正确 | 主观 1-5 分；官方参考同样 6 图 |

## 7. 想更接近发布权重（完整数据路线，约 2 天）

| 阶段 | 数据 | steps/epoch | 每 epoch 计算 token | 你的实测耗时/epoch |
|---|---|---:|---:|---:|
| Pretrain 完整 | 8,468,827 条 | 264,650 | 2.88B | **≈9.2h** |
| SFT 完整 | 5,109,432 条 | 319,339 | 3.93B | **≈13.9h** |

2 epochs 合计 ≈ **46h**；建议按 1 epoch 起跑（≈23h）后评估，再决定是否加第二轮。
token 预算：完整 2 epochs ≈5.8B tokens / 63.9M ≈ **90 tokens/param**，明显更接近发布权重。
数据顺序与随机种子差异会让曲线不同，但最终 PPL/对话质量应明显优于 mini 路线。

## 8. 断点、磁盘与监控

- 保存：每 `--save_interval 1000` 步覆盖写一次权重；每次同时写 `out/<name>_768_resume.pth`（≈620MB）。
- 续训：加 `--from_resume 1` 从上次中断处继续。
- 磁盘：mini 数据 3GB、完整数据 22GB；每个 768 ckpt 131MB、resume 620MB；VLM ckpt ≈130MB。
- 监控：`--log_interval 100` 打印 step/loss/lr；建议每 2~3k 步跑一次 `mymm.eval --skip_gen 1` 看 PPL。
- 显存：上述配置峰值 6~7GB（SFT 7GB），16GB 充裕；VLM SFT 峰值 ≈6~7GB。
- Windows 上 `--num_workers 4` 需在正常 PowerShell 里跑（沙箱内多进程会受限）。

## 9. 时间表（7 天版）

| 天 | 内容 | 耗时 |
|---|---|---|
| D1 | 对齐自检 + 50 步冒烟（pretrain/SFT/VLM 各一次）+ 数据确认 | 2h |
| D2 | Pretrain 2 epochs（后台） | 3h |
| D3 | 评测 PPL + SFT 2 epochs | 6h |
| D4 | 对话评测 + DPO + LoRA | 2h |
| D5-6 | VLM SFT 2 epochs（可选；可分两晚跑） | 14~16h |
| D7 | VLM 评测 + 汇总报告（PPL/速度/显存/对话） | 2h |

如果只有 1 天：pretrain 1 epoch（1.4h）+ SFT 1 epoch（2.5h）+ 评测（0.5h）≈ **5h**，可验证整条链路，质量低于 2+2 但足以展示。
