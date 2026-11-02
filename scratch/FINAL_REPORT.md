# MiniMind 从零手写 · 完成总结

> 工作区：`D:\minimind\learning\scratch\mymm`
> 环境：`D:\minimind\.venv`（Python 3.11 / torch 2.9.1+cu128 / RTX 5070 Ti 16GB）
> 约定：所有 Python 文件 4 空格缩进；测试在 `mymm/tests/`，可单独运行。

## 0. 快速开始

```powershell
. D:\minimind\activate_env.ps1
cd D:\minimind\learning\scratch

# 全部测试（20 个文件，121 项）
Get-ChildItem mymm\tests\test_*.py | ForEach-Object { python $_.FullName }

# 常用脚本
python -m mymm.train_sft --show_samples 1        # SFT 数据检查
python -m mymm.train_sft                          # SFT 训练
python -m mymm.chat                               # 对话
python -m mymm.eval --weights pretrain full_sft   # PPL/速度/显存
python -m mymm.train_dpo                          # DPO
python -m mymm.train_distill --max_steps 50       # 蒸馏
python -m mymm.train_vlm --stage pretrain --max_steps 50   # VLM 阶段 1
python -m mymm.vlm_chat --weight sft_vlm          # 带图对话
python -m mymm.vlm_eval                           # VLM 评测报告
python -m mymm.moe_compare --run dense moe --steps 100      # MoE vs dense
```

## 1. 测试总览（当前全部通过）

| 阶段 | 测试文件 | 项数 | 覆盖 |
|---|---|---:|---|
| D1 采样 | `test_sampling.py` | 9 | 温度/top-k/top-p/停条件/generate |
| D2 基础算子 | `test_ops.py` | 13 | RMSNorm / RoPE / SwiGLU 等 |
| D3 attention | `test_attention.py` | 8 | GQA / KV cache / mask |
| D4 FFN+Block | `test_ffn_block.py` | 11 | SwiGLU / 残差 / 参数量 |
| D5 组装 LLM | `test_model.py` | 9 | CausalLM 输出 / 参数 63,912,192 |
| D6 权重对齐 | `test_align.py` | 8 | 与官方 768 权重逐层对照 |
| D7 预训练 | `test_data_train.py` | 5 | 数据集 / train_epoch / 保存 |
| D8 实验 | `test_experiments.py` | 4 | lr/bs/dtype/规模对照运行器 |
| D9 续训 | `test_resume.py` | 4 | resume / 中途保存 |
| D10 SFT | `test_sft.py` | 4 | mask / 与仓库 label 对齐 / 训练 |
| D11 评测+实验 | `test_chat.py` + `test_sft_experiments.py` | 8 | 对话脚本 / SFT lr 对照 |
| D12 阶段 6 | `test_eval.py` | 5 | 滑窗 PPL / 生成评测 / 报告 |
| D13 LoRA | `test_lora.py` | 7 | 初始化 / 挂载 / 合并 / 训练 / chat |
| D14 DPO | `test_dpo.py` | 5 | 数据 / dpo_loss / 训练 / margin |
| D15 蒸馏+RL | `test_distill.py` + `test_rl.py` | 7 | KL/T² / GAE / group adv / clip |
| D16 VLM | `test_vlm.py` | 7 | SigLIP2 / projector / 注图 / 两阶段训练 |
| D17 VLM 推理 | `test_vlm_chat.py` | 3 | 带图生成 / 评测报告 |
| D18 MoE | `test_moe.py` | 4 | aux loss / 参数量 / 对照实验 |

合计：**20 个测试文件 / 121 项全部通过**。

## 2. 代码地图（mymm/）

| 模块 | 作用 |
|---|---|
| `config.py` | MiniMindConfig（LLM/VLM/MoE 字段，自动推导 head_dim/intermediate） |
| `norm.py` `rope.py` `attention.py` `ffn.py` `block.py` `model.py` | 从零手写 LLM 全部组件 |
| `sampling.py` `chat.py` | 采样与对话推理 |
| `data.py` `train.py` `train_sft.py` | 预训练/SFT 数据与训练、resume |
| `experiments.py` `sft_experiments.py` | 对照实验运行器（CSV + 曲线） |
| `eval.py` | 滑窗 PPL / 生成评测 / 四列表报告 |
| `lora.py` `train_lora.py` `lora_merge.py` | LoRA + 合并 |
| `dpo.py` `train_dpo.py` `dpo_metrics.py` | DPO + margin 统计 |
| `distill.py` `train_distill.py` `rl.py` | 白盒蒸馏 + RL 公式函数 |
| `vlm.py` `vlm_data.py` `train_vlm.py` | VLM 模型 / 数据 / 两阶段训练 |
| `vlm_chat.py` `vlm_eval.py` | VLM 推理与评测 |
| `moe.py` `moe_compare.py` | MoE 层与 dense 对照 |

## 3. 主要产物

- 权重（`learning/scratch/out/`）：`pretrain_128.pth`、`pretrain_768.pth`、`full_sft_128.pth`、`full_sft_768.pth`、`full_sft_none_128.pth`、`lora_medical_tiny_128.pth`、`full_sft_lora_merged_128.pth`
- 实验：`out/exp/*.csv + curves.png`、`out/exp_sft/*.csv`
- 评估：`out/eval/report.md/csv`、`out/vlm_eval/report.md/csv`
- 对照：`out/moe/exp/*.csv + curves.png`
- 数据：`learning/data/{pretrain_5000, sft_2000, dpo_500, lora_medical_500, vlm_1000, vlm_sft_500}.*`
- 官方权重：`D:\minimind\minimind\out\full_sft_768.pth`、`D:\minimind\minimind-v\out\{llm_768, pretrain_vlm_768, sft_vlm_768}.pth`

## 4. 任务覆盖与遗留

已完成主线：**S2 ~ S7、S9、S10 框架**（S0/S1 已按要求跳过）。
选做/遗留（不影响主线，想补再补）：

- **S8-07 / S8-08**：GRPO/PPO 小步训练（需下载 InternLM2-1.8B reward，16GB 可跑 128 模型小 batch；S8 全部为选做）；
- **S4-07/08**：bf16/fp16 与更大规模对照（运行器已就绪，跑一下即可出数据）；
- **S5-08**：官方 768 权重对话对照（`mymm.chat --save_dir D:\minimind\minimind\out --hidden_size 768 --num_hidden_layers 8 --auto 1`）；
- **S6-05**：YaRN 长上下文 PPL 曲线（需给 config 加 rope scaling，属选做）；
- **S9-12/13**：Gradio 界面、冻结策略参数对照（冻结策略已有 `freeze_vlm` 三档，只差把数字填表）；
- **S10-05/07/08**：主实验、评测、报告（框架与模板已给，需真实跑实验并填写数字）。

## 5. G1~G10 状态（基于现有测试与产物）

| Gate | 状态 | 依据 |
|---|---|---|
| G1 环境/数据 | ✅ | tokenizer 6400、数据可加载 |
| G2 基础算子 | ✅ | test_ops 13/13 |
| G3 与官方对齐 | ✅ | test_align 8/8（逐层 + 端到端） |
| G4 预训练闭环 | ✅ | loss 曲线 + resume 测试 4/4 |
| G5 SFT | ✅ | mask 对齐 + SFT 训练测试 4/4 |
| G6 评估 | ✅ | PPL/速度/显存四列表 + 报告 |
| G7 LoRA+DPO | ✅ | 参数/merge/DPO margin 测试全过 |
| G8 蒸馏/RL（选做） | ✅（代码/公式） | distill/rl 测试全过；RL 训练选做 |
| G9 VLM | ✅ | 两阶段 + 推理 + 评测报告；官方 768 实测描述正确 |
| G10 项目 | 🟡 框架就绪 | MoE 对照可跑，需你完成主实验与报告 |

## 6. 建议的下一步

1. 选 S10 选题（推荐 MoE vs Dense，脚本现成），跑 `moe_compare --steps 300~500` 并填 `PROJECT_TEMPLATE.md`；
2. 把 `out/eval/report.csv`、`out/moe/exp/*.csv`、`out/vlm_eval/report.csv` 的数字填进报告；
3. 补 S8-07/08 或 S4-07/08 任选其一作为加分项；
4. 所有结果汇总成一份 500-1000 字报告 + 2-3 张图。
