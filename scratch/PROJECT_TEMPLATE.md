# 毕业项目模板（S10，5-7 天任选其一）

> 交付要求：**代码 + loss 曲线 + 评测表 + 500 字结论**（baseline / 变量 / 结果 / 失败原因）。
> 本文件是填空模板：把 `<>` 里的内容换成你的项目内容即可，报告可直接从第 4 节整理。

## 1. 选题（S10-01 一页纸）

| 项目 | 要改的东西 | 你需要准备的交付物 |
|---|---|---|
| 领域 LoRA | 造 200 条医疗/法律对话，训 LoRA | 前后对比 + 参数量 |
| 自我认知注入 | `lora_identity` 500 条，rank 对比 | 对话截图 + loss |
| **MoE vs Dense** | 4 expert、top-k 2，同参数预算对比 | 速度/显存/效果表 |
| 长上下文 | 开 YaRN，测 512→4096 PPL | PPL-长度曲线 |
| 小型 DPO 对齐 | 自造 200 对偏好数据 | chosen 概率提升曲线 |
| 图文 caption 任务 | VLM 子集微调 + 自拍图片推理 | 推理示例 |
| Tokenizer 研究 | 不同词表大小对压缩率/效果影响 | 对比报告 |

**我选的项目**：`<MoE vs Dense / 领域 LoRA / ...>`

**基线（不改什么）**：
- 模型：`<128 维 2 层 / 768 维 8 层>`
- 权重：`<out/full_sft_128.pth / pretrain_768.pth>`
- 数据：`<learning/data/xxx.jsonl，N 条>`
- 基线指标：`<loss/PPL/主观分 = ...>`

**改什么（只改一个变量）**：
- `<例如：FFN 换成 4 expert top-2 MoE，其余超参完全不变>`

**怎么算成功**：
- 主指标：`<PPL 降低 x% / 同预算下 loss 更低 / 显存或速度快 y%>`
- 对照：`<dense vs moe 同一份数据、同样步数、同样 lr>`

## 2. 数据与配置（S10-03 / S10-04）

**数据检查**：
- 路径：`learning/data/project_<name>.jsonl`
- 格式：`<pretrain {"text"} / sft {"conversations"} / dpo {"chosen","rejected"}>`
- 条数：`<>`；平均 token 长度：`<>`；重复条数：`<>`
- 用 `python -m mymm.train_sft --show_samples 1 --data_path ...` 或 Dataset 单条打印确认

**训练预算**：
| 项 | 值 |
|---|---|
| 显存估算 | 参数量 <x>M × 4B + 激活 ≈ <y>GB |
| batch / accum / seq | `<8 / 2 / 256>` |
| lr / epochs / steps | `<>` |
| 预计耗时 | `<>` min（先跑 50 步测每步耗时再乘） |

**50 步冒烟**：`python -m mymm.<train_script> ... --max_steps 50`，确认 loss 正常、显存不爆。

## 3. 实验记录（S10-05 / S10-06）

**主实验日志**（贴摘要）：
```
<step 100 loss 7.123 | step 200 loss 6.987 | ...>
```

**对照表**（至少 1 组）：
| 配置 | 参数量 | 训练步数 | 最终 loss | PPL | tok/s | 显存峰值 |
|---|---|---|---|---|---|---|
| 基线 dense | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |
| 改动 MoE | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |

MoE 对照直接用现成脚本：
```powershell
python -m mymm.moe_compare --run dense moe --steps 100 --batch_size 8 --aux_coef 0.01 --plot
```
产出：`out/moe/exp/dense.csv`、`out/moe/exp/moe.csv`、`out/moe/curves.png`。

## 4. 评测与可视化（S10-07）

- PPL / 生成：`python -m mymm.eval --weights ... --skip_gen 0`
- 速度/显存：`mymm.eval` 的 report.csv（tokens_per_s / peak_mb）
- 主观质量：`out/vlm_eval/report.md` 或 `out/eval/generations_*.md` 打 1-5 分
- 图：
  1. loss 曲线（`out/moe/curves.png` 或 `out/exp/curves.png`）
  2. PPL / 速度柱状图（从 report.csv 画）
  3. 生成样例对比（前后各 3 条）

## 5. 报告骨架（S10-08，500-1000 字）

1. **背景与问题**（100 字）：为什么做这个项目，原模型/数据哪里不够。
2. **方法**（150 字）：基线是什么；只改了哪个变量；关键配置（参数量、超参、数据）。
3. **实验设置**（100 字）：数据量、训练步数、硬件、评测指标。
4. **结果**（200 字）：主表 + 曲线；同预算下对比；速度/显存。
5. **失败分析**（100 字）：<例如 MoE 在 100 步内没超过 dense，因为路由还没学好 / aux 系数太大 / 数据太少>。
6. **下一步**（50 字）：<加数据、调 aux_coef、换 top-k、蒸馏回 dense>。

## 6. 复盘（S10-R）

写 3 条可执行改进：
1. `<例如：先跑 50 步测速再定实验规模，避免浪费 2 小时>`
2. `<例如：MoE 对照要固定随机种子，否则差异是噪声>`
3. `<例如：评测表提前定好，不要训完才想怎么评>`
