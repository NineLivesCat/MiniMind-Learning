# learning/scratch —— 学习代码草稿区

按任务清单把每个阶段要手写的代码放在这里（不要改仓库源码，方便对比）。建议文件：

| 文件 | 对应任务 | 内容 |
|---|---|---|
| `01_token_stats.py` | S1-03 | 字符数 / token 数 / 压缩比统计 |
| `02_generate.py` | S2-02/03 | 手写 greedy、top-k、top-p 采样 |
| `03_model.py` | S3 全部 | 手写 RMSNorm/RoPE/GQA/SwiGLU/Attention/Model |
| `03_align.py` | S3-11/12 | 加载官方权重、与仓库实现对齐 logits |
| `06_ppl.py` | S6-01 | 滑动窗口 PPL 计算 |
| `07_lora_demo.py` | S7-09 | DPO 前后 chosen/rejected logp margin 统计 |

约定：

- 每个脚本都能独立运行：`.venv\Scripts\python.exe learning\scratch\xx.py`；
- 顶部用注释写清「任务编号 + 目的 + 运行方式」；
- 和仓库实现对比的脚本，把两边输出都打印出来，方便写结论。
