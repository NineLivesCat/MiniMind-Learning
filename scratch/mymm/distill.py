"""D15 · 白盒蒸馏（LEARNING_TASKS: S8-01 / S8-02）

    L = alpha * CE(student, labels) + (1 - alpha) * T² * KL(teacher/T || student/T)

验收：mymm/tests/test_distill.py 全 PASS。
对照材料：先自己写，测试失败后再看 minimind/trainer/train_distillation.py:25。
关键点：
    - teacher 只在前向时用，no_grad + eval；学生才回传梯度
    - KL 用 softmax(teacher/T) 和 log_softmax(student/T)，再乘 T² 补偿温度缩放
    - 只在 assistant 段（label != -100）上算 CE 和 KL；prompt 段不参与
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["distillation_loss", "distill_loss"]


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, temperature: float = 2.0) -> Tensor:
    """T² * KL(teacher/T || student/T)（TODO D15-1）。

    契约：
    - 支持任意形状 (..., vocab)，内部 reshape 成 (N, vocab)
    - teacher 侧先在 no_grad 下算 softmax(teacher_logits / T)
    - student 侧算 log_softmax(student_logits / T)
    - kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    - 返回标量 (temperature ** 2) * kl
    """
    teacher_probs = None
    vocab = student_logits.size(-1)
    student_logits = student_logits.reshape(-1, vocab)
    teacher_logits = teacher_logits.reshape(-1, vocab)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim = -1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim = -1)
    k1 = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return (temperature ** 2) * k1
    # raise NotImplementedError("TODO D15-1: distillation_loss")


def distill_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> dict:
    """把 shift / mask / CE / KD 组合起来（已给好，不用改）。

    参数：
    - student_logits / teacher_logits: (B, T, vocab)，teacher 的词表可以更大（按学生截断）
    - labels: (B, T)，-100 表示不算 loss（prompt/pad）
    返回 dict：{"loss", "ce", "kd"}
    """
    student = student_logits[:, :-1, :].contiguous()
    teacher = teacher_logits[:, :-1, :].contiguous()[..., : student.size(-1)]
    shift_labels = labels[:, 1:].contiguous()
    use_mask = shift_labels != -100
    flat_labels = shift_labels.view(-1)
    flat_student = student.view(-1, student.size(-1))
    flat_teacher = teacher.view(-1, teacher.size(-1))

    ce = F.cross_entropy(flat_student, flat_labels, ignore_index=-100)
    mask_flat = use_mask.view(-1)
    kd = distillation_loss(
        flat_student[mask_flat],
        flat_teacher[mask_flat],
        temperature=temperature,
    )
    loss = alpha * ce + (1.0 - alpha) * kd
    return {"loss": loss, "ce": ce, "kd": kd}
