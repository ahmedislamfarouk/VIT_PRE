"""
Knowledge distillation loss function.

Implements the classic Hinton distillation loss:
    L = alpha * CE(student, hard_labels)
      + (1 - alpha) * T^2 * KL(student/T || teacher/T)

where T is the temperature that softens the probability distributions
and alpha balances hard-target vs. soft-target supervision.

The T^2 factor ensures the KL gradient magnitude stays consistent
across different temperature values.
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    label_smoothing: float,
) -> torch.Tensor:
    """
    Compute the knowledge distillation loss.

    Args:
        student_logits: Raw logits from the student model (B, C).
        teacher_logits: Raw logits from the teacher model (B, C).
        labels: Ground-truth class indices (B,).
        temperature: Softening temperature (higher = softer distributions).
        alpha: Weight for the hard-label CE loss (0-1).
        label_smoothing: Label smoothing factor for the CE component.

    Returns:
        Scalar loss tensor.
    """
    hard_loss = F.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing
    )
    soft_loss = (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * (temperature ** 2)
    )
    return alpha * hard_loss + (1 - alpha) * soft_loss
