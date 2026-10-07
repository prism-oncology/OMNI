"""
Distillation losses.

    - distillation_loss : cosine + SmoothL1 between student predictions and teacher targets (Phi, Eq. 4).
    - info_nce          : symmetric InfoNCE between student and teacher CLS embeddings (Eq. 10).
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    pred: torch.Tensor, target: torch.Tensor, lambda_reg: float = 1.0, beta: float = 1.0
) -> torch.Tensor:
    """
    Phi(x, y) = [1 - cos(x, y)] + lambda_reg * SmoothL1(x_hat - y_hat), with x_hat, y_hat the
    L2-normalized vectors.

    Args:
        pred (torch.Tensor): Student predictions of shape (..., D).
        target (torch.Tensor): Teacher embeddings of shape (..., D).
        lambda_reg (float): Weight of the SmoothL1 term.
        beta (float): SmoothL1 beta.

    Returns:
        torch.Tensor: Loss for every vector, of shape (...).
    """
    pred, target = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
    cosine = 1.0 - (pred * target).sum(dim=-1)
    smooth_l1 = F.smooth_l1_loss(pred, target, beta=beta, reduction="none").mean(dim=-1)
    return cosine + lambda_reg * smooth_l1


def info_nce(
    pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """
    Symmetric InfoNCE: the i-th student embedding must match the i-th teacher embedding
    among all teacher embeddings of the batch (and vice versa).

    Args:
        pred (torch.Tensor): Student CLS predictions of shape (B, D).
        target (torch.Tensor): Teacher CLS embeddings of shape (B, D).
        temperature (float): Softmax temperature tau.

    Returns:
        torch.Tensor: Scalar loss.
    """
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    logits = pred @ target.T / temperature  # (B, B), positives on the diagonal
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
