"""
Train an OMNI student by multi-teacher distillation (Sec. 3 of the paper).

Usage:
    python train.py --model base --data /path/to/webdataset_shards --output ./outputs/omni-base

Training resumes automatically from the latest checkpoint found in `--output`.
"""

import argparse
import glob
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import (IMG_SIZE, MODELS, NUM_EXPERTS, PATCH_SIZE,
                     TEACHER_CLS_DIM, TEACHER_NUM_PATCHES, TEACHER_PATCH_DIM,
                     TEACHERS, TOP_K, TRAINING)
from omni.data import make_loader
from omni.losses import distillation_loss, info_nce
from omni.model import OmniStudent, TeacherProjector


def set_seed(seed: int):
    """
    Seeds python, numpy and torch.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def align_patches(patches: torch.Tensor, grid_sizes: list) -> dict:
    """
    Resizes the student patch grid to every teacher grid with bilinear interpolation (Eq. 12).

    Args:
        patches (torch.Tensor): Student patch tokens of shape (B, Ns, D), Ns a square number.
        grid_sizes (list[int]): Distinct numbers of patch tokens Ni of the teachers.

    Returns:
        dict[int, torch.Tensor]: For each Ni, the aligned student tokens of shape (B, Ni, D).
    """
    B, Ns, D = patches.shape
    side = math.isqrt(Ns)
    grid = patches.transpose(1, 2).reshape(B, D, side, side)
    aligned = {}
    for Ni in grid_sizes:
        if Ni == Ns:
            aligned[Ni] = patches
        else:
            s = math.isqrt(Ni)
            aligned[Ni] = (
                F.interpolate(grid, size=(s, s), mode="bilinear", align_corners=False)
                .flatten(2)
                .transpose(1, 2)
            )
    return aligned


def latest_checkpoint(folder: str):
    """
    Args:
        folder (str): Output folder of the run.

    Returns:
        str | None: Path of the checkpoint with the most iterations, or None.
    """
    ckpts = glob.glob(os.path.join(folder, "iters_*.pth"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(re.search(r"iters_(\d+)\.pth$", p).group(1)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, choices=list(MODELS), help="Student size."
    )
    parser.add_argument(
        "--data", required=True, help="Folder with the WebDataset shards."
    )
    parser.add_argument(
        "--output", required=True, help="Folder for checkpoints and TensorBoard logs."
    )
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=TEACHERS,
        choices=TEACHERS,
        help="Teachers (default: the 10 of the paper).",
    )
    parser.add_argument("--iterations", type=int, default=TRAINING["iterations"])
    parser.add_argument("--batch-size", type=int, default=TRAINING["batch_size"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "flash_attn"],
        help="Attention backend (flash_attn requires `pip install flash-attn`).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers = args.teachers
    os.makedirs(args.output, exist_ok=True)

    # ---- Student and per-teacher heads ----
    student = OmniStudent(
        teachers,
        **MODELS[args.model],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        attn_backend=args.attn,
    ).to(device)
    D = MODELS[args.model]["embed_dim"]
    cls_heads = TeacherProjector(D, TEACHER_CLS_DIM, teachers).to(device)
    patch_heads = TeacherProjector(D, TEACHER_PATCH_DIM, teachers).to(device)
    params = (
        list(student.parameters())
        + list(cls_heads.parameters())
        + list(patch_heads.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=TRAINING["lr"], fused=device.type == "cuda")
    print(
        f"Student: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M parameters, {len(teachers)} teachers"
    )

    # ---- Resume ----
    start = 0
    ckpt_path = latest_checkpoint(args.output)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert (
            ckpt["teachers"] == teachers and ckpt["model"] == args.model
        ), "Checkpoint does not match the run."
        student.load_state_dict(ckpt["student"])
        cls_heads.load_state_dict(ckpt["cls_heads"])
        patch_heads.load_state_dict(ckpt["patch_heads"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["iters"]
        print(f"Resumed from {ckpt_path}")

    loader = iter(make_loader(args.data, teachers, args.batch_size, args.num_workers))
    grid_sizes = sorted({TEACHER_NUM_PATCHES[t] for t in teachers})
    writer = SummaryWriter(os.path.join(args.output, "tb"))
    logs = {}  # running sums of the logged losses

    student.train(), cls_heads.train(), patch_heads.train()
    for it in tqdm(range(start, args.iterations), initial=start, total=args.iterations):
        images, cls_targets, dense_targets, indices = next(loader)
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = student(images, return_teacher_tokens=True, return_patch_tokens=True)
            teacher_tokens, patches = (
                out["teacher_tokens"],
                out["patch_tokens"],
            )  # (B, M, D), (B, Np, D)
            aligned = align_patches(patches, grid_sizes)

            teacher_losses = []
            for i, t in enumerate(teachers):
                # CLS loss + InfoNCE on the teacher-specific token.
                pred_cls = cls_heads(teacher_tokens[:, i], t)
                tgt_cls = (
                    cls_targets[i]
                    .to(device, non_blocking=True)
                    .float()[:, : TEACHER_CLS_DIM[t]]
                )
                loss_cls = distillation_loss(
                    pred_cls,
                    tgt_cls,
                    TRAINING["lambda_reg"],
                    TRAINING["smooth_l1_beta"],
                ).mean()
                loss_ctr = info_nce(pred_cls, tgt_cls, TRAINING["temperature"])

                # Patch loss on the K sampled positions of the teacher grid.
                idx = indices[i].to(device, non_blocking=True)  # (B, K)
                if it == start:
                    assert (
                        idx.max() < TEACHER_NUM_PATCHES[t]
                    ), f"Sampled indices out of the {t} grid."
                tokens = aligned[TEACHER_NUM_PATCHES[t]].gather(
                    1, idx.unsqueeze(-1).expand(-1, -1, D)
                )
                pred_patch = patch_heads(tokens, t)
                tgt_patch = dense_targets[i].to(device, non_blocking=True).float()
                loss_patch = distillation_loss(
                    pred_patch,
                    tgt_patch,
                    TRAINING["lambda_reg"],
                    TRAINING["smooth_l1_beta"],
                ).mean()

                teacher_losses.append(
                    loss_cls + loss_patch + TRAINING["lambda_ctr"] * loss_ctr
                )
                for name, value in (
                    ("cls", loss_cls),
                    ("patch", loss_patch),
                    ("contrastive", loss_ctr),
                ):
                    logs[f"{name}/{t}"] = logs.get(f"{name}/{t}", 0.0) + value.detach()

            loss_moe = student.moe_aux_loss()
            loss = (
                torch.stack(teacher_losses).mean() + TRAINING["moe_aux_coef"] * loss_moe
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        logs["total"] = logs.get("total", 0.0) + loss.detach()
        logs["moe"] = logs.get("moe", 0.0) + loss_moe.detach()

        # ---- Logging ----
        step = it + 1
        if step % args.log_every == 0:
            n = step - max(start, step - args.log_every)
            for name, value in logs.items():
                writer.add_scalar(f"loss/{name}", (value / n).item(), step)
            logs = {}

        # ---- Checkpointing ----
        if step % args.ckpt_every == 0 or step == args.iterations:
            torch.save(
                {
                    "iters": step,
                    "model": args.model,
                    "teachers": teachers,
                    "student": student.state_dict(),
                    "cls_heads": cls_heads.state_dict(),
                    "patch_heads": patch_heads.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.output, f"iters_{step}.pth"),
            )

    writer.close()


if __name__ == "__main__":
    main()
