"""
OMNI student and distillation heads.

    - OmniStudent       : ViT with a Mixture-of-Experts tail and one learnable token per teacher.
    - TeacherProjector  : one linear head per teacher, mapping student tokens to the teacher space.

The parameter names of OmniStudent are identical to the released Hugging Face model
(`modeling_omni.py`), so a trained `OmniStudent.state_dict()` can be exported as is.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

ATTN_BACKENDS = ("sdpa", "flash_attn")

# ============================================================
# Building blocks
# ============================================================


class Attention(nn.Module):
    """
    Multi-head self-attention over [teacher tokens (M) | patch tokens (Np)] with the
    structured mask of Sec. 3.2. Two equivalent backends are available:

        - "sdpa"       : PyTorch scaled-dot-product attention with the full additive mask.
        - "flash_attn" : FlashAttention (`pip install flash-attn`, CUDA + fp16/bf16 only).
                         The mask is applied exactly by splitting the computation:
                         patch tokens only attend to patch tokens (unmasked flash-attn), and
                         each teacher token attends to itself + the patch tokens (small M x (1 + Np)
                         attention computed explicitly).

    Args:
        dim (int): Token dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether the query/key/value projection has a bias.
        num_teachers (int): Number of teacher tokens M at the start of the sequence.
        backend (str): "sdpa" or "flash_attn".
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        num_teachers: int,
        backend: str = "sdpa",
    ):
        super().__init__()
        assert backend in ATTN_BACKENDS, f"Unknown attention backend {backend}"
        if backend == "flash_attn" and flash_attn_func is None:
            raise ImportError(
                "backend='flash_attn' requires the flash-attn package: pip install flash-attn"
            )
        self.num_heads = num_heads
        self.num_teachers = num_teachers
        self.backend = backend
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tokens of shape (B, N, D).
            attn_mask (torch.Tensor): Additive mask of shape (N, N) (0 = attend, -inf = masked).
                Only used by the "sdpa" backend.

        Returns:
            torch.Tensor: Updated tokens of shape (B, N, D).
        """
        B, N, D = x.shape
        q, k, v = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).unbind(2)
        )  # (B, N, H, Dh)
        if self.backend == "flash_attn":
            x = self._structured_attention(q, k, v)
        else:
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, N, Dh)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(
                1, 2
            )
        return self.proj(x.reshape(B, N, D))

    def _structured_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Exact structured attention using flash-attn for the patch-to-patch part.

        Args:
            q, k, v (torch.Tensor): Queries, keys and values of shape (B, M + Np, H, Dh).

        Returns:
            torch.Tensor: Attention output of shape (B, M + Np, H, Dh).
        """
        M = self.num_teachers
        q_t, k_t, v_t = q[:, :M], k[:, :M], v[:, :M]  # teacher tokens
        q_p, k_p, v_p = q[:, M:], k[:, M:], v[:, M:]  # patch tokens

        # Patch tokens attend only to patch tokens: unmasked flash attention (the expensive part).
        out_p = flash_attn_func(q_p, k_p, v_p)  # (B, Np, H, Dh)

        # Each teacher token attends to itself and to all patch tokens.
        scale = q.shape[-1] ** -0.5
        s_self = (q_t * k_t).sum(-1, keepdim=True) * scale  # (B, M, H, 1)
        s_patch = torch.einsum("bmhd,bnhd->bmhn", q_t, k_p) * scale  # (B, M, H, Np)
        scores = torch.cat([s_self, s_patch], dim=-1)
        w = torch.softmax(
            scores, dim=-1, dtype=torch.promote_types(scores.dtype, torch.float32)
        ).to(v.dtype)
        out_t = w[..., :1] * v_t + torch.einsum(
            "bmhn,bnhd->bmhd", w[..., 1:], v_p
        )  # (B, M, H, Dh)

        return torch.cat([out_t, out_p], dim=1)


class Mlp(nn.Module):
    """
    Two-layer feed-forward network (Linear -> GELU -> Linear).

    Args:
        dim (int): Input / output dimension.
        hidden_dim (int): Hidden dimension.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (..., dim).

        Returns:
            torch.Tensor: Output of shape (..., dim).
        """
        return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    """
    Sparse Mixture-of-Experts layer with top-k routing and the Switch Transformer
    load-balancing loss (Eq. 7), stored in `self.aux_loss` at every forward pass.

    Args:
        dim (int): Token dimension.
        hidden_dim (int): Hidden dimension of each expert.
        num_experts (int): Number of experts E.
        top_k (int): Number of experts used per token.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([Mlp(dim, hidden_dim) for _ in range(num_experts)])
        self.aux_loss = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tokens of shape (B, N, D).

        Returns:
            torch.Tensor: Tokens of shape (B, N, D).
        """
        B, N, D = x.shape
        x = x.reshape(-1, D)

        # Select the top-k experts per token and re-normalize their weights to sum to 1.
        probs = F.softmax(self.router(x), dim=-1)
        weights, selected = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Load-balancing loss: E * sum_e f_e * p_e, with f_e the fraction of routing slots
        # assigned to expert e and p_e its mean routing probability.
        f = F.one_hot(selected, self.num_experts).float().mean(dim=(0, 1))
        p = probs.mean(dim=0)
        self.aux_loss = self.num_experts * (f * p).sum()

        # Run each expert only on the tokens that were routed to it.
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(selected == i)
            if token_idx.numel() > 0:
                w = weights[token_idx, slot_idx].unsqueeze(1)
                out.index_add_(0, token_idx, (expert(x[token_idx]) * w).to(out.dtype))
        return out.view(B, N, D)


class Block(nn.Module):
    """
    Pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x)).

    Args:
        dim (int): Token dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): FFN hidden size as a multiple of dim.
        use_moe (bool): If True, the FFN is a MoE layer, otherwise a dense MLP.
        num_experts (int): Number of experts (MoE blocks only).
        top_k (int): Experts per token (MoE blocks only).
        num_teachers (int): Number of teacher tokens.
        attn_backend (str): "sdpa" or "flash_attn".
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        use_moe: bool,
        num_experts: int,
        top_k: int,
        num_teachers: int,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        # Dense blocks use a qkv bias, MoE blocks do not.
        self.attn = Attention(dim, num_heads, not use_moe, num_teachers, attn_backend)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = (
            MoE(dim, hidden_dim, num_experts, top_k)
            if use_moe
            else Mlp(dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tokens of shape (B, N, D).
            attn_mask (torch.Tensor): Additive attention mask of shape (N, N).

        Returns:
            torch.Tensor: Tokens of shape (B, N, D).
        """
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Student
# ============================================================


class OmniStudent(nn.Module):
    """
    OMNI student: [teacher tokens (M) | patch tokens (Np)] processed by a ViT whose last
    `num_moe_layers` blocks are MoE blocks, with the structured attention mask of Sec. 3.2.

    Args:
        teachers (list[str]): Teacher names (one token per teacher, in this order).
        embed_dim (int): Token dimension D.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): FFN / expert hidden size as a multiple of D.
        num_moe_layers (int): Number of final blocks using a MoE FFN.
        num_experts (int): Number of experts per MoE layer.
        top_k (int): Experts per token.
        img_size (int): Input resolution.
        patch_size (int): Patch size.
        attn_backend (str): "sdpa" (default) or "flash_attn". Both compute the same attention.
    """

    def __init__(
        self,
        teachers: list,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        num_moe_layers: int,
        num_experts: int = 5,
        top_k: int = 2,
        img_size: int = 224,
        patch_size: int = 14,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.teachers = list(teachers)
        num_patches = (img_size // patch_size) ** 2
        M, D = len(teachers), embed_dim

        self.patch_embed = nn.Conv2d(3, D, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, D))
        self.teacher_tokens = nn.Parameter(torch.zeros(1, M, D))
        self.teacher_pos_embed = nn.Parameter(torch.zeros(1, M, D))

        first_moe = depth - num_moe_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    D,
                    num_heads,
                    mlp_ratio,
                    i >= first_moe,
                    num_experts,
                    top_k,
                    M,
                    attn_backend,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(D)

        self._init_weights()
        self.register_buffer(
            "attn_mask", self._attention_mask(M, num_patches), persistent=False
        )

    def _init_weights(self):
        """Random initialization: truncated normal (std 0.02) for embeddings and linear layers."""
        for p in (self.pos_embed, self.teacher_tokens, self.teacher_pos_embed):
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _attention_mask(num_teachers: int, num_patches: int) -> torch.Tensor:
        """
        Builds the structured attention mask (rows = queries, columns = keys).

            - teacher token -> itself only among teacher tokens
            - teacher token -> all patch tokens
            - patch token   -> all patch tokens (patch tokens never see teacher tokens)

        Args:
            num_teachers (int): Number of teacher tokens M.
            num_patches (int): Number of patch tokens Np.

        Returns:
            torch.Tensor: Additive mask of shape (M + Np, M + Np) with 0 (attend) or -inf (masked).
        """
        M, L = num_teachers, num_teachers + num_patches
        allowed = torch.zeros(L, L, dtype=torch.bool)
        allowed[:M, :M] = torch.eye(M, dtype=torch.bool)
        allowed[:M, M:] = True
        allowed[M:, M:] = True
        return torch.full((L, L), float("-inf")).masked_fill(allowed, 0.0)

    def moe_aux_loss(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Load-balancing loss of the last forward pass, summed over MoE blocks.
        """
        return sum(b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, MoE))

    def forward(
        self,
        x: torch.Tensor,
        return_teacher_tokens: bool = False,
        return_patch_tokens: bool = False,
    ):
        """
        Encodes a batch of tiles (same API as the released Hugging Face model).

        Args:
            x (torch.Tensor): Normalized images of shape (B, 3, 224, 224).
            return_teacher_tokens (bool): Also return the M teacher-specific tokens.
            return_patch_tokens (bool): Also return the patch tokens.

        Returns:
            torch.Tensor: If no flag is set, the tile embedding of shape (B, D), i.e. the
                average of the M teacher tokens (default for inference).
            dict: If a flag is set, a dict with:
                - "embedding" (B, D): average of the teacher tokens.
                - "teacher_tokens" (B, M, D): one token per teacher (if `return_teacher_tokens`).
                - "patch_tokens" (B, Np, D): patch tokens (if `return_patch_tokens`).
        """
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        teachers = (self.teacher_tokens + self.teacher_pos_embed).expand(B, -1, -1)
        x = torch.cat([teachers, patches], dim=1)

        attn_mask = self.attn_mask.to(x.dtype)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)

        M = teachers.shape[1]
        teacher_tokens, patch_tokens = x[:, :M], x[:, M:]
        embedding = teacher_tokens.mean(dim=1)

        if not (return_teacher_tokens or return_patch_tokens):
            return embedding

        out = {"embedding": embedding}
        if return_teacher_tokens:
            out["teacher_tokens"] = teacher_tokens
        if return_patch_tokens:
            out["patch_tokens"] = patch_tokens
        return out


# ============================================================
# Distillation heads
# ============================================================


class TeacherProjector(nn.Module):
    """
    One linear head per teacher, projecting student tokens to the teacher embedding space.
    Only used for training (not part of the released encoder).

    Args:
        in_dim (int): Student token dimension.
        teacher_dims (dict[str, int]): Output dimension of each teacher.
        teachers (list[str]): Teachers to create a head for.
    """

    def __init__(self, in_dim: int, teacher_dims: dict, teachers: list):
        super().__init__()
        self.heads = nn.ModuleDict(
            {t: nn.Linear(in_dim, teacher_dims[t]) for t in teachers}
        )

    def forward(self, x: torch.Tensor, teacher: str) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Student tokens of shape (..., in_dim).
            teacher (str): Teacher name.

        Returns:
            torch.Tensor: Projected tokens of shape (..., teacher_dim).
        """
        return self.heads[teacher](x)
