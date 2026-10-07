"""
Configuration of the OMNI students, the teachers and the training (Table 6 of the paper).
"""

# ============================================================
# Teachers
# ============================================================

# The 10 teachers used in the paper. The order defines the order of the teacher tokens.
TEACHERS = [
    "hoptimus1",  # H-optimus-1
    "virchow2",  # Virchow2
    "uni2h",  # UNI2-h
    "provgigapath",  # Prov-GigaPath
    "kaiko_vitb8",  # Kaiko-ViT-B/8
    "titan",  # CONCH v1.5 (tile encoder of TITAN)
    "hiboul",  # Hibou-L
    "h0mini",  # H0-mini
    "keep",  # KEEP
    "dinov3vitl16pretrainlvd1689m",  # DINOv3-ViT-L/16
]

# Dimension of each teacher's global (CLS) embedding.
# Stored targets that are longer (Virchow2 and H0-mini store [CLS, mean patch token])
# are truncated to their first TEACHER_CLS_DIM values.
TEACHER_CLS_DIM = {
    "hoptimus1": 1536,
    "virchow2": 1280,
    "uni2h": 1536,
    "provgigapath": 1536,
    "kaiko_vitb8": 768,
    "titan": 768,
    "hiboul": 1024,
    "h0mini": 768,
    "keep": 768,
    "dinov3vitl16pretrainlvd1689m": 1024,
}

# Dimension of each teacher's patch embeddings.
TEACHER_PATCH_DIM = {
    **TEACHER_CLS_DIM,
    "titan": 1024,
    "keep": 1024,
}

# Number of patch tokens of each teacher (square grid: 196 = 14 x 14, 256 = 16 x 16, 784 = 28 x 28).
TEACHER_NUM_PATCHES = {
    "hoptimus1": 256,
    "virchow2": 256,
    "uni2h": 256,
    "provgigapath": 196,
    "kaiko_vitb8": 784,
    "titan": 784,
    "hiboul": 256,
    "h0mini": 256,
    "keep": 196,
    "dinov3vitl16pretrainlvd1689m": 196,
}

# ============================================================
# Students
# ============================================================

MODELS = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=3.0, num_moe_layers=3),
    "small": dict(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=3.0, num_moe_layers=3
    ),
    "base": dict(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=3.0, num_moe_layers=3
    ),
    "large": dict(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=2.5, num_moe_layers=5
    ),
}

# Shared by all students.
NUM_EXPERTS = 5
TOP_K = 2
IMG_SIZE = 224
PATCH_SIZE = 14

# ============================================================
# Training
# ============================================================

TRAINING = dict(
    batch_size=128,
    lr=1e-4,  # Adam, constant learning rate
    iterations=700_000,
    moe_aux_coef=0.05,  # alpha_MoE
    lambda_reg=1.0,  # weight of the SmoothL1 term
    smooth_l1_beta=1.0,
    lambda_ctr=0.1,  # weight of the InfoNCE term
    temperature=0.1,  # InfoNCE temperature
)
