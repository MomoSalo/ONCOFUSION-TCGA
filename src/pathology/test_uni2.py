import torch
import timm

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# UNI2-h configuration
# ============================================================

timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}


print("Loading UNI2-h...")


model = timm.create_model(
    "hf-hub:MahmoodLab/UNI2-h",
    pretrained=True,
    **timm_kwargs
)

model = model.to(device)

model.eval()


# ============================================================
# Official preprocessing transform
# ============================================================

transform = create_transform(
    **resolve_data_config(
        model.pretrained_cfg,
        model=model
    )
)


# ============================================================
# Model information
# ============================================================

n_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("\nUNI2-h loaded successfully.")

print(
    "Parameters:",
    f"{n_parameters / 1e6:.1f} M"
)

print(
    "Embedding dimension:",
    1536
)