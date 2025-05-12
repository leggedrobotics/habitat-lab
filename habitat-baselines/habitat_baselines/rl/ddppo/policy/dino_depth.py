import torch
import torch.nn as nn
from dinov2.configs import load_and_merge_config
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils

class DINOv2DepthEncoder(nn.Module):
    def __init__(self, cfg_path, ckpt_path):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)  # Downsample 224x224 → 112x112

        conf = load_and_merge_config(cfg_path)
        self.model, _ = build_model_from_cfg(conf, only_teacher=True)
        dinov2_utils.load_pretrained_weights(self.model, ckpt_path, "teacher")
        self.model.eval()
        self.output_dim = 384  # 384 for dinov2-small
        print(f"Loaded DINOv2 model from {ckpt_path}")

    @property
    def is_blind(self):
        return False

    def forward(self, observations):
        depth = observations["depth"]  # [B, H, W, C]
        # Need to convert to 3 channel
        depth = torch.cat([depth, depth, depth], dim=-1)  # [B, H, W, 3]
        depth = depth.permute(0, 3, 1, 2)  # B x C x H x W
        depth = self.pool(depth)           # Downsample to 112x112

        with torch.no_grad():
            output = self.model(depth)

        return output  # B x 384