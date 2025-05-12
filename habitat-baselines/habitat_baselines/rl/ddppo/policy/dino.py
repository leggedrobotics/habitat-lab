import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model

class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)  # Downsample 224x224 → 112x112
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-small")
        self.model.eval()

        self.output_dim = self.model.config.hidden_size  # 384 for dinov2-small

    @property
    def is_blind(self):
        return False

    # def forward(self, observations):
    #     rgb = observations["rgb"]  # [B, H, W, C]
    #     rgb = rgb.permute(0, 3, 1, 2)  # B x C x H x W
    #     rgb = rgb.float() / 255.0      # ✅ Convert to float in [0, 1] range
    #     rgb = self.pool(rgb)           # Downsample to 112x112

    #     with torch.no_grad():
    #         outputs = self.model(pixel_values=rgb)
    #         cls_token = outputs.last_hidden_state[:, 0]  # CLS token

    #     return cls_token  # B x 384

    def forward(self, observations):
        depth = observations["depth"]  # [B, H, W, C]
        # Need to convert to 3 channel
        depth = torch.cat([depth, depth, depth], dim=-1)  # [B, H, W, 3]
        depth = depth.permute(0, 3, 1, 2)  # B x C x H x W
        depth = self.pool(depth)           # Downsample to 112x112

        with torch.no_grad():
            outputs = self.model(pixel_values=depth)
            cls_token = outputs.last_hidden_state[:, 0]  # CLS token

        return cls_token  # B x 384