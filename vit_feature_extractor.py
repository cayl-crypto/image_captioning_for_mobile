import torch
import torchvision.transforms as T
import torch.nn as nn
import timm


class ViT(nn.Module):
    def __init__(self) -> None:
        super(ViT, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.avgpool = nn.AdaptiveMaxPool1d(16)  # reduce sequence length from 196 to 16

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        # Add reshaping here.
        x = x.transpose(1, 2)  # swap sequence length and embedding dimension
        x = self.avgpool(x)
        x = x.transpose(1, 2)  # swap back
        x = self.model.norm(x)
        return x



