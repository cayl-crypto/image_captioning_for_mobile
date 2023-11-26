import torch
import torchvision.transforms as T
import torch.nn as nn
import timm


class ViT(nn.Module):
    def __init__(self, pretrained=True) -> None:
        super(ViT, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)

    def forward(self, x):
        x = self.model.forward_features(x)
        return x

# model = ViT()
# torch.save(model.cpu(), "vit_model.pt")