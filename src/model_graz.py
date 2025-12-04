# src/model_graz.py
import torch.nn as nn
import timm

class SwinFractureNet(nn.Module):
    def __init__(self, num_type_classes=3, num_severity_classes=3, backbone="swinv2_base_window8_256", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.frac_head = nn.Linear(feat_dim, 2)               # binary
        self.type_head = nn.Linear(feat_dim, num_type_classes)  # multiclass
        self.sev_head = nn.Linear(feat_dim, num_severity_classes)  # severity (3 classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.frac_head(feats), self.type_head(feats), self.sev_head(feats)
