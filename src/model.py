import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
try:
    from .capi import CAPIProjection
except ImportError:
    from capi import CAPIProjection

class GeometricAwareModel(nn.Module):
    def __init__(self, 
                 model_name='resnet50', 
                 num_classes=200, 
                 pretrained=True, 
                 capi_dim=32):
        super().__init__()
        
        # Load backbone
        # We need the pooled features, so we remove the classifier 'reset_classifier'
        # num_classes=0 returns the pooled feature vector
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        
        # Get feature dimension
        self.in_features = self.backbone.num_features
        
        # 1. Standard Euclidean Classifier
        self.classifier = nn.Linear(self.in_features, num_classes)
        
        # 2. CAPI Projection Module (Geometric Regularization Branch)
        self.capi = CAPIProjection(self.in_features, m=capi_dim)

    def forward(self, x):
        """
        Returns:
            logits: (B, num_classes) for CE Loss
            lie_features: (B, m, m) for Lie Loss
        """
        # Extract features (B, in_features)
        # timm's forward_features usually gives unpooled spatial features or tokens
        # timm's forward usually gives what we asked for in num_classes (pooled if 0)
        features = self.backbone(x)
        
        # Branch 1: Classification
        logits = self.classifier(features)
        
        # Branch 2: Geometric Projection
        lie_features = self.capi(features)
        
        return logits, lie_features
