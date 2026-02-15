import torch.nn as nn
import torchvision.models as models



#  ResNet backbone with a custom classification head
#  --------------------------------------------------
#  @param num_classes: number of output classes
#  @param backbone: which ResNet variant to use ("resnet18", "resnet34", "resnet50")
#  @param dropout: dropout rate for the classifier head

class NABirdsResNet(nn.Module):
    def __init__(self, num_classes, backbone="resnet18", dropout=0.3):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone_name = backbone
        self.features = nn.Sequential(*list(base.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        # Init classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))
