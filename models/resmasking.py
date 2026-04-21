import torch
import torch.nn as nn

from .masking import masking
from .resnet import BasicBlock, ResNet


class ResMasking(ResNet):
    def __init__(self, in_channels=3, num_classes=7):
        super(ResMasking, self).__init__(
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            in_channels=in_channels,
            num_classes=1000,
        )
        
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet34-333f7ec4.pth", progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, num_classes)

        self.mask1 = masking(64, 64, depth=4)
        self.mask2 = masking(128, 128, depth=3)
        self.mask3 = masking(256, 256, depth=2)
        self.mask4 = masking(512, 512, depth=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        m = self.mask1(x)
        x = x * (1 + m)

        x = self.layer2(x)
        m = self.mask2(x)
        x = x * (1 + m)

        x = self.layer3(x)
        m = self.mask3(x)
        x = x * (1 + m)

        x = self.layer4(x)
        m = self.mask4(x)
        x = x * (1 + m)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resmasking_dropout1(in_channels=3, num_classes=7, weight_path=""):
    del weight_path

    model = ResMasking(in_channels=in_channels, num_classes=num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model
