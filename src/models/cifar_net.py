import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigurableCIFARNet(nn.Module):
    def __init__(self, num_layers: int = 3, base_channels: int = 64,
                 dropout_rate: float = 0.2, num_classes: int = 10):
        super(ConfigurableCIFARNet, self).__init__()

        self.num_layers = num_layers
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = 3
        current_channels = base_channels

        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv2d(in_channels, current_channels, kernel_size=3, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(current_channels))

            if i < num_layers - 1:
                self.conv_layers.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1)
                )
                self.bn_layers.append(nn.BatchNorm2d(current_channels))
                self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = current_channels
            current_channels = min(current_channels * 2, 512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        fc_input_size = in_channels * 4 * 4

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        conv_idx = 0
        bn_idx = 0
        pool_idx = 0

        for i in range(self.num_layers):
            x = self.conv_layers[conv_idx](x)
            x = self.bn_layers[bn_idx](x)
            x = F.relu(x)
            conv_idx += 1
            bn_idx += 1

            if i < self.num_layers - 1:
                x = self.conv_layers[conv_idx](x)
                x = self.bn_layers[bn_idx](x)
                x = F.relu(x)
                conv_idx += 1
                bn_idx += 1

                x = self.pool_layers[pool_idx](x)
                pool_idx += 1

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
