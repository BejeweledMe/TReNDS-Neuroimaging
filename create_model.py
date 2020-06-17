import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x)
        return out


class Conv3DRegressor(nn.Module):

    def __init__(self, n_classes=1):
        super(Conv3DRegressor, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=53, out_channels=128, kernel_size=(3, 3, 3), padding=2)
        self.bn1 = nn.BatchNorm3d(128)
        self.m_pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.res_conv1 = ResBlock()
        self.m_pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.res_conv2 = ResBlock()
        self.adap_pool = nn.AdaptiveMaxPool3d((4, 4, 4))
        self.res_conv3 = ResBlock()
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.m_pool1(x)
        x = self.res_conv1(x)
        x = self.m_pool2(x)
        x = self.res_conv2(x)
        x = self.adap_pool(x)
        x = self.res_conv3(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def create_nn(n_classes=1):
    return Conv3DRegressor(n_classes=n_classes)