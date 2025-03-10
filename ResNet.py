import math
import random
import pdb
import torch
import torch.nn as nn

def draw_single_ellipse(H, W):
    pi = math.pi

    # Sample the ellipse parameters using only torch
    x = torch.rand(1) * H
    y = torch.rand(1) * W
    alpha = torch.rand(1) * 2 * pi
    w = torch.rand(1)  # Not used in the mask generation, but could be used for weighting
    a = torch.clamp(torch.normal(mean=torch.tensor(max(H, W)/2), std=torch.tensor(max(H, W)/6)), 0, max(H, W)/2)
    b = torch.clamp(torch.normal(mean=torch.tensor(min(H, W)/2), std=torch.tensor(min(H, W)/6)), 0, min(H, W)/2)

    # Generate the grid without specifying indexing
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))

    xx_centered = xx - x
    yy_centered = yy - y

    # Rotate the coordinates
    xx_rotated = xx_centered * torch.cos(alpha) + yy_centered * torch.sin(alpha)
    yy_rotated = -xx_centered * torch.sin(alpha) + yy_centered * torch.cos(alpha)

    # Generate the mask
    mask = (((xx_rotated**2) / a**2 + (yy_rotated**2) / b**2) <= 1).float()
    mask = mask * w

    return mask.float()

def draw_full_noise(H, W):
    mask = draw_single_ellipse(H, W)
    noisenum = random.randint(2,4)
    for _ in range(noisenum):
        mask = torch.maximum(mask, draw_single_ellipse(H, W))
    return mask


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.expansion = block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, noise=False):
        h0 = self.conv1(x)
        h0 = self.bn1(h0)
        h0 = self.relu(h0)
        h1 = self.layer1(h0)
        if noise:
            h1 = self.inject_noise(h1)
        h2 = self.layer2(h1)
        if noise:
            h2 = self.inject_noise(h2)
        h3 = self.layer3(h2)
        if noise:
            h3 = self.inject_noise(h3)
        h4 = self.layer4(h3)
        if noise:
            h4 = self.inject_noise(h4)
        dim = h4.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        f = pool(h4)
        f = f.view(f.size(0), -1)
        return f, (h0, h1, h2, h3, h4)

    def noise_map(self, map, device):
        h, w = map.shape[-2:]
        mask = draw_full_noise(h, w).unsqueeze(0).to(device)
        return mask

    def inject_noise(self, x):
        batch_size, n_channel, h, w = x.shape
        num_select = int(0.1 * n_channel)
        res = x.clone()
        for i in range(batch_size):
            select_channel = torch.randperm(n_channel)[:num_select]
            res[i, select_channel] = x[i, select_channel] * self.noise_map(x[i, select_channel], x.device)
        return res


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def test_resnet18_cbam():
    model = resnet18_cbam(num_classes=10)
    model.eval()
    
    # 创建测试输入(batch_size=4, 3通道, 32x32分辨率)
    input_tensor = torch.randn(4, 3, 32, 32)
    
    # 进行前向传播
    features, (h0, h1, h2, h3, h4) = model(input_tensor)
    
    # 打印输出形状
    print("Feature shape:", features.shape)
    print("h0 shape:", h0.shape)
    print("h1 shape:", h1.shape)
    print("h2 shape:", h2.shape)
    print("h3 shape:", h3.shape)
    print("h4 shape:", h4.shape)

if __name__ == "__main__":
    test_resnet18_cbam()