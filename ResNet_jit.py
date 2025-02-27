import math
import jittor as jt
from jittor import nn
import pdb

def draw_single_ellipse(H, W):
    pi = math.pi
    x = jt.rand(1) * H
    y = jt.rand(1) * W
    alpha = jt.rand(1) * 2 * pi
    w = jt.rand(1)
    a = jt.clamp(jt.normal(mean=max(H, W)/2, std=max(H, W)/6), 0, max(H, W)/2)
    b = jt.clamp(jt.normal(mean=min(H, W)/2, std=min(H, W)/6), 0, min(H, W)/2)
    yy, xx = jt.meshgrid(jt.arange(H), jt.arange(W))
    xx_centered = xx - x
    yy_centered = yy - y
    xx_rotated = xx_centered * jt.cos(alpha) + yy_centered * jt.sin(alpha)
    yy_rotated = -xx_centered * jt.sin(alpha) + yy_centered * jt.cos(alpha)
    mask = (((xx_rotated**2) / a**2 + (yy_rotated**2) / b**2) <= 1).float()
    mask = mask * w
    return mask.float()

def draw_full_noise(H, W):
    mask = draw_single_ellipse(H, W)
    noisenum = jt.randint(2, 5, (1,)).item()
    for _ in range(noisenum):
        mask = jt.maximum(mask, draw_single_ellipse(H, W))
    return mask

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding in Jittor"""
    return nn.Conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def execute(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def execute(self, x):
        avg_out = jt.mean(x, dim=1, keepdims=True)
        max_out = jt.max(x, dim=1, keepdims=True)[0]
        x = jt.concat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
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
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
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
        super().__init__()
        self.inplanes = 64
        self.expansion = block.expansion
        self.conv1 = nn.Conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                jt.init.gauss_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm):
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def execute(self, x, noise=False):
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
        pool = nn.AdaptiveAvgPool2d(1)
        f = pool(h4)
        f = f.view(f.size(0), -1)
        return f, (h0, h1, h2, h3, h4)

def resnet18_cbam_jittor(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def test_resnet18_cbam_jittor():
    model = resnet18_cbam_jittor(num_classes=10)
    model.eval()
    input_tensor = jt.randn(4, 3, 32, 32)
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
    test_resnet18_cbam_jittor()