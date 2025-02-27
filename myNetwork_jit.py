from itertools import chain
import pdb
import jittor as jt
from jittor import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def execute(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.in_channels = in_channels
        
        self.convblock = ConvBlock(in_channels=2*in_channels, out_channels=in_channels, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def execute(self, input_1, input_2):
        # Concatenate inputs along the channel dimension
        x = jt.concat((input_1, input_2), dim=1)  # Concatenation along channel axis (C)
        
        # Check for consistency in the number of channels after concatenation
        assert 2 * self.in_channels == x.shape[1], f'Expected 2 * in_channels, but got {x.shape[1]}'

        # Pass through the ConvBlock
        feature = self.convblock(x)
        
        # Adaptive average pooling
        x = self.avgpool(feature)

        # Apply the convolution layers with ReLU and Sigmoid
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        
        # Element-wise multiplication (Hadamard product) with the feature map
        x = jt.multiply(feature, x)
        
        # Add the original feature map back to the output
        x = jt.add(x, feature)

        return x

class joint_network_dual(nn.Module):
    def __init__(self, numclass, numsuperclass, feature_extractor):
        super(joint_network_dual, self).__init__()
        # Slow Learner
        self.feature = feature_extractor
        self.classifier_coarse = nn.Linear(512, numsuperclass, bias=True)

        # Fast Learner (CIL)
        self.f_conv1 = self._make_conv2d_layer(3, 64, padding=1, max_pool=False)
        self.fusion_blocks1 = FeatureFusionModule(in_channels=64)
        self.f_conv2 = self._make_conv2d_layer(64, 128, padding=1, max_pool=True)
        self.fusion_blocks2 = FeatureFusionModule(in_channels=128)
        self.f_conv3 = self._make_conv2d_layer(128, 256, padding=1, max_pool=True)
        self.fusion_blocks3 = FeatureFusionModule(in_channels=256)
        self.f_conv4 = self._make_conv2d_layer(256, 512, padding=1, max_pool=True)
        self.fusion_blocks4 = FeatureFusionModule(in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, numclass * 4, bias=True)
        self.classifier = nn.Linear(512, numclass, bias=True)

        # Decoder
        self.downsample = 8
        self.fc_pixel_sal = nn.Conv(512 * self.feature.expansion, self.downsample ** 2, 1)
        self.fc_pixel_edge = nn.Conv(512 * self.feature.expansion, self.downsample ** 2, 1)

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, max_pool=False, padding=1):
        layers = [nn.Conv(in_maps, out_maps, kernel_size=3, stride=1, padding=padding),
                  nn.BatchNorm2d(out_maps), nn.ReLU()]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def slow_learner(self):
        param = chain(self.feature.parameters(), self.classifier_coarse.parameters())
        for p in param:
            yield p

    def fast_learner(self):
        param = chain(self.f_conv1.parameters(), self.f_conv2.parameters(), self.f_conv3.parameters(),
                      self.f_conv4.parameters(), self.fc.parameters(), self.classifier.parameters(),
                      self.fusion_blocks1.parameters(), self.fusion_blocks2.parameters(),
                      self.fusion_blocks3.parameters(), self.fusion_blocks4.parameters())
        for p in param:
            yield p

    def learnable_parameters(self):
        param = chain(self.feature.parameters(), self.classifier_coarse.parameters(),
                      self.f_conv1.parameters(), self.f_conv2.parameters(), self.f_conv3.parameters(),
                      self.f_conv4.parameters(), self.fc.parameters(), self.classifier.parameters(),
                      self.fusion_blocks1.parameters(), self.fusion_blocks2.parameters(),
                      self.fusion_blocks3.parameters(), self.fusion_blocks4.parameters())
        for p in param:
            yield p

    def execute(self, x, **kwargs):
        noise, sal = kwargs['noise'], kwargs['sal']

        coarse_feature, (h0, h1, h2, h3, h4) = self.feature(x, noise=noise)

        m1_ = self.f_conv1(x)
        m1 = self.fusion_blocks1(m1_, h1)
        m2_ = self.f_conv2(m1)
        m2 = self.fusion_blocks2(m2_, h2)
        m3_ = self.f_conv3(m2)
        m3 = self.fusion_blocks3(m3_, h3)
        m4_ = self.f_conv4(m3)
        m4 = self.fusion_blocks4(m4_, h4)
        fine_feature = self.avgpool(m4)
        fine_feature = fine_feature.view(fine_feature.size(0), -1)
        fine_output = self.classifier(fine_feature)

        if not sal:
            return fine_output, coarse_feature, fine_feature
        else:
            intermediate_x = [m2[::4], m3[::4], m4[::4]]
            for i in range(len(intermediate_x)):
                intermediate_x[i] = jt.mean(intermediate_x[i], dim=1, keepdim=True)

            def convert(module, x):
                x = module(x)
                spatial_size = x.shape[-2:]
                x = x.view(x.shape[0], self.downsample, self.downsample, *x.shape[-2:]).permute(0, 1, 3, 2, 4)
                x = x.contiguous().view(x.shape[0], self.downsample * spatial_size[0], self.downsample * spatial_size[1])
                return x

            x_sal = convert(self.fc_pixel_sal, m4[::4])
            x_edge = convert(self.fc_pixel_edge, m4[::4])

            return fine_output, coarse_feature, fine_feature, (x_sal, x_edge, intermediate_x)

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass * 4, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_feature = self.classifier.in_features
        out_feature = self.classifier.out_features

        self.classifier = nn.Linear(in_feature, numclass, bias=True)
        self.classifier.weight.data[:out_feature] = weight[:out_feature]
        self.classifier.bias.data[:out_feature] = bias[:out_feature]

    def feature_extractor(self, inputs):
        return self.feature(inputs)

# Test the model
if __name__ == '__main__':
    from ResNet_jit import resnet18_cbam_jittor
    fe = resnet18_cbam_jittor(num_classes=100)
    model = joint_network_dual(50, 20, fe)
    inp = jt.rand(512, 3, 32, 32)
    fine_output, coarse_feature, fine_feature, (oi_sal, oi_edge, inx) = model(inp,noise=False,sal=True)
