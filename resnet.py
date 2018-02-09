import torch
from torch import nn
from torch.nn import functional as f
import math

class basicLayer(nn.Module):
    def __init__(self, inChannel, outChannel, kernel_size=3, stride=1,
                 preactivation=False):
        super(basicBlock, self).__init__()
        self.preactivation = preactivation
        self.projection = (stride > 1) or (inChannel != outChannel)
        self.bn0 = nn.BatchNorm2d(inChannel)
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=kernel_size,
                               stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(outChannel)
        self.relu2 = nn.ReLU(inplace=True)
        if self.projection:
            self.conv_project = nn.Conv2d(inChannel, outChannel, kernel_size=1,
                                          stride=stride)
            self.bn_project = nn.BatchNorm2d(outChannel)


    def forward(self, x):
        ## TODO: Implement preactivation version
        if not self.preactivation:
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(x))
            if self.projection:
                residual = self.bn_project(self.conv_project(x))
            else:
                residual = x
            return self.relu2(residual+out)
        else:
            out = self.conv1(self.relu1(self.bn0(x)))
            out = self.conv2(self.relu2(self.bn1(out)))
            if self.projection:
                residual = self.conv_project(x)
            else:
                residual = x
            return residual+out




class bottleNeck(nn.Sequential):
    def __init__(self, inChannel, outChannel, kernel_size=3, stride=1,
                 preactivation=False):
        super(bottleNeck, self).__init__()
        self.preactivation = preactivation
        self.projection = (stride > 1) or (inChannel != outChannel)
        downsampleChannel = outChannel // 4
        self.bn0 = nn.BatchNorm2d(inChannel)
        ## Compression
        self.conv1 = nn.Conv2d(inChannel, downsampleChannel, kernel_size=1,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(downsampleChannel)
        self.relu1 = nn.ReLU(inplace=True)
        ## Spatial downsampling actually happens here
        self.conv2 = nn.Conv2d(downsampleChannel, downsampleChannel,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(downsampleChannel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(downsampleChannel, outChannel, kernel_size=1,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(outChannel)
        self.relu3 = nn.ReLU()
        if self.projection:
            self.conv_project = nn.Conv2d(inChannel, outChannel, kernel_size=1,
                                          stride=stride)
            self.bn_project = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        if not self.preactivation:
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.relu2(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.projection:
                residual = self.bn_project(self.conv_project(x))
            else:
                residual = x
            return self.relu3(out+residual)
        else:
            out = self.conv1(self.relu1(self.bn0(x)))
            out = self.conv2(self.relu2(self.bn1(out)))
            out = self.conv3(self.relu3(self.bn2(out)))
            if self.projection:
                residual = self.conv_project(x)
            else:
                residual = x
            return out+residual


class resBlock(nn.Module):
    def __init__(self, unitLayer, inChannel, outChannel, depth, kernel_size=3,
                 init_stride=2, preactivation=False):
        super(resBlock, self).__init__()
        self.layers=[unitLayer(inChannel, outChannel, kernel_size=kernel_size,
                               stride=init_stride, preactivation=preactivation)]
        for i in range(1, depth):
            self.layers.append(unitLayer(outChannel, outChannel, kernel_size=kernel_size,
                                         stride=1, preactivation=preactivation))
        self.one_block = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.one_block(x)

class ResNet152(nn.Module):
    '''
    Configuration of resnet-152 [3, 8, 36, 3]
    '''
    def __init__(self, preactivation=False, nClasses=10, dataset='cifar'):
        super(ResNet152, self).__init__()
        self.dataset = dataset
        self.conv0 = nn.Conv2d(3, 64, 7, 2)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.max_pool0 = nn.MaxPool2d(3, 2, padding=1)
        self.block1 = resBlock(bottleNeck, 64, 256, 3, init_stride=1,
                               preactivation=preactivation)
        self.block2 = resBlock(bottleNeck, 256, 512, 8,
                               preactivation=preactivation)
        self.block3 = resBlock(bottleNeck, 512, 1024, 36,
                               preactivation=preactivation)
        self.block4 = resBlock(bottleNeck, 1024, 2048, 3,
                               preactivation=preactivation)
        if dataset == 'imagenet':
            self.avg_pool = nn.AvgPool2d(7, 2)
        else:
            self.avg_pool = nn.AvgPool2d(1, 2)
        self.fc = nn.Linear(2048, nClasses)
    def forward(self, x):
        out = nn.Sequential(self.conv0, self.bn0, self.relu0, self.max_pool0,
                            self.block1, self.block2, self.block3, self.block4,
                            self.avg_pool)(x)
        ## Batch*2048
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = f.log_softmax(out)
        return out
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

class ResNet101(nn.Module):
    '''
    Configuration of resnet-101 [3, 4, 23, 3]
    '''
    def __init__(self, preactivation=False, nClasses=10, dataset='cifar'):
        super(ResNet101, self).__init__()
        self.dataset = dataset
        self.conv0 = nn.Conv2d(3, 64, 7, 2)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.max_pool0 = nn.MaxPool2d(3, 2, padding=1)
        self.block1 = resBlock(bottleNeck, 64, 256, 3, init_stride=1,
                               preactivation=preactivation)
        self.block2 = resBlock(bottleNeck, 256, 512, 4,
                               preactivation=preactivation)
        self.block3 = resBlock(bottleNeck, 512, 1024, 23,
                               preactivation=preactivation)
        self.block4 = resBlock(bottleNeck, 1024, 2048, 3,
                               preactivation=preactivation)
        if dataset == 'imagenet':
            self.avg_pool = nn.AvgPool2d(7, 2)
        else:
            self.avg_pool = nn.AvgPool2d(1, 2)
        self.fc = nn.Linear(2048, nclass)
    def forward(self, x):
        out = nn.Sequential(self.conv0, self.bn0, self.relu0, self.max_pool0,
                            self.block1, self.block2, self.block3, self.block4,
                            self.avg_pool)(x)
        ## Batch*2048
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = f.log_soft_max(out)
        return out
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

if __name__ == '__main__':
    net = ResNet152()
    net.init_weights()
    print(net)
    total = sum([p.data.nelement() for p in net.parameters()])
    print('  + Number of params: %.2f' % (total / 1e6))
    sample_input = torch.ones([12, 3, 32, 32])
    sample_input = torch.autograd.Variable(sample_input)
    out = net(sample_input)
    print(out.size(), out.sum())
