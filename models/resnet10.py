
# 自定义的ResNet10

import torch
import torch.nn as nn
from collections import OrderedDict


class ResLayer(nn.Module):
    def __init__(self, in_c, out_c, celu_parm=0.075):
        super(ResLayer, self).__init__()

        conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        norm = nn.BatchNorm2d(num_features=out_c)
        act = nn.CELU(celu_parm, inplace=False)
        pool = nn.MaxPool2d(2)
        self.pre_conv = nn.Sequential(OrderedDict([('conv', conv), ('pool', pool), ('norm', norm), ('act', act)]))
        self.res1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)), ('bn', nn.BatchNorm2d(out_c)), ('act', act)]))
        self.res2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)), ('bn', nn.BatchNorm2d(out_c)), ('act', act)]))

    def forward(self, x):
        x = self.pre_conv(x)
        out = self.res1(x)
        out = self.res2(out)
        out = out + x
        return out

class ResNet10(nn.Module):
    def __init__(self, drop_rate=0.2, num_classes=2, has_feature=False, celu_parm=0.075, **kwargs):
        super(ResNet10, self).__init__()
        self.has_feature = has_feature
        conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        norm1 = nn.BatchNorm2d(num_features=64)
        act = nn.CELU(celu_parm, inplace=False)
        pool = nn.MaxPool2d(2)
        self.prep = nn.Sequential(OrderedDict([('conv', conv1), ('bn', norm1), ('act', act)]))
        self.layer1 = ResLayer(64, 128, celu_parm)
        conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        norm2 = nn.BatchNorm2d(num_features=256)
        self.layer2 = nn.Sequential(OrderedDict([('conv', conv2), ('pool', pool), ('bn', norm2), ('act', act)]))
        self.layer3 = ResLayer(256, 512, celu_parm)
        self.layer4 = ResLayer(512, 1024, celu_parm)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(1024, num_classes, bias=False)

    def forward(self, x, tau=8.0):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        features = torch.flatten(x, 1)
        x = self.dropout(features)
        logits = self.fc(x)
        logits /= tau

        return features, logits



def resnet10():
    net = ResNet10()
    return net
    

if __name__ == "__main__":
    # test code
    pretrained = True
    net = resnet10()
    print(net)

    input = torch.randn(1, 3, 224, 224)
    output = net(input)
    print(output[0].shape)
    print(output[1].shape)