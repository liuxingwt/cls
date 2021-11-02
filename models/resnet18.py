import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
# from termcolor import cprint
from timm.models import resnet



class ResNet18(nn.Module):
    def __init__(self, pretrained=False, dim=128, num_classes_end=2):
        super(ResNet18, self).__init__()
        self.encoder = resnet.seresnet18(pretrained=pretrained)
        self.dropout = nn.Dropout(0.1)
        self.ReLU = nn.ReLU(inplace=True)
        self.linear = nn.Linear(1000, num_classes_end)

    def forward(self, x):
        embedding = self.encoder(x)
        x = self.dropout(embedding)
        x = self.linear(self.ReLU(x))
        return embedding, x

def resnet18(pretrained):
    net = ResNet18(pretrained)
    return net
    


if __name__ == "__main__":
    # test code
    pretrained = True
    net = resnet18(pretrained)
    print(net)

    input = torch.randn(1, 3, 224, 224)
    output = net(input)
    print(output[0].shape)
    print(output[1].shape)