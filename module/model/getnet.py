import torchvision
import torch.nn as nn

def net(num_classes):
    net = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    net.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
    net.maxpool=nn.MaxPool2d(1,1)
    return net