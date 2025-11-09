import torchvision
from torchvision.datasets.utils import download_url

# train_data=torchvision.datasets.ImageNet("./data_image_net",split='train',download=False,transform=torchvision.transforms.ToTensor())

import torchvision
from torch import nn

vgg16_true=torchvision.models.vgg16(pretrained=True) #预训练
vgg16_false=torchvision.models.vgg16(pretrained=False) #非预训练 # 随机初始化

# vgg16_true.classifier.add_module('add_linear',nn.Linear(in_features=1000,out_features=10)) #增加了一层
# print(vgg16_true)
#
vgg16_false.classifier[6]=nn.Linear(in_features=4096,out_features=10)  # 修改 vgg16_false 的分类器  替换了原始分类器中最后一个全连接层（classifier[6]）
# 这是更常见的微调方式，适用于自定义分类任务。 替换了原始分类器中最后一个全连接层（classifier[6]）
print(vgg16_false)

