import torch
import torchvision
from torchvision.models import vgg16

# 方式1->保存方式1，加载模型 torch.load()
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型
# vgg16 = torchvision.models.vgg16(pretrained=False)

# 方式2 加载时  先创建结构
from torchvision.models import vgg16
vgg16 = vgg16(weights=None) #没有预训练权重
# 参数和结构放一起  权重参数保存在dict字典
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)

#注意，有个陷阱，如果你要加载自己的模型，需要将模型那个类的定义导进来（可以通过import方法，一般我们会单独创建一个文件夹，保存所有模型的定义），不然报错