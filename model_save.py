import torch
import torchvision

vgg16=torchvision.models.vgg16(pretrained=False)  #等价于vgg16 = vgg16(weights=None) 没有预训练权重
#你看到的这两行 UserWarning 是因为你使用的 torchvision.models.vgg16(pretrained=False) 写法在 新版本的 torchvision（≥0.13） 中已经被弃用了。

# 保存方式1 后缀最好是.pth，   方式1可以同时保存模型及参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，把vgg16的状态（参数）保存为字典，没有结构 【官方推荐】 因为空间小
torch.save(vgg16.state_dict(),"vgg16_method2.pth")


##注意，有个陷阱，如果你要加载自己的模型，需要将模型那个类的定义导进来（可以通过import方法，一般我们会单独创建一个文件夹，保存所有模型的定义），不然报错