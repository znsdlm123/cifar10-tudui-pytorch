import torchvision
from torch.utils.tensorboard import SummaryWriter


dataset_transform=torchvision.transforms.Compose([ #transforms.Compose([transforms参数])
    torchvision.transforms.ToTensor()
])
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)
# Parameters:
# root (str or pathlib.Path) – Root directory of dataset where directory cifar-10-batches-py exists or will be saved to if download is set to True.
# train (bool, optional) – If True, creates dataset from training set, otherwise creates from test set.
# transform (callable, optional) – A function/transform that takes in a PIL image and returns a transformed version. E.g, transforms.RandomCrop
# target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
# download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.



# print(test_set[0])
# print(test_set.classes)

# img,target=test_set[0]  #返回一个 (image, label)
# print(img)  #PIL对象，RGB 彩色图，尺寸 32×32
# print(target)#标签 整数代表什么类型
# print(test_set.classes[target])
# img.show()

print(test_set[0])

writer=SummaryWriter("p10") #日志写入  日志保存目录p10
for i in range(10):
    img,target=test_set[i]    # img 是 (C,H,W) 的Tensor  # target 是 int 类别标签
    print(test_set.classes[target])  # 输出 'cat'、'dog' 等
    writer.add_image("test_set",img,i)

writer.close()