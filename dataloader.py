import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备的测试数据集  每张图在读取时自动转成 Tensor
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
#shuffle (可选): 是否在每个周期开始时打乱数据。默认为False。如果设置为True，则在每个周期开始时，数据将被随机打乱顺序。
#drop_last (可选): 如果数据集大小不能被批次大小整除，是否丢弃最后一个不完整的批次。默认为False。

#测试数据集中第一张图片及target标签
img,target=test_data[0]  #内置的_getitem（） 返回img，target
print(img.shape)
print(target)

writer=SummaryWriter("dataloader")
for epoch in range(2):
    step=0
    for data in test_loader: #循环
        imgs,targets=data #batchsize=n,n个img一组，n个target一组
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step) #drop_last 最后除不尽batch_size 舍去 所以step会少1
        #这是 字符串格式化 的写法，把 epoch 变量的值嵌入到字符串里。
        step=step+1

writer.close()