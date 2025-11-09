import torch
import torchvision
from jinja2.optimizer import optimize
from torch import nn
# from model import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

#准备数据集
train_data=torchvision.datasets.CIFAR10(root="./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#length长度
train_data_size=len(train_data)
test_data_size=len(test_data)
#{}.format() 字符串格式化
print("训练数据集长度是:{}".format(train_data_size))
print("测试数据集长度是:{}".format(test_data_size))

#加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#模型实例化 创建网络模型
#搭建神经网络  十个类别  十分类   模块化
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

tudui=Tudui()
if torch.cuda.is_available():
    tudui=tudui.cuda()

#损失函数
loss_func=nn.CrossEntropyLoss()  #交叉熵
if torch.cuda.is_available():
    loss_func=loss_func.cuda()

#优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#设置训练的一些参数
#记录训练的次数
total_train_step=0
#记录测试的次数
total_test_step=0
#训练的轮次
epoch=10

#添加tensorboard
writer=SummaryWriter(log_dir="logs")
start_time=time.time()
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))

    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs,targets=data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        outputs=tudui(imgs)
        loss=loss_func(outputs,targets) #注意参数是targets

        # 优化器优化模型
        optimizer.zero_grad()#梯度清零
        loss.backward() #反向传播
        optimizer.step()#一步优化

        total_train_step=total_train_step+1
        if total_train_step % 100 == 0:
            end_time=time.time()
            print(end_time-start_time)
            print("训练次数：{}，Loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    tudui.eval()
    total_test_loss=0
    total_accuracy = 0
    with torch.no_grad():#梯度都没有
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs=tudui(imgs)
            loss=loss_func(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
        print("整体测试集上的Loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar('accuracy', total_accuracy / test_data_size, total_test_step)
    total_test_step+=1

    torch.save(tudui,"tudui_{}.pth".format(i))
    # torch.save(tudui.state_dict(),"tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()