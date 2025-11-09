import torch
import torchvision.transforms
from PIL import Image  #永远用 from PIL import Image，简单、直观、官方文档就这么写。
from torch import nn

image_path="../img/airplane.PNG" # 文件层级 两个..

image=Image.open(image_path)
print(image)

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])

image=image.convert('RGB')  #png是四通道  转为RGB
image=transform(image)

print(image.shape)
#将图片映射到 gpu上
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# image = image.to(device)

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

model=torch.load("../tudui_0.pth")  #文件夹层级 两个点..退回到上一级
print(model)
image=torch.reshape(image,(1,3,32,32))  #  要输出单个图片   batch=1   b c h w 变成四维
model = torch.load("../tudui_0.pth", map_location=torch.device('cpu')) #模型原来是gpu保存的  要映射到cpu上
#良好代码习惯
model.eval() #验证
with torch.no_grad():
    output=model(image)
print(output)

print(output.argmax(1))  #横向   分类问题 输出 利于解读