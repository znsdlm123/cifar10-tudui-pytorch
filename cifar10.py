import torchvision
d = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
print(d.classes)        # ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print(d.class_to_idx)   # {'airplane':0, 'automobile':1, ... 'truck':9}