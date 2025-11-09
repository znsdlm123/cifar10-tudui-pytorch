# CIFAR-10 图像分类项目

基于 PyTorch 的 CIFAR-10 图像分类项目，使用卷积神经网络（CNN）对 10 个类别的图像进行分类。

## 项目简介

本项目实现了一个完整的深度学习图像分类流程，包括：
- 数据加载和预处理
- CNN 模型定义（Tudui 模型）
- 模型训练（支持 CPU/GPU）
- 模型评估和保存
- 模型推理应用

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision tensorboard pillow
```

### 2. 训练模型

```bash
# CPU 训练
python train.py

# GPU 训练（推荐）
python train_gpu_2.py
```

### 3. 查看训练过程

```bash
tensorboard --logdir=logs
```

### 4. 模型推理

```bash
python src/test.py
```

## 核心文件

| 文件 | 说明 |
|------|------|
| `model.py` | 模型架构定义（Tudui CNN） |
| `train.py` | CPU 训练脚本 |
| `train_gpu_2.py` | GPU 训练脚本（推荐） |
| `src/test.py` | 模型推理测试 |
| `dataset_transform.py` | 数据集处理 |
| `dataloader.py` | 数据加载器 |

详细文件说明请查看 [核心文件说明.md](核心文件说明.md)

## 项目结构

```
├── model.py              # 模型定义
├── train.py              # CPU 训练
├── train_gpu_2.py        # GPU 训练（推荐）
├── dataset_transform.py  # 数据集处理
├── dataloader.py         # 数据加载
├── src/test.py           # 模型推理
├── data/                 # CIFAR-10 数据集
├── logs/                 # TensorBoard 日志
└── tudui_*.pth           # 训练好的模型
```

## 模型架构

```
输入 (3, 32, 32)
  ↓
Conv2d(3→32) + MaxPool2d
  ↓
Conv2d(32→32) + MaxPool2d
  ↓
Conv2d(32→64) + MaxPool2d
  ↓
Flatten
  ↓
Linear(1024→64)
  ↓
Linear(64→10)
  ↓
输出 (10类)
```

## 数据集

CIFAR-10 数据集包含 10 个类别的 32×32 彩色图像：
- 训练集：50,000 张
- 测试集：10,000 张

类别：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## 训练参数

- **损失函数：** CrossEntropyLoss
- **优化器：** SGD
- **学习率：** 0.01
- **批次大小：** 64
- **训练轮次：** 10 epochs

## 详细文档

- [完整项目文档](项目文档.md) - 详细的项目说明和使用指南
- [核心文件说明](核心文件说明.md) - 核心文件列表和功能说明

## 注意事项

1. 首次运行会自动下载 CIFAR-10 数据集
2. GPU 训练需要安装 CUDA 版本的 PyTorch
3. 模型文件保存在项目根目录（`tudui_0.pth` ~ `tudui_9.pth`）

## 许可证

本项目仅用于学习和研究目的。

