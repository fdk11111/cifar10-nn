# cifar10 - nn

## 项目简介
`cifar10 - nn` 是一个专门针对 CIFAR - 10 数据集开发的图像分类项目。CIFAR - 10 数据集由 10 个不同类别的 60000 张 32x32 彩色图像组成，每类包含 6000 张图像。本项目构建了一个三层神经网络模型，通过对这些图像进行训练和学习，实现对图像的准确分类。

## 项目背景
图像分类是计算机视觉领域的基础任务之一，在安防监控、医学影像分析、自动驾驶等众多领域有着广泛的应用。CIFAR - 10 数据集作为一个经典的图像分类数据集，常被用于测试和验证新的图像分类算法和模型。本项目旨在通过实现一个简单的三层神经网络，帮助开发者理解和掌握基本的图像分类原理和流程。

## 项目结构
```
cifar10-nn/
├── data/               # 存放 CIFAR - 10 数据集
├── models/             # 存放训练好的模型文件
├── src/                # 源代码目录
│   ├── data_utils.py   # 数据加载和预处理模块
│   ├── model.py        # 定义三层神经网络模型
│   ├── train.py        # 模型训练脚本
│   ├── test.py         # 模型测试脚本
│   └── visualize.py    # 可视化工具脚本
├── requirements.txt    # 项目依赖库列表
└── README.md           # 项目说明文档
```

## 环境准备
### 依赖安装
在运行项目之前，需要安装必要的 Python 库。可以使用以下命令进行安装：
```bash
pip install -r requirements.txt
```
### 数据集下载
项目会在首次运行时自动下载 CIFAR - 10 数据集，并存放在 `data/` 目录下。如果你想手动下载数据集，可以从 [CIFAR - 10 官方网站](https://www.cs.toronto.edu/~kriz/cifar.html) 下载，并将其解压到 `data/` 目录。

## 代码运行步骤
### 模型训练
在项目根目录下，使用以下命令训练模型：
```bash
python src/train.py
```
训练过程中，模型的训练损失和准确率会实时显示在控制台，训练完成后，模型会保存到 `models/` 目录下。

### 模型测试
训练完成后，可以使用以下命令对模型进行测试：
```bash
python src/test.py
```
测试脚本会加载训练好的模型，并在测试集上进行评估，输出分类准确率等指标。

### 可视化
为了更直观地观察训练过程和模型效果，可以使用可视化脚本：
```bash
python src/visualize.py
```
该脚本会生成训练损失曲线、准确率曲线等可视化图表。

### 超参数调整
重要超参数及其影响
在训练三层神经网络时，以下超参数对模型性能起着关键作用：
学习率（Learning Rate）
学习率控制着模型参数在每次迭代中的更新步长。过大的学习率可能使模型跳过最优解，导致无法收敛；而过小的学习率会使模型收敛速度极为缓慢。常见的学习率取值范围为 0.001 - 0.1。在本项目中，我们可以先尝试 0.01 作为初始学习率，并观察训练过程中损失函数的变化。
批量大小（Batch Size）
批量大小指的是每次迭代中用于更新模型参数的样本数量。较大的批量大小能提高训练效率，但可能使模型陷入局部最优解；较小的批量大小则增加了模型的随机性，有助于跳出局部最优，但会降低训练速度。常见的批量大小取值有 32、64、128 等，我们可以从 64 开始进行尝试。
训练轮数（Epochs）
训练轮数表示整个数据集被模型训练的次数。训练轮数过少，模型可能无法充分学习数据特征；训练轮数过多，模型可能会出现过拟合现象。可以通过观察训练集和验证集的准确率及损失函数变化来确定合适的训练轮数，初始可设置为 20。
超参数调整方法
网格搜索（Grid Search）
网格搜索是一种穷举搜索方法，它会遍历所有可能的超参数组合，并选择在验证集上表现最优的组合。例如，我们可以定义如下超参数网格：
python
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30]
}
然后使用网格搜索算法对所有组合进行评估，以找到最优超参数组合。
随机搜索（Random Search）
随机搜索是在超参数空间中进行随机采样的方法，能在较短时间内找到较好的超参数组合。相较于网格搜索，它无需遍历所有可能的组合，因此效率更高。

## 模型评估
模型在测试集上的评估指标主要包括准确率（Accuracy）、精确率（Precision）、召回率（Recall）和 F1 值（F1 - Score）。这些指标可以帮助我们全面了解模型的性能。



## 贡献与反馈
如果你对本项目有任何改进建议或发现了 bug，欢迎提交 Issue 或 Pull Request。我们非常欢迎社区的贡献，共同完善这个项目。

## 许可证
本项目采用 [具体许可证名称] 许可证，详情请参阅 `LICENSE` 文件。

以上内容根据常见的图像分类项目需求进行编写，你可以根据实际情况进行调整和修改。如果能提供更多关于项目的细节，我可以生成更贴合实际的 README 文件。 
