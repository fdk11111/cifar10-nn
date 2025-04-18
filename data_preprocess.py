import numpy as np
import pickle
import os

def load_cifar10_batch(file_path):
    """加载单个CIFAR-10批次文件（无依赖）"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        # 手动转换图像格式：原始数据为3072维向量，转换为32x32x3
        images = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch['labels'])
    return images, labels

# 指定数据集路径
dataset_path = r'C:\Users\fengdingkang\Desktop\作业\神经网络\homework1\cifar-10-python (1)\cifar-10-batches-py'

# --- 手动加载所有数据 ---
# 加载训练批次
train_images, train_labels = [], []
for i in range(1, 6):
    batch_file = os.path.join(dataset_path, f'data_batch_{i}')
    images, labels = load_cifar10_batch(batch_file)
    train_images.append(images)
    train_labels.append(labels)
train_images = np.concatenate(train_images, axis=0)  # (50000, 32, 32, 3)
train_labels = np.concatenate(train_labels, axis=0)   # (50000,)

# 加载测试集
test_file = os.path.join(dataset_path, 'test_batch')
test_images, test_labels = load_cifar10_batch(test_file)

# --- 手动预处理 ---
# 归一化到[0,1]
train_images = (train_images.astype(np.float32) / 255.0).reshape(-1, 3072)  # (50000, 3072)
test_images = (test_images.astype(np.float32) / 255.0).reshape(-1, 3072)    # (10000, 3072)

# 手动One-hot编码标签
def manual_one_hot(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.0
    return one_hot

train_labels = manual_one_hot(train_labels)  # (50000, 10)
test_labels = manual_one_hot(test_labels)    # (10000, 10)

# --- 手动分层划分验证集（无需sklearn） ---
def manual_stratified_split(images, labels, val_ratio=0.1):
    """
    手动实现分层抽样
    返回: 
        train_images, val_images, train_labels, val_labels
    """
    np.random.seed(42)  # 固定随机种子
    val_indices = []
    
    # 按类别收集索引
    for cls in range(10):
        # 找到当前类别的所有样本索引
        cls_indices = np.where(labels[:, cls] == 1)[0]
        # 打乱顺序
        np.random.shuffle(cls_indices)
        # 计算验证集样本数
        n_val = int(len(cls_indices) * val_ratio)
        # 取前n_val个作为验证集
        val_indices.extend(cls_indices[:n_val])
    
    # 打乱验证集索引顺序
    np.random.shuffle(val_indices)
    
    # 生成掩码分离数据
    mask = np.ones(len(images), dtype=bool)
    mask[val_indices] = False
    
    # 分离训练集和验证集
    return images[mask], images[~mask], labels[mask], labels[~mask]

# 划分验证集（10%）
train_images, val_images, train_labels, val_labels = manual_stratified_split(train_images, train_labels)

# --- 验证数据形状 ---
print("训练集图像:", train_images.shape)  # (45000, 3072)
print("训练集标签:", train_labels.shape)  # (45000, 10)
print("验证集图像:", val_images.shape)    # (5000, 3072)
print("验证集标签:", val_labels.shape)    # (5000, 10)
print("测试集图像:", test_images.shape)   # (10000, 3072)
print("测试集标签:", test_labels.shape)    # (10000, 10)