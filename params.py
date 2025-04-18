# %%
# %% 超参数搜索与实验管理
import itertools
import json
import os
import numpy as np
from three_layer_nn import ThreeLayerNN
from data_preprocess import train_images, train_labels, val_images, val_labels, test_images, test_labels
from datetime import datetime
from training import train_model
import time

param_combinations = [
    {'hidden_size':256, 'lr':0.01, 'reg_lambda':0.001},  # 基线配置
    {'hidden_size':512, 'lr':0.005, 'reg_lambda':0.005},  # 增大模型+正则化
    {'hidden_size':128, 'lr':0.02, 'reg_lambda':0.0005},  # 小模型+高学习率
    {'hidden_size':512, 'lr':0.001, 'reg_lambda':0.01}    # 大模型+强正则
]

# 快速测试函数
def quick_train_test(params, epochs=30):
    """快速训练验证（减少epochs）"""
    model = ThreeLayerNN(hidden_size=params['hidden_size'], activation='relu')
    
    # 缩短训练时间：减少epochs，增大batch_size
    history, _ = train_model(
        model,
        train_images, train_labels,
        val_images, val_labels,
        lr=params['lr'],
        reg_lambda=params['reg_lambda'],
        batch_size=128,  # 增大batch加速训练
        epochs=epochs,
        decay=0.95,
        patience=2  # 减少早停耐心
    )
    
    # 快速验证测试集
    test_probs = model.forward(test_images)
    test_acc = np.mean(np.argmax(test_probs, 1) == np.argmax(test_labels, 1))
    
    return {
        'val_acc': max(history['val_acc']),
        'test_acc': test_acc,
        'params': params
    }

# %% 运行所有参数组合（预计总耗时：基线时间×组合数）
results = []
for params in param_combinations:
    print(f"\n=== Testing {params} ===")
    start_time = time.time()
    
    result = quick_train_test(params, epochs=30)
    results.append(result)
    
    print(f"Val Acc: {result['val_acc']:.2%} | Test Acc: {result['test_acc']:.2%}")
    print(f"Time: {time.time()-start_time:.1f}s")

# %% 结果展示
print("\n=== Final Results ===")
for i, res in enumerate(sorted(results, key=lambda x: -x['test_acc'])):
    print(f"Rank{i+1}: Test Acc {res['test_acc']:.2%} | Params {res['params']}")

# %% 保存最佳模型（手动选择最优参数后重新训练完整epochs）
best_params = results[0]['params']
print(f"\nRetraining best config: {best_params}")

final_model = ThreeLayerNN(
    hidden_size=best_params['hidden_size'],
    activation='relu'
)

# 完整训练（使用原参数设置）
history, _ = train_model(
    final_model,
    train_images, train_labels,
    val_images, val_labels,
    lr=best_params['lr'],
    reg_lambda=best_params['reg_lambda'],
    batch_size=64,  # 恢复原batch_size
    epochs=100,
    save_dir="final_model"
)

# %% 最终测试
test_probs = final_model.forward(test_images)
final_acc = np.mean(np.argmax(test_probs, 1) == np.argmax(test_labels, 1))
print(f"\n=== Final Test Accuracy: {final_acc:.2%} ===")

# %%
import itertools
import json
import os
import numpy as np
from three_layer_nn import ThreeLayerNN
from data_preprocess import train_images, train_labels, val_images, val_labels, test_images, test_labels
from datetime import datetime
from training import train_model
import time

# 定义要测试的关键参数组合
param_combinations = [
    {'hidden_size':256, 'lr':0.01, 'reg_lambda':0.001},  # 基线配置
    {'hidden_size':512, 'lr':0.005, 'reg_lambda':0.005},  # 增大模型+正则化
    {'hidden_size':128, 'lr':0.02, 'reg_lambda':0.0005},  # 小模型+高学习率
    {'hidden_size':512, 'lr':0.001, 'reg_lambda':0.01}    # 大模型+强正则
]

# 快速测试函数
def quick_train_test(params, epochs=30):
    """快速训练验证（减少epochs）"""
    model = ThreeLayerNN(hidden_size=params['hidden_size'], activation='relu')
    
    # 缩短训练时间：减少epochs，增大batch_size
    history, _ = train_model(
        model,
        train_images, train_labels,
        val_images, val_labels,
        lr=params['lr'],
        reg_lambda=params['reg_lambda'],
        batch_size=128,  # 增大batch加速训练
        epochs=epochs,
        decay=0.95,
        patience=2  # 减少早停耐心
    )
    
    # 快速验证测试集
    test_probs = model.forward(test_images)
    test_acc = np.mean(np.argmax(test_probs, 1) == np.argmax(test_labels, 1))
    
    return {
        'val_acc': max(history['val_acc']),
        'test_acc': test_acc,
        'params': params
    }

# %% 运行所有参数组合（预计总耗时：基线时间×组合数）
results = []
for params in param_combinations:
    print(f"\n=== Testing {params} ===")
    start_time = time.time()
    
    result = quick_train_test(params, epochs=30)
    results.append(result)
    
    print(f"Val Acc: {result['val_acc']:.2%} | Test Acc: {result['test_acc']:.2%}")
    print(f"Time: {time.time()-start_time:.1f}s")

# %% 结果展示
print("\n=== Final Results ===")
for i, res in enumerate(sorted(results, key=lambda x: -x['test_acc'])):
    print(f"Rank{i+1}: Test Acc {res['test_acc']:.2%} | Params {res['params']}")

# %% 保存最佳模型（手动选择最优参数后重新训练完整epochs）
best_params = results[0]['params']
print(f"\nRetraining best config: {best_params}")

final_model = ThreeLayerNN(
    hidden_size=best_params['hidden_size'],
    activation='relu'
)


history, _ = train_model(
    final_model,
    train_images, train_labels,
    val_images, val_labels,
    lr=['lr'],
    reg_lambda=best_params['reg_lambda'],
    batch_size=64,  # 恢复原batch_size
    epochs=100,
    save_dir="final_model"
)

# %% 最终测试
test_probs = final_model.forward(test_images)
final_acc = np.mean(np.argmax(test_probs, 1) == np.argmax(test_labels, 1))
print(f"\n=== Final Test Accuracy: {final_acc:.2%} ===")

# %%
# %% 独立测试脚本 test_model.py
import numpy as np
import os
from three_layer_nn import ThreeLayerNN
from data_preprocess import test_images, test_labels

def load_trained_model(model_path):
    """加载预训练模型"""
    # 从路径解析参数（示例参数，需要根据实际训练配置修改）
    params = {
        'hidden_size': 256,  # 必须与训练时的hidden_size一致
        'activation': 'relu'
    }
    
    # 初始化模型结构
    model = ThreeLayerNN(**params)
    
    # 加载权重（完整路径拼接）
    weight_path = os.path.join(model_path, 'best_model.npz')
    weights = np.load(weight_path)
    model.W1 = weights['W1']
    model.b1 = weights['b1']
    model.W2 = weights['W2']
    model.b2 = weights['b2']
    
    return model

# %% 执行测试
if __name__ == "__main__":
    # 需要测试的模型路径列表（Windows路径使用双反斜杠）
    model_paths = [
        r"C:\Users\fengdingkang\Desktop\作业\神经网络\homework1\models",  # 你提供的路径
        "cifar_models",         # 之前的基线模型
        "final_model"           # 完整训练模型
    ]
    
    for path in model_paths:
        # 检查模型文件是否存在
        model_file = os.path.join(path, 'best_model.npz')
        if not os.path.exists(model_file):
            print(f"警告：跳过不存在的模型路径 {path}")
            continue
            
        print(f"\n=== 正在测试模型 [{path}] ===")
        try:
            model = load_trained_model(path)
            
            # 前向传播
            probs = model.forward(test_images)
            predictions = np.argmax(probs, axis=1)
            true_labels = np.argmax(test_labels, axis=1)
            
            # 计算准确率
            accuracy = np.mean(predictions == true_labels)
            print(f"测试准确率: {accuracy:.2%}")
            
        except Exception as e:
            print(f"测试失败：{str(e)}")

# %%



