# %%
import numpy as np
import time
import os
from three_layer_nn import ThreeLayerNN
from data_preprocess import train_images, train_labels, val_images, val_labels, test_images, test_labels

def train_model(model, X_train, y_train, X_val, y_val, 
                lr=0.01, reg_lambda=0.01, batch_size=64, 
                epochs=50, decay=0.95, patience=3,
                save_dir="models"):
    """
    训练神经网络并监控验证集性能
    参数：
        model: 初始化的模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        lr: 初始学习率
        reg_lambda: L2正则化强度
        batch_size: 批次大小
        epochs: 最大迭代次数
        decay: 学习率衰减系数
        patience: 早停耐心值（连续patience次验证损失不下降则停止）
        save_dir: 模型保存路径
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录训练过程
    history = {
        'train_loss': [], 
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    current_lr = lr
    
    # 开始训练
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.0
        
        # 随机打乱数据
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # 分批训练
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # 前向传播
            probs = model.forward(X_batch)
            
            # 计算损失
            loss = model.compute_loss(X_batch, y_batch, reg_lambda)
            train_loss += loss * len(X_batch)
            
            # 反向传播
            grads = model.backward(X_batch, y_batch, reg_lambda)
            
            # SGD参数更新
            for param in ['W1', 'b1', 'W2', 'b2']:
                model.__dict__[param] -= current_lr * grads[f'd{param}']
        
        # 计算验证集指标
        val_probs = model.forward(X_val)
        val_loss = model.compute_loss(X_val, y_val, reg_lambda)
        val_acc = np.mean(np.argmax(val_probs, axis=1) == np.argmax(y_val, axis=1))
        
        # 记录历史
        history['train_loss'].append(train_loss / len(X_train))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 学习率衰减
        current_lr *= decay
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            # 保存最佳模型
            np.savez(os.path.join(save_dir, f'best_model.npz'), 
                     W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"早停触发于第 {epoch+1} 轮")
                break
        
        # 打印进度
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}% | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {current_lr:.4f}")
    
    return history, model

# %%
# 初始化模型
model = ThreeLayerNN(hidden_size=256, activation='relu')

# 启动训练（示例参数）
history, trained_model = train_model(
    model, 
    train_images, train_labels,  
    val_images, val_labels,      
    lr=0.01,
    reg_lambda=0.001,
    batch_size=64,
    epochs=50,
    save_dir="cifar_models"
)

# %%
# 初始化模型
model = ThreeLayerNN(hidden_size=256, activation='relu')

# 启动训练
history, trained_model = train_model(
    model, 
    train_images, train_labels,  
    val_images, val_labels,      
    lr=0.01,
    reg_lambda=0.001,
    batch_size=64,
    epochs=150,
    save_dir="cifar_models"
)

# %%



