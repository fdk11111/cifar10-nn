import numpy as np

class ThreeLayerNN:
    def __init__(self, input_size=3072, hidden_size=128, output_size=10, activation='relu'):
        """
        初始化神经网络参数
        参数：
            input_size: 输入维度 (CIFAR-10展平后为32x32x3=3072)
            hidden_size: 隐藏层神经元数量
            output_size: 输出类别数 (CIFAR-10为10)
            activation: 隐藏层激活函数 ('relu' 或 'sigmoid')
        """
        # 参数初始化 (He初始化)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # 激活函数配置
        self.activation = activation.lower()
        self.act_fn, self.act_deriv = self._get_activation()

    def _get_activation(self):
        """根据配置返回激活函数及其导数函数"""
        if self.activation == 'relu':
            return self._relu, self._relu_deriv
        elif self.activation == 'sigmoid':
            return self._sigmoid, self._sigmoid_deriv
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    @staticmethod
    def _relu(x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    @staticmethod
    def _relu_deriv(x):
        """ReLU导数 (x>0时为1，否则为0)"""
        return (x > 0).astype(x.dtype)

    @staticmethod
    def _sigmoid(x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_deriv(x):
        """Sigmoid导数"""
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)

    def forward(self, X):
        """
        前向传播计算
        输入：
            X: 输入数据 (batch_size, input_size)
        返回：
            probs: 输出概率 (batch_size, output_size)
        """
        # 第一层计算 (输入层 → 隐藏层)
        self.z1 = np.dot(X, self.W1) + self.b1  # (batch_size, hidden_size)
        self.a1 = self.act_fn(self.z1)          # 激活
        
        # 第二层计算 (隐藏层 → 输出层)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (batch_size, output_size)
        self.probs = self._softmax(self.z2)      # Softmax归一化
        return self.probs

    @staticmethod
    def _softmax(x):
        """数值稳定的Softmax函数"""
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def compute_loss(self, X, y, reg_lambda=0.01):
        """计算交叉熵损失 + L2正则化"""
        m = X.shape[0]  # 样本数
        probs = self.forward(X)
        
        # 交叉熵损失（添加1e-8防止log(0)）
        correct_logprobs = -np.log(probs[range(m), np.argmax(y, axis=1)] + 1e-8)
        data_loss = np.sum(correct_logprobs) / m
        
        # L2正则化项
        reg_loss = 0.5 * reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2)) / m
        return data_loss + reg_loss
    
    def backward(self, X, y, reg_lambda=0.01):
        """
        反向传播计算梯度
        输入：
            X: 输入数据 (batch_size, input_size)
            y: one-hot标签 (batch_size, output_size)
            reg_lambda: L2正则化强度
        返回：
            梯度字典：
                dW1: 第一层权重梯度
                db1: 第一层偏置梯度
                dW2: 第二层权重梯度
                db2: 第二层偏置梯度
        """
        m = X.shape[0]
        
        # 输出层梯度 (交叉熵 + softmax导数简化)
        delta3 = self.probs - y  # (batch_size, output_size)
        dW2 = (self.a1.T.dot(delta3) + reg_lambda * self.W2) / m
        db2 = np.sum(delta3, axis=0) / m
        
        # 隐藏层梯度
        delta2 = delta3.dot(self.W2.T) * self.act_deriv(self.z1)
        dW1 = (X.T.dot(delta2) + reg_lambda * self.W1) / m
        db1 = np.sum(delta2, axis=0) / m
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
# 测试模型初始化与前向传播
model = ThreeLayerNN(hidden_size=256)
dummy_input = np.random.randn(10, 3072)  # 10个样本
probs = model.forward(dummy_input)
assert probs.shape == (10, 10), "前向传播输出形状应为 (10, 10)"

# 测试反向传播梯度形状
dummy_labels = np.eye(10)[np.random.randint(0, 10, 10)]  # 随机生成one-hot标签
grads = model.backward(dummy_input, dummy_labels)
assert grads['dW1'].shape == model.W1.shape, "梯度dW1形状错误"
assert grads['dW2'].shape == model.W2.shape, "梯度dW2形状错误"