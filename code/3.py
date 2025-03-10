# 构建神经网络模型
# 现在来学习如何使用PyTorch构建一个简单的神经网络模型。PyTorch提供了nn模块，它包含了构建神经网络所需的所有组件。

# nn.Module是PyTorch中所有神经网络模型的基类。它提供了一个统一的接口来管理模型参数和前向传播过程。
# 我们自定义的模型都应该继承这个类。


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子, 确保结果可以复现
torch.manual_seed(666)

# 来实现一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义线性层，输入输出维度为1
        self.linear = nn.Linear(1, 1)
    
    # 定义向前传播的过程
    def forward(self, x):
        return self.linear(x)
    


# 创建模型实例
model = LinearRegressionModel()
print(f"模型结构：\n{model}")

# 查看模型参数
for name, param in model.named_parameters():
    print(f"参数名: {name}, 形状: {param.shape}, 参数值: {param.data}")

# 生成一些模拟数据, 假设真实分布为 y = 2x + 3 + 噪声
x = torch.linspace(0, 10, 100).reshape(-1, 1)

# 添加一些随机噪声
y_true = 2 * x + 3 + 0.2 * torch.randn(x.shape)  

# 可视化数据
plt.figure(figsize=(10, 6))
plt.scatter(x.numpy(), y_true.numpy(), alpha=0.7)
plt.title("模拟的线性回归数据")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
# 真实数据分布可视化
plt.show()

# 测试模型未训练前的预测
with torch.no_grad():  # 不需要计算梯度
    y_pred = model(x)
    plt.plot(x.numpy(), y_pred.numpy(), 'r', linewidth=2, label='初始模型预测')
    plt.legend()
    # 来看看，未经过训练的模型“预测”的数据是什么样的
    plt.show()


