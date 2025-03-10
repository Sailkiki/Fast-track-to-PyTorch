# 现在进入了学习的关键环节：训练模型。在这一步，我们将定义损失函数和优化器，并通过多轮迭代来优化我们的模型参数。

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

# 定义损失函数 MSE-均方误差
criterion = nn.MSELoss()

# 定义随机梯度下降优化器(SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 100 # 轮数
losses = [] # 记录每轮的损失

print("开始训练..")
for epoch in range(epochs):
    # 向前传播
    y_pred = model(x)
    # 计算损失
    loss = criterion(y_pred, y_true)
    losses.append[loss.item()]

    # 反向传播
    optimizer.zero_grad() # 清除之前的梯度
    loss.backward() 
    optimizer.step()

    # 每20轮打印一次进度
    if (epoch+1) % 20 == 0:
        print(f'轮次: {epoch+1}/{epochs}, 损失: {loss.item():.4f}')

print("训练完成")


# 可视化结果
plt.figure(figsize=(12, 5))

# 绘制损失下降曲线
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('训练过程中的损失变化')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.grid(True)

# 绘制数据点和拟合线
plt.subplot(1, 2, 2)
plt.scatter(x.numpy(), y_true.numpy(), alpha=0.7, label='原始数据')

# 获取训练后的预测值
with torch.no_grad():
    y_pred = model(x)

plt.plot(x.numpy(), y_pred.numpy(), 'r', linewidth=2, label='训练后的模型')
plt.title('线性回归拟合结果')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()