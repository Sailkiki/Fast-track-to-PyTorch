# 在来学习PyTorch最强大的特性之一：自动微分（Autograd）。神经网络训练的核心是反向传播算法，而PyTorch的自动微分功能让我们不必手动计算复杂的导数。

import torch

# 创建需要计算梯度的张量
x = torch.tensor([2.0], requires_grad=True)
print(x) # tensor([2.], requires_grad=True)

# 定义一个表达式
y = x * x + 2
y.backward()
print(f"x的梯度(dy / dx): {x.grad}" ) # 4 (2 * x  = 4)

# 更复杂的例子
a = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
b = torch.tensor([1.0, 4.0, 7.0], requires_grad=True)
c = a * a + b * b
d = c.mean()

# a * a = [4.0, 9.0] (2² = 4, 3² = 9)
# b * b = [1.0, 16.0] (1² = 1, 4² = 16)
# c = [5.0, 25.0] (4+1 = 5, 9+16 = 25)
# d = (5.0 + 25.0) / 2 = 15.0
print(d)  # tensor(15., grad_fn=<MeanBackward0>)

d.backward()
# tensor([2., 3.])
# tensor([1., 4.])
# 对于a，梯度是∂d/∂a = 2a/n（其中n是元素个数）
# 对于b，梯度是∂d/∂b = 2b/n
print(a.grad)
print(b.grad)
