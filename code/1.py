

# 先创建一个最简单的PyTorch程序，了解一下张量(Tensor)这个基本概念。
import torch

x = torch.tensor([1, 2, 3, 4])

random_tensor = torch.rand(3, 4)

y = x + 10

z = x * 2

print(y, z)