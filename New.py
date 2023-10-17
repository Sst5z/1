import numpy as np

a = np.array([4, 5, 6])
print(type(a))
print(a.shape)
print(a[0])
b = np.array([[4, 5, 6], [1, 2, 3]])
print(b.shape)
print(b[0, 0])
print(b[0, 1])
print(b[1, 1])
a = np.zeros([3, 3], dtype=int)  # 全0矩阵，类型为整形
b = np.ones([4, 5])  # 全1矩阵
c = np.eye(4)  # 单位矩阵
d = np.random.rand(3, 2)  # 随机数矩阵

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
print(a[2, 3], a[0, 0])
b = a[0:2, :]
print(b)
b = b[:, 2:4]
print(b)  # 把a的0到1行，2到3列，放到b里面去
print(b[0, 0])
c = a[0:2]

print(c)

a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])

a = np.arange(1, 13).reshape(4, 3)
b = np.array([0, 2, 0, 1])
print(a[[np.arange(4), b]])  # [ 1  6  7 11]

a[[np.arange(4), b]] += 10
var = a[[np.arange(4), b]]  # array([21, 26, 27, 31])

x = np.array([1, 2])
x = np.array([1.0, 2.0])
print(x.dtype)  # dtype('float64')
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x + y)
np.add(x, y)
print(x - y)
np.subtract(x, y)
print(x * y)  # 两个矩阵对应位置元素相乘
np.multiply(x, y)  # 两个矩阵对应位置元素相乘
np.dot(x, y)  # 矩阵相乘
print(x / y)
np.divide(x, y)
np.sqrt(x)
print(x.dot(y))
print(np.dot(x, y))
print(np.sum(x))  # 10
print(np.sum(x, axis=0))  # [4. 6.] 两列之和
print(np.sum(x, axis=1))  # [3. 7.] 两行之和
print(np.mean(x))
print(np.mean(x, axis=0))
print(np.mean(x, axis=1))
print(x.T)
print(x.T)
np.exp(x)
print(np.argmax(x))
print(np.argmax(x, axis=0))
print(np.argmax(x, axis=1))
import matplotlib.pyplot as plt

x = np.arange(0, 100, 0.1)
y = x * x
plt.figure(figsize=(6, 6))
plt.plot(x, y)
plt.show()
x = np.arange(0, 3 * np.pi, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y1, color='Red')
plt.plot(x, y2, color='Blue')
plt.legend(['Sin', 'Cos'])
plt.show()

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)
# other common initialization methods
x = torch.empty(size=(3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(5, 5)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(15)).normal_(mean=0, std=1)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
x = torch.diag(torch.ones(3))
# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short0)
print(tensor.long())
print(tensor.half())
print(tensor.float(()))
print(tensor.double0)
# Array to Tensor conversion and vice-versa
import numpy as np

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])
z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z = x + y
# Subtraction
z = x - y
# Division
z = torch.true_divide(x, y)
# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x  # t=t+x27
# Exponentiation29 z=x.pow(2)30 z=x** 231
# Simple comparison
z = x > 0
z = x < 0
# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)  #
x3 = x1.mm(x2)
# matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# element wise mult.
z = x * y
print(z)
# dot product
z = torch.dot(x, y)
print(z)
# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1tensor2)  # (batchnp)

# Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2
z = x1 ** x2
# other useful tensor operations
sum_x = torch.sum(xdim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(xdim=o)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(xdim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eg(x, y)
sorted_y, indices = torch.sort(y, dim=o, descending=False)
z = torch.clamp(x, min=0)
x = torch.tensor([1, 0, 1, 1.1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x)

import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)  # x[0,:]
print(x[:, 0], shape)
print(x[2, 0:10])
x[0, 0] = 10018
# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])
x = torch.rand((3, 5))
rows = torch.tensor([10])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)
# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) & (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(torch.where(x > 5, x, x * 2))
print(torch.tensor([0, 0.1, 2, 2, 3, 4]).unique())
print(x.ndimension())  # 5x5x5
print(x.numel())
x = torch.arange(9)
x_3x3 = x.view(3, 3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)
y = x_3x3.t()  # [0，3，6，1，4，7，2，5，8]
print(y.contiguous0.view(9))
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)
z = x1.view(-1)
print(z.shape)
batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)
print(z.shape)

x = torch.arange(10)  # [10]
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
z = x.squeeze(1)
print(z.shape)

import torch

w1, w2, w3, w4, w5, w6, w7, w8 = 0.2, -0.4, 0.5, 0.6, 0.1, -0.5, -0.3, 0.8
x1, x2 = 0.5, 0.3
h1 = torch.tensor(w1 * x1 + w3 * x2)
h1 = torch.sigmoid(h1)
h2 = torch.tensor(w2 * x1 + w4 * x2)
h2 = torch.sigmoid(h2)
o1 = torch.tensor(w5 * h1 + w7 * h2)
o1 = torch.sigmoid(o1)
o2 = torch.tensor(w6 * h1 + w8 * h2)
o2 = torch.sigmoid(o2)
print(o1, o2)

import numpy as np
import torch
import torch.nn as nn


def forward(x, w1, w2):
    net1 = nn.Linear(2, 2)
    net1.weight.data = w1
    net1.bias.data = torch.Tensor([0])
    h = net1(x)
    h = torch.sigmoid(h)
    net2 = nn.Linear(2, 2)
    net2.weight.data = w2
    net2.bias.data = torch.Tensor([0])
    o = net2.forward(x)
    o = torch.sigmoid(o)
    return o


if __name__ == '__main__':
    x = torch.tensor([0.5, 0.3])
    w1 = torch.tensor([[0.2, 0.5], [-0.4, 0.6]])
    w2 = torch.tensor([[0.1, -0.3], [-0.5, 0.8]])
    output = forward(x, w1, w2)
print('最终的输出值是：', output)

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, w1, w2):
        self.fc1.weight.data = w1
        self.fc1.bias.data = torch.Tensor([0])
        h = self.fc1(x)
        h = self.sigmoid(h)
        self.fc2.weight.data = w2
        self.fc2.bias.data = torch.Tensor([0])
        o = self.fc2(h)
        o = self.sigmoid(o)
        return o


if __name__ == '__main__':
    x = torch.tensor([0.5, 0.3])
    w1 = torch.tensor([[0.2, 0.5], [-0.4, 0.6]])
    w2 = torch.tensor([[0.1, -0.3], [-0.5, 0.8]])

    net = Net(2, 2, 2)
    output = net(x, w1, w2)
    # net.forward(x,w1,w2)
    print('最终的输出值为：', output)

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, w1, w2):
        self.fc1.weight.data = w1
        self.fc1.bias.data = torch.Tensor([0])
        out = self.fc1(x)
        out = self.sigmoid(out)

        self.fc2.weight.data = w2
        self.fc2.bias.data = torch.Tensor([0])
        output = self.fc2(out)
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    x = torch.tensor([0.5, 0.3])
    y = torch.tensor([0.23, 0.07])
    w1 = torch.tensor([[0.2, 0.5], [-0.4, 0.6]])
    w2 = torch.tensor([[0.1, -0.3], [-0.5, 0.8]])
    net = Net(2, 2, 2)

    loss = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.SGD(params=net.parameters(), lr=1e-2)  # 定义优化器

    for i in range(1000):
        output = net(x, w1, w2)
        loss_fn = loss(output, y)
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
        print('损失函数的变化情况', i, loss_fn)
print(output)

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root="data", transform=torchvision.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


image, targe = train_data[1]
image = torch.reshape(image, (-1, 3, 32, 32))
print(image.shape)
model = Model()
print(model)
output = model(image)
print(output)