import torch
w1,w2,w3,w4,w5,w6,w7,w8=0.2,-0.4,0.5,0.6,0.1,-0.5,-0.3,0.8
x1,x2=0.5,0.3
h1=torch.tensor(w1*x1+w3*x2)
h1=torch.sigmoid(h1)
h2=torch.tensor(w2*x1+w4*x2)
h2=torch.sigmoid(h2)
o1=torch.tensor(w5*h1+w7*h2)
o1=torch.sigmoid(o1)
o2=torch.tensor(w6*h1+w8*h2)
o2=torch.sigmoid(o2)
print(o1,o2)

import numpy as np
import torch
import torch.nn as nn
def forward(x,w1,w2):
    net1=nn.Linear(2,2)
    net1.weight.data=w1
    net1.bias.data=torch.Tensor([0])
    h=net1(x)
    h=torch.sigmoid(h)
    net2=nn.Linear(2,2)
    net2.weight.data=w2
    net2.bias.data=torch.Tensor([0])
    o=net2.forward(x)
    o=torch.sigmoid(o)
    return o
if __name__=='__main__':
    x=torch.tensor([0.5,0.3])
    w1=torch.tensor([[0.2,0.5],[-0.4,0.6]])
    w2=torch.tensor([[0.1,-0.3],[-0.5,0.8]])
    output=forward(x,w1,w2)
print('最终的输出值是：',output)


import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size) -> None:
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.sigmoid=torch.nn.Sigmoid()
        self.fc2=nn.Linear(hidden_size,output_size)

    def forward(self,x,w1,w2):
        self.fc1.weight.data=w1
        self.fc1.bias.data = torch.Tensor([0])
        h=self.fc1(x)
        h=self.sigmoid(h)
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
    def __init__(self,input_size,hidden_size,output_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size,output_size)
    def forward(self,x,w1,w2):
        self.fc1.weight.data = w1
        self.fc1.bias.data = torch.Tensor([0])
        out = self.fc1(x)
        out = self.sigmoid(out)

        self.fc2.weight.data=w2
        self.fc2.bias.data = torch.Tensor([0])
        output = self.fc2(out)
        output=self.sigmoid(output)
        return output

if __name__=='__main__':
    x = torch.tensor([0.5, 0.3])
    y = torch.tensor([0.23, 0.07])
    w1 = torch.tensor([[0.2, 0.5], [-0.4, 0.6]])
    w2 = torch.tensor([[0.1, -0.3], [-0.5, 0.8]])
    net = Net(2,2,2)

    loss = nn.MSELoss()  #定义损失函数
    optimizer = torch.optim.SGD(params=net.parameters(),lr=1e-2) #定义优化器

    for i in range(1000):
        output = net(x,w1,w2)
        loss_fn = loss(output,y)
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
        print('损失函数的变化情况',i,loss_fn)
print(output)