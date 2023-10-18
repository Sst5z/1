my_tensor=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,device=device,requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)
# other common initialization methods
x=torch.empty(size =(3,3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3,3))
x=torch.eye(5,5)
x=torch.arange(start=0,end=5,step=1)
x=torch.linspace(start=0.1,end=1,steps=10)
x=torch.empty(size=(15)).normal_(mean=0,std=1)
x=torch.empty(size=(1,5)).uniform_(0,1)
x= torch.diag(torch.ones(3))
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
np_array = np.zeros((5,5))
tensor =torch.from_numpy(np_array)
np_array_back = tensor.numpy()



import torch
x=torch.tensor([1,2,3])
y=torch.tensor([9,8,7])
z1 = torch.empty(3)
torch.add(x,y,out=z1)
z2 =torch.add(x,y)
z=x+y
# Subtraction
z=x-y
# Division
z=torch.true_divide(x, y)
 # inplace operations
t =torch.zeros(3)
t.add_(x)
t+=x#t=t+x27
# Exponentiation29 z=x.pow(2)30 z=x** 231
# Simple comparison
z= x >0
z= x < 0
 # Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5,3))
x3=torch.mm(x1,x2) #
x3=x1.mm(x2)
 # matrix exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

# element wise mult.
z=x*y
print(z)
 # dot product
z=torch.dot(x, y)
print(z)
# Batch Matrix Multiplication
batch = 32
n =10
m=20
p=30

tensor1 =torch.rand((batch,n,m))
tensor2 = torch.rand((batch, m, p))
out_bmm =torch.bmm(tensor1tensor2)#(batchnp)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
z=x1-x2
z = x1 ** x2
 # other useful tensor operations
sum_x=torch.sum(xdim=0)
values, indices = torch.max(x, dim=0)
values,indices=torch.min(xdim=o)
abs_x=torch.abs(x)
z=torch.argmax(x,dim=0)
z=torch.argmin(xdim=0)
mean_x=torch.mean(x.float(),dim=0)
z=torch.eg(x,y)
sorted_y, indices =torch.sort(y,dim=o,descending=False )
z=torch.clamp(x,min=0)
x=torch.tensor([1,0,1,1.1],dtype=torch.bool)
z=torch.any(x)
z=torch.all(x)



import torch
batch_size =10
features =25
x= torch.rand((batch_size, features))

print(x[0].shape)# x[0,:]
print(x[:,0],shape)
print(x[2,0:10])
x[0,0] = 10018
 # Fancy indexing
x=torch.arange(10)
indices = [2,5,8]
print(x[indices])
x = torch.rand((3, 5))
rows=torch.tensor([10])
cols = torch.tensor([4,0])
print(x[rows,cols].shape)
# More advanced indexing
x=torch.arange(10)
print(x[(x <2) & (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(torch.where(x > 5,x,x*2))
print(torch.tensor([0,0.1,2,2,3,4]).unique())
print(x.ndimension())# 5x5x5
print(x.numel())
x=torch.arange(9)
x_3x3 =x.view(3,3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)
y=x_3x3.t()#[0，3，6，1，4，7，2，5，8]
print(y.contiguous0.view(9))
x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2),dim=0).shape)
print(torch.cat((x1,x2),dim=1).shape)
z =x1.view(-1)
print(z.shape)
batch =64
x=torch.rand((batch,2,5))
z=x.view(batch,-1)
print(z.shape)

z=x.permute(0,2,1)
print(z.shape)

x=torch.arange(10)# [10]
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
x=torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10
z=x.squeeze(1)
print(z.shape)