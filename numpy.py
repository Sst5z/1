import numpy as np
a = np.array([4,5,6])
print(type(a))
print(a.shape)
print(a[0])
b = np.array([[4,5,6],[1,2,3]])
print(b.shape)
print(b[0,0])
print(b[0,1])
print(b[1,1])
a=np.zeros([3,3],dtype=int) #全0矩阵，类型为整形
b=np.ones([4,5]) #全1矩阵
c=np.eye(4) #单位矩阵
d=np.random.rand(3,2) #随机数矩阵

a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a)
print(a[2,3],a[0,0])
b=a[0:2,:]
print(b)
b=b[:,2:4]
print(b) #把a的0到1行，2到3列，放到b里面去
print(b[0,0])
c=a[0:2]

 print(c)

a = np.array([[1,2],[3,4],[5,6]])
print(a[[0,1,2],[0,1,0]])

a = np.arange(1,13).reshape(4,3)
b = np.array([0,2,0,1])
print(a[[np.arange(4),b]]) # [ 1  6  7 11]

a[[np.arange(4),b]] += 10
var = a[[np.arange(4), b]]  # array([21, 26, 27, 31])

x = np.array([1,2])
x =np.array([1.0,2.0])
print(x.dtype) # dtype('float64')
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x + y)
np.add(x,y)
print(x-y)
np.subtract(x,y)
print(x * y) # 两个矩阵对应位置元素相乘
np.multiply(x,y) # 两个矩阵对应位置元素相乘
np.dot(x,y) # 矩阵相乘
print(x / y)
np.divide(x,y)
np.sqrt(x)
print(x.dot(y))
print(np.dot(x,y))
print(np.sum(x)) # 10
print(np.sum(x,axis=0)) # [4. 6.] 两列之和
print(np.sum(x,axis=1)) # [3. 7.] 两行之和
print(np.mean(x))
print(np.mean(x,axis=0))
print(np.mean(x,axis=1))
print(x.T)
print(x.T)
np.exp(x)
print(np.argmax(x))
print(np.argmax(x,axis=0))
print(np.argmax(x,axis=1))
import matplotlib.pyplot as plt
x = np.arange(0,100,0.1)
y = x * x
plt.figure(figsize=(6,6))
plt.plot(x,y)
plt.show()
x = np.arange(0,3*np.pi,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure(figsize=(10,6))
plt.plot(x,y1,color='Red')
plt.plot(x,y2,color='Blue')
plt.legend(['Sin','Cos'])
plt.show()