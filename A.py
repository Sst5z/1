import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear,Flatten,Sequential
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
train_data = torchvision.datasets.CIFAR10(root="data",  transform=torchvision.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False,num_workers=0,drop_last=False)


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

image,targe = train_data[1]
image = torch.reshape(image,(-1,3,32,32))
print(image.shape)
model= Model()
print(model)
output = model(image)
print(output)