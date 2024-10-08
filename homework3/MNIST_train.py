import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn
from pathlib import Path
import pickle
import gzip


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class WrappedDataLoader: #定义一种数据加载器的封装（可使用func进行预处理）
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

def preprocess(x,y): #数据预处理的函数
    return x.view(-1, 1, 28, 28).to(dev),y.to(dev)


def loss_batch(model, loss_func, xb, yb, opt=None): #对批次计算损失和更改权重
    loss = loss_func(model(xb), yb)

    if opt is not None: 
        loss.backward()#反向传播
        opt.step()     #沿梯度方向更新权重
        opt.zero_grad()# 梯度清零

    return loss.item(), len(xb) #返回损失和样本数


def fit(epochs, model, loss_func, opt, train_dl, valid_dl): #训练函数
    for epoch in range(epochs): #训练次数
        model.train() #设置为训练模式
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt) #对批次进行训练

        model.eval() #设置为评估模式
        with torch.no_grad(): 
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]  #此处opt=None，便是只计算损失，不修改梯度
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

bs=64
epochs=3
lr=0.1
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

PATH = Path(__file__).parent
DATA_PATH = PATH / "../homework2/data/mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((DATA_PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
train_ds = TensorDataset(x_train, y_train) #用tensordataset组合数据：训练集
valid_ds = TensorDataset(x_valid, y_valid) #用tensordataset组合数据：验证集
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
loss_func = F.cross_entropy #损失函数
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), #三层卷积和三个ReLU
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4), #平均池化
    Flatten()
)
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl) #训练
torch.save(model.state_dict(), PATH / "mnist.pth")

