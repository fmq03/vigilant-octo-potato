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

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10) #线性模型，输入784维，输出10维

    def forward(self, xb):
        return self.lin(xb)#前向传播（计算结果）

loss_func = F.cross_entropy #损失函数

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr) #返回模型和优化器


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
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
train_ds = TensorDataset(x_train, y_train) #用tensordataset组合数据：训练集
valid_ds = TensorDataset(x_valid, y_valid) #用tensordataset组合数据：验证集
train_dl, valid_dl = get_data(train_ds, valid_ds, bs) #用dataloader进行批次的管理，不用手动指定批次
model, opt = get_model() #获取模型和
fit(epochs, model, loss_func, opt, train_dl, valid_dl) #训练