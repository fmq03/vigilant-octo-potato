import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from pathlib import Path
import pickle
import gzip
import matplotlib.pyplot as plt


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Flatten()
)

# 加载数据
PATH = Path(__file__).parent
DATA_PATH = PATH / "../homework2/data/mnist"

FILENAME = "mnist.pkl.gz"
with gzip.open((DATA_PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=False)

# 检查设备
dev="cpu"
# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(dev)

# 加载训练好的模型参数
model.load_state_dict(torch.load(PATH / "mnist.pth"))
model.eval()  

# 推理函数
def infer(model, x):
    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output, dim=1)
    return predicted

# 显示预测结果的函数
def display_prediction(index):
    # 获取验证集的样本
    image, label = valid_ds[index]
    image = image.view(1, 1, 28, 28).to(dev)  # 调整形状以适应模型输入

    # 进行推理
    prediction = infer(model, image)

    # 显示图像及其预测结果
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    plt.title(f'Predicted: {prediction.item()}, Actual: {label.item()}')
    plt.axis('off')
    plt.show()

def calculate_accuracy(model, valid_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.view(-1, 1, 28, 28).to(dev)
            yb = yb.to(dev)
            outputs = model(xb)
            _, predicted = torch.max(outputs, dim=1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

    accuracy = correct / total
    return accuracy

# ex_x=torch.randn(1,1,28,28)
# torch.onnx.export(model,ex_x,PATH / 'mymodel.onnx')

accuracy = calculate_accuracy(model, valid_dl)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 选择验证集中某个索引进行显示
for i in np.random.randint(100,size=10):
    display_prediction(i)