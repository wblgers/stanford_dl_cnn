import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import optim


class CNN_net(nn.Module):
    def __init__(self):
        # 先运行nn.Module的初始化函数
        super(CNN_net, self).__init__()
        # 卷积层的定义，输入为1channel的灰度图，输出为4特征，每个卷积kernal为9*9
        self.conv = nn.Conv2d(1, 4, 9)
        # 均值池化
        self.pool = nn.AvgPool2d(2, 2)
        # 全连接后接softmax
        self.fc = nn.Linear(10*10*4, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # 卷积层，分别是二维卷积->sigmoid激励->池化
        out = self.conv(x)
        out = F.sigmoid(out)
        out = self.pool(out)
        print(out.size())
        # 将特征的维度进行变化(batchSize*filterDim*featureDim*featureDim->batchSize*flat_features)
        out = out.view(-1, self.num_flat_features(out))
        # 全连接层和softmax处理
        out = self.fc(out)
        out = self.softmax(out)
        return out
    def num_flat_features(self, x):
        # 四维特征，第一维是batchSize
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# 定义一个cnn网络
net = CNN_net()
print(net)

# 参数设置
learning_rate = 1e-3
batch_size = 100
epoches = 10

# MNIST图像数据的转换函数
trans_img = transforms.Compose([
        transforms.ToTensor()
    ])

# 下载MNIST的训练集和测试集
trainset = MNIST('./MNIST', train=True, transform=trans_img, download=True)
testset = MNIST('./MNIST', train=False, transform=trans_img, download=True)

# 构建Dataloader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# cost function的选用
criterian = nn.CrossEntropyLoss(size_average=False)

# 选用SGD来求解
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.95)

# 训练过程
for i in range(epoches):
    running_loss = 0.
    running_acc = 0.
    for (img, label) in trainloader:
        # 转换为Variable类型
        img = Variable(img)
        label = Variable(label)

        optimizer.zero_grad()

        # feedforward
        output = net(img)
        loss = criterian(output, label)
        # backward
        loss.backward()
        optimizer.step()

        # 记录当前的lost以及batchSize数据对应的分类准确数量
        running_loss += loss.data[0]
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        running_acc += correct_num.data[0]

    # 计算并打印训练的分类准确率
    running_loss /= len(trainset)
    running_acc /= len(trainset)

    print("[%d/%d] Loss: %.5f, Acc: %.2f" %(i+1, epoches, running_loss, 100*running_acc))

# 将当前模型设置到测试模式
net.eval()

# 测试过程
testloss = 0.
testacc = 0.
for (img, label) in testloader:
    # 转换为Variable类型
    img = Variable(img)
    label = Variable(label)

    # feedforward
    output = net(img)
    loss = criterian(output, label)

    # 记录当前的lost以及累加分类正确的样本数
    testloss += loss.data[0]
    _, predict = torch.max(output, 1)
    num_correct = (predict == label).sum()
    testacc += num_correct.data[0]

# 计算并打印测试集的分类准确率
testloss /= len(testset)
testacc /= len(testset)
print("Test: Loss: %.5f, Acc: %.2f %%" %(testloss, 100*testacc))
