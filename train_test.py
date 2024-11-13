import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()  # e，并保存到self.data中
        self.labels = torch.from_numpy(labels).long()  # 将输入的labels转换为PyTorch的Tensor类型，并保存到self.labels中

    def __len__(self):
        return len(self.data)  # 返回self.data的长度，即数据集的大小

    def __getitem__(self, index):
        x = self.data[index]  # 获取索引为index的数据
        y = self.labels[index]  # 获取索引为index的标签
        return x, y  # 返回数据和标签


# 加载数据
data = pd.read_csv('dataset/NAVLR_rawdata_310.csv')
features = data.iloc[:, :-1].values # 选择所有行，和除了最后一列之外的数据
labels = data.iloc[:, -1].values # 选择所有行 以及它们的最后一列

"""
这段代码用于创建训练集和测试集的数据加载器。
使用CustomDataset类创建了训练集和测试集的自定义数据集对象，
其中X_train是训练数据，y_train是对应的标签；X_test是测试数据，y_test是对应的标签。
然后使用DataLoader类创建了训练集和测试集的数据加载器。
指定了批量大小为32，表示每次迭代从数据集中加载32个样本。
设置了shuffle=True来打乱训练集数据，以增加随机性和泛化能力；
对于测试集数据加载器，我们设置了shuffle=False，保持测试数据的原始顺序。
"""
# 划分训练集和测试集
'''
X_train:训练集数据
X_test：测试集数据
y_train：测试集标签
y_test：测试集标签
'''

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=28)

# 创建训练集和测试集的自定义数据集对象，传入训练数据X_train和标签y_train
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# 创建训练集和测试集的数据加载器，设定批量大小为32，并打乱训练集数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 32 64 128
class CNNModelWithAttention(nn.Module):
    def __init__(self):
        super(CNNModelWithAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, padding=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, padding=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, padding=3)
        self.fc1 = nn.Linear(10240, 8)
        self.dropout = nn.Dropout(0.52)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool1d(kernel_size=2)(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool1d(kernel_size=2)(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        return x


# 创建模型实例和优化器
model = CNNModelWithAttention()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# 定义训练函数
def train(model, device, train_loader, optimizer, criterion, scheduler=None):
    model.train()  # 将模型设置为训练模式
    train_loss = 0  # 初始化训练损失为0
    train_correct = 0  # 初始化训练正确预测的样本数为0

    for inputs, labels in train_loader:  # 遍历训练数据集
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到指定的设备（例如GPU）上
        optimizer.zero_grad()  # 清零优化器的梯度信息
        # outputs = model(inputs)  # 输入数据进行前向传播，得到模型的输出
        #在这里拓展维度，修改模型的输入维度为64 300 1
        outputs = model(inputs.unsqueeze(1))  # 输入数据进行前向传播，得到模型的输出
        loss = criterion(outputs, labels)  # 计算模型输出与标签之间的损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        train_loss += loss.item()  # 累加训练损失
        _, predicted = torch.max(outputs.data, 1)  # 在模型输出中找到最大值对应的索引作为预测结果
        train_correct += (predicted == labels).sum().item()  # 累加训练正确预测的样本数
    # 返回训练损失和准确率
    conv1_weights = model.conv1.weight
    conv1_bias = model.conv1.bias
    conv2_weights = model.conv2.weight
    conv2_bias = model.conv2.bias
    conv3_weights = model.conv3.weight
    conv3_bias = model.conv3.bias
    fc_weights = model.fc1.weight
    fc_bias = model.fc1.bias
    return train_loss, train_correct, conv1_weights, conv1_bias, conv2_weights, conv2_bias, conv3_weights, conv3_bias, fc_weights, fc_bias


# 定义测试函数
def test(model, device, test_loader):
    model.eval()  # 将模型设置为评估模式
    test_loss = 0  # 初始化测试损失为0
    test_correct = 0  # 初始化测试正确预测的样本数为0
    with torch.no_grad():  # 在评估过程中不需要计算梯度，使用torch.no_grad()上下文管理器
        for inputs, labels in test_loader:  # 遍历测试数据集
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到指定的设备（例如GPU）上
            # outputs = model(inputs)  # 输入数据进行前向传播，得到模型的输出
            outputs = model(inputs.unsqueeze(1))  # 输入数据进行前向传播，得到模型的输出
            loss = criterion(outputs, labels)  # 计算模型输出与标签之间的损失
            test_loss += loss.item()  # 累加测试损失
            _, predicted = torch.max(outputs.data, 1)  # 在模型输出中找到最大值对应的索引作为预测结果
            test_correct += (predicted == labels).sum().item()  # 累加测试正确预测的样本数
    return test_loss, test_correct

if __name__ == '__main__':
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到设备上
    model.to(device)
    # 训练和测试模型
    epochs = 15
    # 定义空列表来保存训练和测试的损失和准确率
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []


    for epoch in range(1, epochs + 1):
        (train_loss, train_correct, conv1_weights, conv1_bias,
         conv2_weights, conv2_bias, conv3_weights, conv3_bias,
         fc_weights, fc_bias)= train(model, device, train_loader, optimizer, criterion)
        test_loss, test_correct = test(model, device, test_loader)
        # 计算训练和测试的准确率
        train_accuracy = (train_correct / len(train_dataset)) * 100
        test_accuracy = (test_correct / len(test_dataset)) * 100
        # 训练和测试的损失和准确率
        train_losses.append(train_loss / len(train_dataset))
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss / len(test_dataset))
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss / len(train_dataset):.4f}, Train Accuracy: {(train_correct / len(train_dataset)) * 100:.2f}%")
        print(f"Test Loss: {test_loss / len(test_dataset):.4f}, Test Accuracy: {(test_correct / len(test_dataset)) * 100:.2f}%")
        if epoch == 15:
            print(f"Epoch is {epoch}")

            conv1_weights = conv1_weights.detach().cpu().numpy()
            conv1_bias = conv1_bias.detach().cpu().numpy()
            conv2_weights = conv2_weights.detach().cpu().numpy()
            conv2_bias = conv2_bias.detach().cpu().numpy()
            conv3_weights = conv3_weights.detach().cpu().numpy()
            conv3_bias = conv3_bias.detach().cpu().numpy()
            fc_weights = fc_weights.detach().cpu().numpy()
            fc_bias = fc_bias.detach().cpu().numpy()

            #这部分代码其实用处不大，只是最开始测的时候提取卷积核权重用的
            # print("Conv1 Weights:", conv1_weights)
            # print("Conv2 Weights:", conv2_weights)
            # print("Conv3 Weights:", conv3_weights)
            # print("fc    Weights:", fc_weights)

            # conv1_weights_formatted = [' '.join([f"{w:.3f}" for w in conv1_weights[i:i + 4]]) for i in
            #                            range(0, len(conv1_weights), 4)]
            # conv2_weights_formatted = [' '.join([f"{w:.3f}" for w in conv2_weights[i:i + 4]]) for i in
            #                            range(0, len(conv2_weights), 4)]
            # conv3_weights_formatted = [' '.join([f"{w:.3f}" for w in conv3_weights[i:i + 4]]) for i in
            #                            range(0, len(conv3_weights), 4)]
            # fc_weights_formatted    = [' '.join([f"{w:.3f}" for w in fc_weights[i:i + 4]]) for i in
            #                            range(0, len(fc_weights), 4)]

            # 创建 DataFrame 并保存到 CSV 文件
            # conv1_df = pd.DataFrame({'Conv1 Weights': conv1_weights})
            # conv1_df.to_csv('weight/conv1_w.csv', index=False)

            # conv2_df = pd.DataFrame({'Conv2 Weights': conv2_weights})
            # conv2_df.to_csv('weight/conv2_w.csv', index=False)

            # conv3_df = pd.DataFrame({'Conv3 Weights': conv3_weights})
            # conv3_df.to_csv('weight/conv3_w.csv', index=False)

            # fc_df    = pd.DataFrame({'Fc Weights': fc_weights})
            # fc_df.to_csv('weight/fc_w.csv', index=False)
            conv1_df_flattened = pd.DataFrame(conv1_weights.flatten())
            conv1_b_df_flattened = pd.DataFrame(conv1_bias.flatten())
            conv2_df_flattened = pd.DataFrame(conv2_weights.flatten())
            conv2_b_df_flattened = pd.DataFrame(conv2_bias.flatten())
            conv3_df_flattened = pd.DataFrame(conv3_weights.flatten())
            conv3_b_df_flattened = pd.DataFrame(conv3_bias.flatten())
            fc_df_flattened    = pd.DataFrame(fc_weights.flatten())
            fc_bias_flattened = pd.DataFrame(fc_bias.flatten())

            conv1_df_flattened.to_csv('weight/conv1_w.csv', index=False)
            conv1_b_df_flattened.to_csv('weight/conv1_b.csv', index=False)

            conv2_df_flattened.to_csv('weight/conv2_w.csv', index=False)
            conv2_b_df_flattened.to_csv('weight/conv2_b.csv', index=False)

            conv3_df_flattened.to_csv('weight/conv3_w.csv', index=False)
            conv3_b_df_flattened.to_csv('weight/conv3_b.csv', index=False)

            fc_df_flattened.to_csv('weight/fc_w.csv', index=False)
            fc_bias_flattened.to_csv('weight/fc_b.csv', index=False)

    # 保存模型为onnx
    dummy_input = torch.randn(64, 1, 300, device='cuda')
    onnx_model_path = "model/cnn.onnx"
    torch.onnx.export(model,dummy_input,onnx_model_path)

    # 绘制
    plt.figure(figsize=(7, 4))  # 设置图形大小
    # 设置 train_losses 的曲线为绿色、实线，使用圆圈标记，标记大小为8
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='green', linestyle='-', marker='o', markersize=6)
    # 设置 test_losses 的曲线为浅红色、虚线，使用三角形标记，标记大小为8
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='lightcoral', linestyle='--', marker='^',
             markersize=6)
    plt.xlabel('Epoch', fontsize=12)  # 设置 x 轴标签和字体大小
    plt.ylabel('Loss', fontsize=12)  # 设置 y 轴标签和字体大小
    plt.title('Training and Test Loss Over Epochs', fontsize=16)  # 设置图表标题和字体大小
    plt.legend(fontsize=12)  # 设置图例字体大小
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，设置线型和透明度
    # 设置背景色、边框颜色
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')  # 浅灰色背景
    ax.spines['bottom'].set_color('blue')  # 底部边框颜色
    ax.spines['top'].set_color('blue')  # 顶部边框颜色
    ax.spines['right'].set_color('blue')  # 右边框颜色
    ax.spines['left'].set_color((0.35, 0.56, 0.86))  # 左边框颜色
    plt.show()

    # 绘制2
    plt.figure(figsize=(7, 4))  # 设置图形大小
    # 设置 train_losses 的曲线为绿色、实线，使用圆圈标记，标记大小为8
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Loss', color='green', linestyle='-', marker='o',
             markersize=8)
    # 设置 test_losses 的曲线为浅红色、虚线，使用三角形标记，标记大小为8
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Loss', color='lightcoral', linestyle='--', marker='^',
             markersize=8)
    plt.xlabel('Epoch', fontsize=14)  # 设置 x 轴标签和字体大小
    plt.ylabel('Accuracy', fontsize=14)  # 设置 y 轴标签和字体大小
    plt.title('Training and Test accuracy Over Epochs', fontsize=16)  # 设置图表标题和字体大小
    plt.legend(fontsize=12)  # 设置图例字体大小
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，设置线型和透明度
    # 设置背景色、边框颜色
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')  # 浅灰色背景
    ax.spines['bottom'].set_color('gray')  # 底部边框颜色
    ax.spines['top'].set_color('gray')  # 顶部边框颜色
    ax.spines['right'].set_color('gray')  # 右边框颜色
    ax.spines['left'].set_color('gray')  # 左边框颜色
    plt.show()
