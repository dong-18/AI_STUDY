# 这是一个“简单的图片分类训练代码”，使用 PyTorch 和 torchvision。
# 它会读取本地文件夹中的图片，并训练一个基础 CNN 来做分类。
# 你只需要把 data_dir 改成你自己的图片数据集路径即可。

# 导入 PyTorch 主库，用于张量计算和深度学习训练
import torch

# 导入神经网络模块，里面有各种网络层和损失函数
import torch.nn as nn

# 导入优化器模块，这里后面会用 Adam 优化器
import torch.optim as optim

# 从 torchvision 导入常用的数据集工具和图像预处理工具
from torchvision import datasets, transforms

# 导入 DataLoader，用于按 batch 读取数据；random_split 用于划分训练集和测试集
from torch.utils.data import DataLoader, random_split


# 判断当前机器是否有可用的 GPU
# 如果有 CUDA，就用 GPU 训练；否则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 打印当前使用的设备，方便确认是否启用了 GPU
print("device:", device)


# 这里填写你的图片文件夹路径
# 文件夹结构必须是：
# dataset/
# ├── class1/
# │   ├── 1.jpg
# │   ├── 2.jpg
# ├── class2/
# │   ├── 1.jpg
# │   ├── 2.jpg
# ImageFolder 会自动把每个子文件夹名当作一个类别
data_dir = "./dataset"


# 定义图像预处理流程
transform = transforms.Compose([
    # 把所有图片缩放到 128x128，方便统一输入网络
    transforms.Resize((128, 128)),

    # 把图片从 PIL 格式转成张量，并把像素值缩放到 [0,1]
    transforms.ToTensor(),
])


# 使用 ImageFolder 读取图片数据集
# 它会自动扫描 data_dir 下的每个子文件夹，并给类别分配标签编号
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 打印类别名称列表，比如 ['cat', 'dog']
print("classes:", dataset.classes)

# 打印“类别名 -> 数字标签”的映射关系，比如 {'cat': 0, 'dog': 1}
print("class_to_idx:", dataset.class_to_idx)


# 按照 80% 和 20% 的比例划分训练集和测试集
train_size = int(0.8 * len(dataset))  # 训练集大小
test_size = len(dataset) - train_size  # 测试集大小

# 随机划分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# 用 DataLoader 按批次加载训练集
train_loader = DataLoader(
    train_dataset,   # 训练集数据
    batch_size=4,   # 每次读取 32 张图片
    shuffle=True     # 每个 epoch 都打乱顺序，有助于训练
)

# 用 DataLoader 按批次加载测试集
test_loader = DataLoader(
    test_dataset,    # 测试集数据
    batch_size=4,   # 每次读取 32 张图片
    shuffle=False    # 测试时通常不打乱
)


# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    # 初始化函数，num_classes 表示分类类别数量
    def __init__(self, num_classes):
        # 调用父类初始化
        super().__init__()

        # 定义特征提取部分
        self.features = nn.Sequential(
            # 第一层卷积：
            # 输入通道数 3（RGB 彩图）
            # 输出通道数 16
            # 卷积核大小 3x3
            # padding=1 表示输出尺寸尽量保持不变
            nn.Conv2d(3, 16, kernel_size=3, padding=1),

            # 激活函数 ReLU，引入非线性
            nn.ReLU(),

            # 最大池化，2x2，下采样一半
            nn.MaxPool2d(2),

            # 第二层卷积：16 通道 -> 32 通道
            nn.Conv2d(16, 32, kernel_size=3, padding=1),

            # ReLU 激活
            nn.ReLU(),

            # 再次池化
            nn.MaxPool2d(2),

            # 第三层卷积：32 通道 -> 64 通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            # ReLU 激活
            nn.ReLU(),

            # 再次池化
            nn.MaxPool2d(2)
        )

        # 定义分类器部分
        self.classifier = nn.Sequential(
            # 把多维特征图拉平成一维向量
            nn.Flatten(),

            # 全连接层：
            # 输入维度是 64 * 16 * 16
            # 因为 128x128 图片经过三次 2 倍池化后变成 16x16
            nn.Linear(64 * 16 * 16, 128),

            # ReLU 激活
            nn.ReLU(),

            # 输出层，输出类别数个分数
            nn.Linear(128, num_classes)
        )

    # 定义前向传播
    def forward(self, x):
        # 先经过卷积提取特征
        x = self.features(x)

        # 再经过全连接层分类
        x = self.classifier(x)

        # 返回输出结果
        return x


# 获取类别数量
num_classes = len(dataset.classes)

# 创建模型实例，并移动到 device（GPU 或 CPU）
model = SimpleCNN(num_classes).to(device)


# 定义损失函数：交叉熵损失，适合多分类任务
criterion = nn.CrossEntropyLoss()

# 定义优化器：Adam，学习率设为 0.001
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# 定义“训练一个 epoch”的函数
def train_one_epoch(model, loader, optimizer, criterion, device):
    # 切换到训练模式
    model.train()

    # 用于累计总损失
    total_loss = 0.0

    # 用于统计预测正确的数量
    correct = 0

    # 用于统计样本总数
    total = 0

    # 遍历一个 epoch 中的所有 batch
    for images, labels in loader:
        # 把图片数据移动到 device
        images = images.to(device)

        # 把标签数据移动到 device
        labels = labels.to(device)

        # 清空上一轮的梯度
        optimizer.zero_grad()

        # 前向传播，得到模型输出
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播，计算梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累加当前 batch 的损失值
        total_loss += loss.item()

        # 取每行最大值对应的索引，作为预测类别
        preds = outputs.argmax(dim=1)

        # 累加总样本数
        total += labels.size(0)

        # 统计当前 batch 预测正确的样本数
        correct += (preds == labels).sum().item()

    # 返回平均损失 和 准确率
    return total_loss / len(loader), correct / total


# 定义评估函数
def evaluate(model, loader, criterion, device):
    # 切换到评估模式
    model.eval()

    # 累计总损失
    total_loss = 0.0

    # 累计正确数
    correct = 0

    # 累计总数
    total = 0

    # 在评估阶段关闭梯度计算，节省显存和加快速度
    with torch.no_grad():
        # 遍历测试集
        for images, labels in loader:
            # 把图片移动到 device
            images = images.to(device)

            # 把标签移动到 device
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 累加损失
            total_loss += loss.item()

            # 取预测类别
            preds = outputs.argmax(dim=1)

            # 累加样本数
            total += labels.size(0)

            # 累加预测正确数
            correct += (preds == labels).sum().item()

    # 返回平均损失和准确率
    return total_loss / len(loader), correct / total


# 设置训练轮数
epochs = 5

# 开始训练
for epoch in range(epochs):
    # 训练一个 epoch，并获得训练损失和训练准确率
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

    # 在测试集上评估，并获得测试损失和测试准确率
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # 打印当前 epoch 的训练结果
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
    )


# 训练完成后，把模型参数保存到本地文件
torch.save(model.state_dict(), "my_image_model.pth")

# 打印保存成功提示
print("模型已保存: my_image_model.pth")