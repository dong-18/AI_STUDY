# 如果你还想要“单张图片预测代码”，下面是加了详细注释的版本：

# 导入 PIL，用来打开图片
from PIL import Image

# 导入 PyTorch
import torch

# 导入神经网络模块
import torch.nn as nn

# 导入 torchvision 的 transforms，用来做图像预处理
import torchvision.transforms as transforms


# 判断设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 这里要写成你训练时的类别顺序
# 必须和 dataset.classes 的顺序一致
classes = ["class1", "class2"]


# 定义和训练时一致的图像预处理
transform = transforms.Compose([
    # 把图片缩放到和训练时一样的大小
    transforms.Resize((128, 128)),

    # 转成张量
    transforms.ToTensor(),
])


# 定义和训练时完全一样的模型结构
class SimpleCNN(nn.Module):
    # 初始化函数
    def __init__(self, num_classes):
        # 调用父类初始化
        super().__init__()

        # 特征提取层
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第二层卷积
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第三层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    # 前向传播
    def forward(self, x):
        # 卷积提特征
        x = self.features(x)

        # 全连接做分类
        x = self.classifier(x)

        # 返回结果
        return x


# 创建模型对象
model = SimpleCNN(num_classes=len(classes)).to(device)

# 加载训练好的模型参数
model.load_state_dict(torch.load("my_image_model.pth", map_location=device))

# 切换到评估模式
model.eval()


# 这里写你要预测的图片路径
img_path = "./1.jpg"

# 打开图片，并强制转成 RGB 三通道
img = Image.open(img_path).convert("RGB")

# 做和训练一致的预处理
img = transform(img)

# 增加一个 batch 维度
# 原来形状是 [3, 128, 128]
# 变成 [1, 3, 128, 128]
img = img.unsqueeze(0).to(device)


# 关闭梯度，进入推理模式
with torch.no_grad():
    # 前向传播得到输出
    output = model(img)

    # 取分数最高的类别索引
    pred = output.argmax(dim=1).item()

# 打印预测结果
print("预测结果:", classes[pred])