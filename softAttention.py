import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入常用函数模块，例如 softmax
import random  # 导入随机数模块，用于生成模拟数据


class SoftAttention(nn.Module):  # 定义软性注意力模块
    def __init__(self, hidden_dim):  # 初始化函数，hidden_dim 是隐藏层维度
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.w = nn.Linear(hidden_dim, 1)  # 定义一个线性层，把每个时间步的 hidden vector 映射成一个分数

    def forward(self, x, mask=None):  # 定义前向传播函数，x 是序列特征，mask 用于屏蔽 padding
        # x 的形状是 [batch_size, seq_len, hidden_dim]
        # mask 的形状是 [batch_size, seq_len]

        scores = self.w(x).squeeze(-1)  # 先通过线性层得到每个时间步的分数，形状从 [B, L, H] 变成 [B, L]

        if mask is not None:  # 如果传入了 mask
            scores = scores.masked_fill(mask == 0, float('-inf'))  # 把无效位置设成负无穷，这样 softmax 后这些位置权重接近 0

        attn_weights = F.softmax(scores, dim=-1)  # 在序列长度维度上做 softmax，得到注意力权重 [B, L]

        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # 用注意力权重对序列特征做加权求和，得到上下文向量 [B, H]

        return context, attn_weights  # 返回上下文向量和注意力权重


class AttentionClassifier(nn.Module):  # 定义带注意力的分类模型
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):  # 初始化函数
        super().__init__()  # 调用父类初始化方法
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # 词嵌入层，把 token id 映射成向量
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)  # 定义 GRU 层，输入维度 embed_dim，输出维度 hidden_dim
        self.attention = SoftAttention(hidden_dim)  # 定义软性注意力层
        self.fc = nn.Linear(hidden_dim, num_classes)  # 定义全连接分类层，把上下文向量映射到类别数

    def forward(self, input_ids, mask=None):  # 定义前向传播函数
        # input_ids 的形状是 [batch_size, seq_len]
        # mask 的形状是 [batch_size, seq_len]

        x = self.embedding(input_ids)  # 把 token id 转成词向量，输出形状 [B, L, E]

        output, _ = self.gru(x)  # 将词向量序列送入 GRU，output 形状 [B, L, H]

        context, attn_weights = self.attention(output, mask)  # 对 GRU 每个时间步输出做注意力加权，得到上下文向量和权重

        logits = self.fc(context)  # 把上下文向量送入全连接层，得到分类 logits，形状 [B, C]

        return logits, attn_weights  # 返回分类输出和注意力权重


def generate_data(num_samples=200, seq_len=6, vocab_size=20, target_token=9):  # 定义一个函数，用来生成模拟数据
    # 规则是：如果序列中出现 target_token，则标签为 1，否则标签为 0

    data = []  # 用于保存所有样本序列
    labels = []  # 用于保存所有样本标签

    for _ in range(num_samples):  # 循环生成 num_samples 条样本
        seq = [random.randint(1, vocab_size - 1) for _ in range(seq_len)]  # 随机生成一个长度为 seq_len 的序列，取值范围 [1, vocab_size-1]
        label = 1 if target_token in seq else 0  # 如果序列中包含 target_token，则标签设为 1，否则设为 0
        data.append(seq)  # 把生成的序列加入数据列表
        labels.append(label)  # 把对应标签加入标签列表

    return torch.tensor(data), torch.tensor(labels)  # 把列表转换成张量并返回


vocab_size = 20  # 词表大小，一共 20 个 token，0 保留给 padding
embed_dim = 16  # 词向量维度
hidden_dim = 32  # GRU 隐藏层维度
num_classes = 2  # 分类类别数，这里是二分类
seq_len = 6  # 每个序列的长度
target_token = 9  # 规定 token=9 是“关键 token”，出现它就标为 1

X, y = generate_data(num_samples=500, seq_len=seq_len, vocab_size=vocab_size, target_token=target_token)  # 生成 500 条训练数据

mask = (X != 0).long()  # 构造 mask，非 0 的位置为 1，表示有效 token；0 的位置为 0，表示 padding

model = AttentionClassifier(vocab_size, embed_dim, hidden_dim, num_classes)  # 实例化模型

criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数，用于分类任务

optimizer = optim.Adam(model.parameters(), lr=0.01)  # 定义 Adam 优化器，学习率设置为 0.01


for epoch in range(100):  # 训练 10 个 epoch
    model.train()  # 把模型切换到训练模式

    optimizer.zero_grad()  # 清空上一轮的梯度，防止梯度累积

    logits, attn_weights = model(X, mask)  # 前向传播，得到分类输出和注意力权重

    loss = criterion(logits, y)  # 计算预测结果和真实标签之间的损失

    loss.backward()  # 反向传播，计算梯度

    optimizer.step()  # 根据梯度更新模型参数

    preds = torch.argmax(logits, dim=-1)  # 取每个样本预测概率最大的类别作为最终预测结果

    acc = (preds == y).float().mean().item()  # 计算当前训练集上的分类准确率

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")  # 打印当前 epoch 的损失和准确率


model.eval()  # 把模型切换到评估模式

test_samples = torch.tensor([  # 构造几个手工测试样本
    [1, 2, 3, 4, 5, 6],  # 不含 9，理论上标签应为 0
    [1, 9, 3, 4, 5, 6],  # 含 9，理论上标签应为 1
    [9, 2, 3, 4, 5, 6],  # 含 9，理论上标签应为 1
    [1, 2, 3, 4, 8, 7]   # 不含 9，理论上标签应为 0
])  # 这里一共 4 个测试样本

test_mask = (test_samples != 0).long()  # 为测试数据构造 mask，非 0 位置为有效 token

with torch.no_grad():  # 在评估阶段关闭梯度计算，节省内存并加快速度
    logits, attn_weights = model(test_samples, test_mask)  # 前向传播，得到测试样本的分类输出和注意力权重
    preds = torch.argmax(logits, dim=-1)  # 取最大值对应的类别作为预测标签

print("\n测试样本：")  # 打印说明文字
print(test_samples)  # 打印测试输入序列

print("\n预测结果：")  # 打印说明文字
print(preds)  # 打印模型预测的类别

print("\n注意力权重：")  # 打印说明文字
print(attn_weights)  # 打印每个测试样本在各个位置上的注意力权重