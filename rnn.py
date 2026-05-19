import random  # 导入 Python 内置的随机模块，用于生成随机的 0/1 序列
import torch  # 导入 PyTorch 主库，用于张量计算和深度学习
import torch.nn as nn  # 导入 PyTorch 神经网络模块，并简写为 nn
import torch.optim as optim  # 导入 PyTorch 优化器模块，并简写为 optim


def generate_data(num_samples=1000, seq_len=8):  # 定义生成数据的函数，默认生成 1000 条、长度为 8 的序列
    """生成二分类序列数据：  # 函数说明：生成一个二分类任务的数据集
    若序列首元素与尾元素相同，标签为 1，否则为 0。  # 标签规则：首尾相同记为 1，不同记为 0
    """  # 文档字符串结束
    X, y = [], []  # 初始化两个空列表，X 用来存放输入序列，y 用来存放对应标签

    for _ in range(num_samples):  # 循环 num_samples 次，逐条生成样本
        seq = [random.randint(0, 1) for _ in range(seq_len)]  # 生成一个长度为 seq_len 的随机二进制序列
        label = 1 if seq[0] == seq[-1] else 0  # 如果序列首元素等于尾元素，则标签为 1，否则为 0
        X.append(seq)  # 将当前生成的序列加入输入列表 X
        y.append(label)  # 将当前序列对应的标签加入标签列表 y

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # 将 X 转成 float32 张量，并在最后增加一维，形状变为 [N, seq_len, 1]
    y = torch.tensor(y, dtype=torch.long)  # 将 y 转成 long 类型张量，形状为 [N]
    return X, y  # 返回输入张量 X 和标签张量 y


class RNNClassifier(nn.Module):  # 定义一个基于 RNN 的分类模型类，继承自 nn.Module
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, num_classes=2):  # 初始化模型参数
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.rnn = nn.RNN(  # 定义一个普通 RNN 层
            input_size=input_size,  # 每个时间步输入特征的维度，这里是 1
            hidden_size=hidden_size,  # RNN 隐藏层状态的维度，这里是 16
            num_layers=num_layers,  # RNN 的层数，这里是 1 层
            batch_first=True,  # 设置输入输出张量格式为 [batch_size, seq_len, feature]
        )  # RNN 层定义结束
        self.fc = nn.Linear(hidden_size, num_classes)  # 定义一个全连接层，将隐藏状态映射到类别数

    def forward(self, x):  # 定义前向传播函数
        _, h_n = self.rnn(x)  # 将输入 x 送入 RNN，忽略所有时间步输出，只保留最终隐藏状态 h_n
        last_hidden = h_n[-1]               # 取最后一层的最终隐藏状态，形状为 [batch_size, hidden_size]
        logits = self.fc(last_hidden)       # 将最后隐藏状态输入全连接层，得到分类输出 logits，形状为 [batch_size, num_classes]
        return logits  # 返回未经过 softmax 的分类分数


class LSTMClassifier(nn.Module):  # 定义一个基于 LSTM 的分类模型类，继承自 nn.Module
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, num_classes=2):  # 初始化模型参数
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.lstm = nn.LSTM(  # 定义一个 LSTM 层
            input_size=input_size,  # 每个时间步输入特征的维度，这里是 1
            hidden_size=hidden_size,  # LSTM 隐藏层状态的维度，这里是 16
            num_layers=num_layers,  # LSTM 的层数，这里是 1 层
            batch_first=True,  # 设置输入输出张量格式为 [batch_size, seq_len, feature]
        )  # LSTM 层定义结束
        self.fc = nn.Linear(hidden_size, num_classes)  # 定义一个全连接层，将隐藏状态映射到类别数

    def forward(self, x):  # 定义前向传播函数
        _, (h_n, _) = self.lstm(x)  # 将输入 x 送入 LSTM，忽略所有时间步输出和细胞状态，只保留最终隐藏状态 h_n
        last_hidden = h_n[-1]               # 取最后一层的最终隐藏状态，形状为 [batch_size, hidden_size]
        logits = self.fc(last_hidden)       # 将最后隐藏状态输入全连接层，得到分类输出 logits，形状为 [batch_size, num_classes]
        return logits  # 返回未经过 softmax 的分类分数


def evaluate(model, X, y):  # 定义模型评估函数，用于计算给定数据上的准确率
    model.eval()  # 将模型切换到评估模式
    with torch.no_grad():  # 在评估阶段关闭梯度计算，减少内存占用并提升速度
        logits = model(X)  # 将输入数据 X 送入模型，得到分类输出 logits
        pred = torch.argmax(logits, dim=1)  # 在类别维度上取最大值索引，作为预测类别
        acc = (pred == y).float().mean().item()  # 计算预测正确的比例，并转成 Python 浮点数
    return acc  # 返回准确率


def train_model(model, X_train, y_train, X_test, y_test, epochs=300, lr=0.001):  # 定义模型训练函数
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数，适用于多分类任务
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 定义 Adam 优化器，并设置学习率

    for epoch in range(epochs):  # 训练指定的轮数 epochs
        model.train()  # 将模型切换到训练模式
        optimizer.zero_grad()  # 清空上一轮残留的梯度

        logits = model(X_train)  # 将整个训练集输入模型，得到输出 logits
        loss = criterion(logits, y_train)  # 计算模型输出与真实标签之间的交叉熵损失

        loss.backward()  # 对损失进行反向传播，计算梯度
        optimizer.step()  # 根据计算得到的梯度更新模型参数

        train_acc = evaluate(model, X_train, y_train)  # 计算当前模型在训练集上的准确率
        test_acc = evaluate(model, X_test, y_test)  # 计算当前模型在测试集上的准确率

        print(  # 打印当前训练轮次的信息
            f"Epoch {epoch + 1:2d} | "  # 打印当前的 epoch 编号
            f"Loss: {loss.item():.4f} | "  # 打印当前损失值，保留 4 位小数
            f"Train Acc: {train_acc:.4f} | "  # 打印训练集准确率，保留 4 位小数
            f"Test Acc: {test_acc:.4f}"  # 打印测试集准确率，保留 4 位小数
        )  # print 结束


def predict(model, seq):  # 定义单条序列的预测函数
    model.eval()  # 将模型切换到评估模式
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # 将输入序列转成张量，并调整形状为 [1, seq_len, 1]

    with torch.no_grad():  # 关闭梯度计算，表示这里只做推理
        logits = model(x)  # 将处理后的输入张量送入模型，得到输出 logits
        pred = torch.argmax(logits, dim=1).item()  # 取 logits 中最大值对应的类别索引，并转为 Python 整数

    return pred  # 返回预测结果


def main():  # 定义主函数，组织数据生成、模型训练和测试预测流程
    # 生成数据  # 注释：下面开始生成训练集和测试集
    X_train, y_train = generate_data(num_samples=2000, seq_len=8)  # 生成 2000 条训练数据，每条序列长度为 8
    X_test, y_test = generate_data(num_samples=400, seq_len=8)  # 生成 400 条测试数据，每条序列长度为 8

    print("Train X shape:", X_train.shape)  # 打印训练输入数据的形状
    print("Train y shape:", y_train.shape)  # 打印训练标签数据的形状

    # 训练 RNN  # 注释：下面开始训练 RNN 模型
    print("\nTraining RNN...")  # 打印提示，表示开始训练 RNN
    rnn_model = RNNClassifier()  # 实例化一个 RNN 分类模型
    train_model(rnn_model, X_train, y_train, X_test, y_test)  # 使用训练集和测试集训练 RNN 模型

    # 训练 LSTM  # 注释：下面开始训练 LSTM 模型
    print("\nTraining LSTM...")  # 打印提示，表示开始训练 LSTM
    lstm_model = LSTMClassifier()  # 实例化一个 LSTM 分类模型
    train_model(lstm_model, X_train, y_train, X_test, y_test)  # 使用训练集和测试集训练 LSTM 模型

    # 测试预测  # 注释：下面定义测试序列并进行预测
    test_seq1 = [1, 0, 0, 1,1, 0, 0, 1]  # 定义测试序列 1，首尾相同，理论标签应为 1
    test_seq2 = [1, 0, 1, 0,1, 0, 1, 0]  # 定义测试序列 2，首尾不同，理论标签应为 0

    print("\nRNN Predictions:")  # 打印提示，表示下面展示 RNN 的预测结果
    print(test_seq1, "->", predict(rnn_model, test_seq1))  # 使用训练好的 RNN 模型预测 test_seq1
    print(test_seq2, "->", predict(rnn_model, test_seq2))  # 使用训练好的 RNN 模型预测 test_seq2

    print("\nLSTM Predictions:")  # 打印提示，表示下面展示 LSTM 的预测结果
    print(test_seq1, "->", predict(lstm_model, test_seq1))  # 使用训练好的 LSTM 模型预测 test_seq1
    print(test_seq2, "->", predict(lstm_model, test_seq2))  # 使用训练好的 LSTM 模型预测 test_seq2
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":# 判断当前脚本是否作为主程序直接运行
    set_seed(42)
    main()  # 如果是主程序运行，则执行 main() 函数