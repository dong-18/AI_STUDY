import torch  # 导入 PyTorch 主库，用于张量计算和深度学习相关操作
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并简写为 nn
import math  # 导入 Python 标准数学库，用于对数等数学运算

class PositionalEncoding(nn.Module):  # 定义位置编码类，继承自 nn.Module
    def __init__(self, d_model, max_len=5000):  # 初始化函数，d_model 是特征维度，max_len 是支持的最大序列长度
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        pe = torch.zeros(max_len, d_model)  # 创建一个形状为 (max_len, d_model) 的全零张量，用于存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 生成位置索引 [0, 1, 2, ..., max_len-1]，并扩展为列向量形状 (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算位置编码中的缩放项，只针对偶数维度生成频率系数
        为什么需要位置信息（没有位置信息会怎么样，是因为最后结果无法区分，然后mlp只对自己计算吗），绝对和相对位置
        pe[:, 0::2] = torch.sin(position * div_term)  # 对位置编码的偶数列使用正弦函数填充
        pe[:, 1::2] = torch.cos(position * div_term)  # 对位置编码的奇数列使用余弦函数填充

        pe = pe.unsqueeze(1)  # 在第 1 维增加一个维度，把形状从 (max_len, d_model) 变成 (max_len, 1, d_model)，方便与 batch 输入相加
        self.register_buffer('pe', pe)  # 将 pe 注册为模型缓冲区，它会随模型保存和加载，但不会被当作可训练参数更新

    def forward(self, x):  # 定义前向传播函数，输入 x
        # x: (seq_len, batch_size, d_model)  # 说明输入张量 x 的形状：序列长度、批大小、特征维度
        return x + self.pe[:x.size(0)]  # 将对应序列长度的位置编码加到输入嵌入上，并返回结果


class SimpleTransformer(nn.Module):  # 定义一个简单的 Transformer 模型类
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2):  # 初始化函数，设置词表大小、特征维度、注意力头数和层数
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.embedding = nn.Embedding(vocab_size, d_model)  # 定义词嵌入层，把词 ID 映射为 d_model 维向量
        self.pos_encoder = PositionalEncoding(d_model)  # 创建位置编码模块，用于给词向量加入位置信息

        self.transformer = nn.Transformer(  # 定义 PyTorch 内置的 Transformer 模型
            d_model=d_model,  # 指定输入和输出的特征维度
            nhead=nhead,  # 指定多头注意力机制中的头数
            num_encoder_layers=num_layers,  # 指定编码器层数
            num_decoder_layers=num_layers  # 指定解码器层数
        )

        self.fc_out = nn.Linear(d_model, vocab_size)  # 定义全连接输出层，把 Transformer 输出映射到词表大小，用于预测每个词的概率分布

    def forward(self, src, tgt):  # 定义前向传播函数，接收源序列 src 和目标序列 tgt
        # src, tgt: (seq_len, batch_size)  # 说明 src 和 tgt 的形状：序列长度、批大小
        src_emb = self.pos_encoder(self.embedding(src))  # 先对源序列做词嵌入，再加上位置编码
        tgt_emb = self.pos_encoder(self.embedding(tgt))  # 先对目标序列做词嵌入，再加上位置编码

        output = self.transformer(src_emb, tgt_emb)  # 将带有位置编码的源序列和目标序列输入 Transformer，得到解码输出
        output = self.fc_out(output)  # 将 Transformer 输出通过线性层映射到词表维度，输出形状为 (tgt_seq_len, batch_size, vocab_size)
        return output  # 返回最终输出结果


# 假设词表大小为 100  # 说明下面示例中使用的词表大小
vocab_size = 100  # 设置词表大小为 100
model = SimpleTransformer(vocab_size=vocab_size)  # 实例化一个 SimpleTransformer 模型

# 构造假的输入  # 说明下面是为了测试模型而随机生成的输入数据
src = torch.randint(0, vocab_size, (5, 2))  # 随机生成源序列，词 ID 范围是 0 到 vocab_size-1，形状为 (5, 2)，表示源序列长度为 5，batch 大小为 2
tgt = torch.randint(0, vocab_size, (6, 2))  # 随机生成目标序列，词 ID 范围是 0 到 vocab_size-1，形状为 (6, 2)，表示目标序列长度为 6，batch 大小为 2

logits = model(src, tgt)  # 将源序列和目标序列输入模型，得到输出 logits
print("logits shape:", logits.shape)  # 打印输出张量的形状，预期为 (6, 2, 100)