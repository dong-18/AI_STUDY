import math  # 导入数学库，用于 sqrt
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用函数接口


class CausalSelfAttention(nn.Module):  # 定义因果自注意力层
    def __init__(self, d_model, n_heads, dropout=0.1):  # 构造函数
        super().__init__()  # 调用父类构造函数
        assert d_model % n_heads == 0  # 保证 hidden dim 能被头数整除
        self.d_model = d_model  # 保存模型维度
        self.n_heads = n_heads  # 保存注意力头数
        self.head_dim = d_model // n_heads  # 计算每个头的维度

        self.q_proj = nn.Linear(d_model, d_model)  # Q 投影层
        self.k_proj = nn.Linear(d_model, d_model)  # K 投影层
        self.v_proj = nn.Linear(d_model, d_model)  # V 投影层
        self.o_proj = nn.Linear(d_model, d_model)  # 输出投影层

        self.dropout = nn.Dropout(dropout)  # dropout 层

    def forward(self, x, attn_mask=None):  # 前向传播，x 形状为 [B, T, C]
        B, T, C = x.shape  # 取出 batch、大写序列长度、通道维度

        q = self.q_proj(x)  # 对输入做 Q 投影
        q = q.view(B, T, self.n_heads, self.head_dim)  # 变形成多头结构
        q = q.transpose(1, 2)  # 调整为 [B, n_heads, T, head_dim]

        k = self.k_proj(x)  # 对输入做 K 投影
        k = k.view(B, T, self.n_heads, self.head_dim)  # 变形成多头结构
        k = k.transpose(1, 2)  # 调整维度顺序

        v = self.v_proj(x)  # 对输入做 V 投影
        v = v.view(B, T, self.n_heads, self.head_dim)  # 变形成多头结构
        v = v.transpose(1, 2)  # 调整维度顺序

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 计算注意力分数并缩放

        causal_mask = torch.triu(  # 构造上三角因果 mask
            torch.ones(T, T, device=x.device, dtype=torch.bool),  # 先建立全 1 布尔矩阵
            diagonal=1,  # 主对角线上方置为 True
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))  # 把未来位置填成负无穷

        if attn_mask is not None:  # 如果传入了 padding mask
            pad_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)  # 扩展成 [B,1,1,T]
            scores = scores.masked_fill(pad_mask, float("-inf"))  # 把 pad 位置屏蔽掉

        attn = torch.softmax(scores, dim=-1)  # 对最后一维做 softmax 得到注意力权重
        attn = self.dropout(attn)  # 对注意力权重做 dropout

        out = attn @ v  # 用注意力权重加权求和 V
        out = out.transpose(1, 2).contiguous()  # 转回 [B, T, n_heads, head_dim] 排列
        out = out.view(B, T, C)  # 合并多头，恢复到 [B, T, C]
        out = self.o_proj(out)  # 做输出线性投影
        return out  # 返回输出


class MLP(nn.Module):  # 定义前馈网络
    def __init__(self, d_model, hidden_dim, dropout=0.1):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.fc1 = nn.Linear(d_model, hidden_dim)  # 第一层线性变换
        self.fc2 = nn.Linear(hidden_dim, d_model)  # 第二层线性变换
        self.dropout = nn.Dropout(dropout)  # dropout 层

    def forward(self, x):  # 前向传播
        x = self.fc1(x)  # 输入先经过第一层线性
        x = F.gelu(x)  # 使用 GELU 激活函数
        x = self.fc2(x)  # 再经过第二层线性映射回原维度
        x = self.dropout(x)  # 对输出做 dropout
        return x  # 返回结果


class Block(nn.Module):  # 定义一个 Transformer Block
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.ln1 = nn.LayerNorm(d_model)  # 注意力前的 LayerNorm
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)  # 因果自注意力层
        self.ln2 = nn.LayerNorm(d_model)  # MLP 前的 LayerNorm
        self.mlp = MLP(d_model, d_model * mlp_ratio, dropout)  # 前馈网络

    def forward(self, x, attn_mask=None):  # 前向传播
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)  # 先做 LN+注意力，再加残差
        x = x + self.mlp(self.ln2(x))  # 再做 LN+MLP，再加残差
        return x  # 返回 block 输出


class MiniGPT(nn.Module):  # 定义一个最小版 GPT 模型
    def __init__(self, vocab_size, max_seq_len=256, d_model=256, n_heads=4, n_layers=4, dropout=0.1):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.max_seq_len = max_seq_len  # 保存最大序列长度
        self.tok_emb = nn.Embedding(vocab_size, d_model)  # token embedding 层
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # 位置 embedding 层
        self.drop = nn.Dropout(dropout)  # 输入 dropout

        self.blocks = nn.ModuleList([  # 构造多个 Transformer Block
            Block(d_model, n_heads, mlp_ratio=4, dropout=dropout)  # 创建单个 block
            for _ in range(n_layers)  # 重复 n_layers 次
        ])

        self.ln_f = nn.LayerNorm(d_model)  # 最终 LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 语言模型输出层

        self.lm_head.weight = self.tok_emb.weight  # 使用权重共享，让输出层权重和输入 embedding 一样

    def forward(self, input_ids, attention_mask=None, labels=None):  # 前向传播
        B, T = input_ids.shape  # 获取 batch 大小和序列长度
        assert T <= self.max_seq_len  # 保证输入长度不超过模型支持的最大长度

        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)  # 构造位置 id，形状 [1, T]

        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # token embedding 与 position embedding 相加
        x = self.drop(x)  # 对输入表示做 dropout

        for block in self.blocks:  # 依次经过每个 Transformer Block
            x = block(x, attn_mask=attention_mask)  # 把 attention mask 传给 block

        x = self.ln_f(x)  # 最后做 LayerNorm
        logits = self.lm_head(x)  # 映射到词表维度，得到 logits

        loss = None  # 先把 loss 初始化为空
        if labels is not None:  # 如果给了 labels，说明当前是训练模式
            loss = F.cross_entropy(  # 计算交叉熵损失
                logits.reshape(-1, logits.size(-1)),  # 把 logits 拉平成 [B*T, V]
                labels.reshape(-1),  # 把 labels 拉平成 [B*T]
                ignore_index=-100,  # 忽略 labels 中等于 -100 的位置
            )

        return {"logits": logits, "loss": loss}  # 返回 logits 和 loss

    @torch.no_grad()  # 推理时不计算梯度
    def generate(self, input_ids, max_new_tokens, eos_token_id=None, temperature=1.0):  # 自回归生成函数
        self.eval()  # 切换到评估模式
        for _ in range(max_new_tokens):  # 循环生成若干新 token
            idx_cond = input_ids[:, -self.max_seq_len:]  # 如果太长，只保留最后 max_seq_len 个 token
            out = self(idx_cond)  # 前向计算当前上下文
            logits = out["logits"][:, -1, :] / temperature  # 取最后一个位置的 logits 并除以温度
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # 用贪心策略选最大概率 token
            input_ids = torch.cat([input_ids, next_token], dim=1)  # 把新 token 拼接到输入序列后面

            if eos_token_id is not None:  # 如果指定了 eos token
                if (next_token == eos_token_id).all():  # 如果 batch 中全部都生成了 eos
                    break  # 提前停止生成

        return input_ids  # 返回完整生成结果