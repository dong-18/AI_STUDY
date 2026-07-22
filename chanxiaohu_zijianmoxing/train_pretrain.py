import torch  # 导入 PyTorch
import torch.optim as optim  # 导入优化器模块
from torch.utils.data import Dataset, DataLoader  # 导入数据集和数据加载器
from tokenizer import CharTokenizer  # 导入 tokenizer
from model import MiniGPT  # 导入模型


class PretrainDataset(Dataset):  # 定义预训练数据集
    def __init__(self, text, tokenizer, max_seq_len=128):  # 构造函数
        self.tokenizer = tokenizer  # 保存 tokenizer
        self.max_seq_len = max_seq_len  # 保存最大序列长度
# "合起来做出网页，模型反向传播"
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)  # 把整段文本编码成 token id
        self.samples = []  # 初始化样本列表

        for i in range(0, len(ids) - 1, max_seq_len):  # 每次按 max_seq_len 做切片
            chunk = ids[i:i + max_seq_len + 1]  # 取一段长度为 max_seq_len+1 的片段
            if len(chunk) < 2:  # 如果片段长度不足 2
                continue  # 跳过
            input_ids = chunk[:-1]  # 输入是前 N 个 token
            labels = chunk[1:]  # 标签是后 N 个 token，也就是右移一位
            self.samples.append((input_ids, labels))  # 保存一条样本

    def __len__(self):  # 返回数据集大小
        return len(self.samples)  # 样本数

    def __getitem__(self, idx):  # 根据索引取样本
        input_ids, labels = self.samples[idx]  # 取出一条样本
        return {  # 返回字典格式，便于后面组 batch
            "input_ids": torch.tensor(input_ids, dtype=torch.long),  # 输入 token id
            "labels": torch.tensor(labels, dtype=torch.long),  # 目标 label
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),  # 预训练这里全部有效，所以全 1
        }


def collate_fn(batch, pad_token_id):  # 自定义 batch 拼接函数
    max_len = max(len(x["input_ids"]) for x in batch)  # 找出当前 batch 的最大长度

    input_ids = []  # 初始化 input_ids 列表
    labels = []  # 初始化 labels 列表
    attention_mask = []  # 初始化 attention_mask 列表

    for x in batch:  # 遍历 batch 中的每个样本
        l = len(x["input_ids"])  # 当前样本长度
        pad_len = max_len - l  # 需要补齐的长度

        input_ids.append(  # 处理 input_ids
            torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])  # 用 pad_token_id 补齐
        )

        labels.append(  # 处理 labels
            torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])  # label 补齐位置设成 -100
        )

        attention_mask.append(  # 处理 attention_mask
            torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])  # pad 位置 mask 设为 0
        )

    return {  # 返回拼好的 batch
        "input_ids": torch.stack(input_ids),  # 堆成 [B, T]
        "labels": torch.stack(labels),  # 堆成 [B, T]
        "attention_mask": torch.stack(attention_mask),  # 堆成 [B, T]
    }


def main():  # 主函数
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

    tokenizer = CharTokenizer.load("data/vocab.json")  # 加载已保存的词表

    with open("data/pretrain.txt", "r", encoding="utf-8") as f:  # 打开预训练文本
        text = "\n".join([line.strip() for line in f if line.strip()])  # 把非空行拼成一个大字符串

    dataset = PretrainDataset(text, tokenizer, max_seq_len=128)  # 构建数据集

    loader = DataLoader(  # 构建数据加载器
        dataset,  # 传入数据集
        batch_size=8,  # batch 大小
        shuffle=True,  # 打乱顺序
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),  # 自定义拼 batch
    )

    model = MiniGPT(  # 创建模型
        vocab_size=tokenizer.vocab_size(),  # 词表大小
        max_seq_len=128,  # 最大序列长度
        d_model=256,  # 隐层维度
        n_heads=4,  # 头数
        n_layers=4,  # 层数
        dropout=0.1,  # dropout
    ).to(device)  # 放到设备上

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)  # 创建 AdamW 优化器

    model.train()  # 切换到训练模式
    for epoch in range(200):  # 训练 10 个 epoch
        total_loss = 0.0  # 用于统计 epoch 总 loss

        for step, batch in enumerate(loader):  # 遍历每个 batch
            batch = {k: v.to(device) for k, v in batch.items()}  # 把 batch 移动到设备上

            out = model(**batch)  # 前向传播
            loss = out["loss"]  # 取出 loss

            optimizer.zero_grad()  # 清空旧梯度

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加 loss

            if step % 10 == 0:  # 每 10 步打印一次
                print(f"[Pretrain] epoch={epoch} step={step} loss={loss.item():.4f}")  # 打印当前训练状态

        avg_loss = total_loss / max(len(loader), 1)  # 计算平均 loss
        print(f"[Pretrain] epoch={epoch} avg_loss={avg_loss:.4f}")  # 打印 epoch 平均 loss

    torch.save(model.state_dict(), "pretrain_ckpt.pt")  # 保存预训练模型参数
    print("saved pretrain checkpoint to pretrain_ckpt.pt")  # 打印保存提示


if __name__ == "__main__":  # 如果当前脚本作为主程序运行
    main()  # 执行主函数