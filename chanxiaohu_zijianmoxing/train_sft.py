import json  # 导入 json，用于读取 jsonl
import torch  # 导入 PyTorch
import torch.optim as optim  # 导入优化器
from torch.utils.data import Dataset, DataLoader  # 导入 Dataset 和 DataLoader
from tokenizer import CharTokenizer  # 导入 tokenizer
from model import MiniGPT  # 导入模型

SYSTEM_PROMPT = "你是公司制度助手，请严格根据制度回答，不知道就明确说不知道。"  # 定义系统提示词


class SFTDataset(Dataset):  # 定义 SFT 数据集
    def __init__(self, path, tokenizer, max_seq_len=128):  # 构造函数
        self.samples = []  # 初始化样本列表
        self.tokenizer = tokenizer  # 保存 tokenizer
        self.max_seq_len = max_seq_len  # 保存最大序列长度

        with open(path, "r", encoding="utf-8") as f:  # 打开 SFT 数据文件
            for line in f:  # 遍历每一行
                obj = json.loads(line)  # 解析 json
                q = obj["question"]  # 取问题
                a = obj["answer"]  # 取答案

                prompt = f"{SYSTEM_PROMPT}\n用户：{q}\n助手："  # 构造 prompt 文本
                answer = a  # 答案文本单独保存

                prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)  # 编码 prompt，前面加 bos
                answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)  # 编码答案，末尾加 eos

                full_ids = prompt_ids + answer_ids  # 拼接成完整训练序列
                full_ids = full_ids[:max_seq_len]  # 截断到最大长度

                input_ids = full_ids[:-1]  # 输入是除最后一个 token 外的所有 token
                labels = full_ids[1:]  # 标签是右移一位后的序列

                prompt_len = max(0, min(len(prompt_ids) - 1, len(labels)))  # 计算 prompt 对应的 label 区间长度
                masked_labels = labels[:]  # 复制一份 labels，后面用于做 mask

                for i in range(prompt_len):  # 遍历 prompt 对应的 label 位置
                    masked_labels[i] = -100  # 把 prompt 部分 label 设为 -100，不参与 loss

                self.samples.append((input_ids, masked_labels))  # 保存一条处理好的样本

    def __len__(self):  # 返回样本数
        return len(self.samples)  # 返回总样本数

    def __getitem__(self, idx):  # 取单条样本
        input_ids, labels = self.samples[idx]  # 取出输入和标签
        return {  # 返回字典
            "input_ids": torch.tensor(input_ids, dtype=torch.long),  # 输入 token ids
            "labels": torch.tensor(labels, dtype=torch.long),  # 标签 ids，其中 prompt 部分已被 mask
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),  # 非 pad 位置全部置 1
        }


def collate_fn(batch, pad_token_id):  # 自定义拼 batch 函数
    max_len = max(len(x["input_ids"]) for x in batch)  # 找到当前 batch 中的最大长度

    input_ids = []  # 存放拼接后的 input_ids
    labels = []  # 存放拼接后的 labels
    attention_mask = []  # 存放拼接后的 attention_mask

    for x in batch:  # 遍历 batch 中每个样本
        l = len(x["input_ids"])  # 当前样本长度
        pad_len = max_len - l  # 计算需要补齐的长度

        input_ids.append(  # 补齐 input_ids
            torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])  # 用 pad 补齐
        )

        labels.append(  # 补齐 labels
            torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])  # 用 -100 补齐
        )

        attention_mask.append(  # 补齐 attention_mask
            torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])  # pad 位置设为 0
        )

    return {  # 返回拼好的 batch
        "input_ids": torch.stack(input_ids),  # [B, T]
        "labels": torch.stack(labels),  # [B, T]
        "attention_mask": torch.stack(attention_mask),  # [B, T]
    }


def main():  # 主函数
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择运行设备
    tokenizer = CharTokenizer.load("data/vocab.json")  # 加载词表

    dataset = SFTDataset("data/sft.jsonl", tokenizer, max_seq_len=128)  # 构建 SFT 数据集

    loader = DataLoader(  # 构建数据加载器
        dataset,  # 数据集
        batch_size=8,  # batch 大小
        shuffle=True,  # 打乱
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),  # 自定义 collate
    )

    model = MiniGPT(  # 创建模型实例
        vocab_size=tokenizer.vocab_size(),  # 词表大小
        max_seq_len=128,  # 最大序列长度
        d_model=256,  # 隐藏维度
        n_heads=4,  # 注意力头数
        n_layers=4,  # Transformer 层数
        dropout=0.1,  # dropout 概率
    ).to(device)  # 放到设备上

    model.load_state_dict(torch.load("pretrain_ckpt.pt", map_location=device))  # 加载预训练阶段的参数

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # 创建 SFT 优化器

    model.train()  # 切换到训练模式
    for epoch in range(100):  # 训练 10 个 epoch
        total_loss = 0.0  # 初始化总 loss

        for step, batch in enumerate(loader):  # 遍历每个 batch
            batch = {k: v.to(device) for k, v in batch.items()}  # 把 batch 放到对应设备

            out = model(**batch)  # 前向传播
            loss = out["loss"]  # 取出损失

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加 loss

            if step % 10 == 0:  # 每 10 步打印一次
                print(f"[SFT] epoch={epoch} step={step} loss={loss.item():.4f}")  # 打印当前损失

        avg_loss = total_loss / max(len(loader), 1)  # 计算平均 loss
        print(f"[SFT] epoch={epoch} avg_loss={avg_loss:.4f}")  # 打印平均 loss

    torch.save(model.state_dict(), "sft_ckpt.pt")  # 保存 SFT 模型参数
    print("saved sft checkpoint to sft_ckpt.pt")  # 打印保存提示


if __name__ == "__main__":  # 如果脚本直接运行
    main()  # 执行主函数