import json  # 导入 json，用于读 jsonl
import torch  # 导入 PyTorch
import torch.optim as optim  # 导入优化器
import torch.nn.functional as F  # 导入函数接口
from torch.utils.data import Dataset, DataLoader  # 导入数据集与加载器
from tokenizer import CharTokenizer  # 导入 tokenizer
from model import MiniGPT  # 导入模型

SYSTEM_PROMPT = "你是公司制度助手，请严格根据制度回答，不知道就明确说不知道。"  # 定义系统 prompt


class DPODataset(Dataset):  # 定义 DPO 数据集
    def __init__(self, path):  # 构造函数
        self.samples = []  # 初始化样本列表
        with open(path, "r", encoding="utf-8") as f:  # 打开 DPO 数据文件
            for line in f:  # 遍历每一行
                obj = json.loads(line)  # 解析 json
                self.samples.append(obj)  # 保存样本

    def __len__(self):  # 返回样本数
        return len(self.samples)  # 数据集大小

    def __getitem__(self, idx):  # 根据索引取样本
        return self.samples[idx]  # 返回一条字典样本


def build_input_and_labels(tokenizer, prompt_text, answer_text, max_seq_len):  # 构造 prompt+answer 的输入和标签
    prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)  # 编码 prompt
    answer_ids = tokenizer.encode(answer_text, add_bos=False, add_eos=True)  # 编码 answer，末尾加 eos

    full_ids = prompt_ids + answer_ids  # 拼接为整段序列
    full_ids = full_ids[:max_seq_len]  # 截断

    input_ids = full_ids[:-1]  # 输入是去掉最后一个 token
    labels = full_ids[1:]  # 标签是右移后的序列

    prompt_len = max(0, min(len(prompt_ids) - 1, len(labels)))  # 计算 prompt 对应的 label 长度
    masked_labels = labels[:]  # 复制 labels

    for i in range(prompt_len):  # 遍历 prompt 对应的位置
        masked_labels[i] = -100  # prompt 部分 label 置为 -100，表示不参与计算

    attention_mask = [1] * len(input_ids)  # 所有非 pad 位置都设为有效

    return (  # 返回三个张量
        torch.tensor(input_ids, dtype=torch.long),  # 输入张量
        torch.tensor(masked_labels, dtype=torch.long),  # 掩码后的标签张量
        torch.tensor(attention_mask, dtype=torch.long),  # attention mask 张量
    )


def seq_logprob(model, input_ids, labels, attention_mask):  # 计算一条回答在模型下的总 log-prob
    out = model(  # 前向传播
        input_ids=input_ids.unsqueeze(0),  # 增加 batch 维
        attention_mask=attention_mask.unsqueeze(0),  # 增加 batch 维
        labels=None,  # 不直接调用 CE loss
    )

    logits = out["logits"][0]  # 取出 batch 中第一条样本的 logits，形状 [T, V]
    log_probs = F.log_softmax(logits, dim=-1)  # 对词表维做 log-softmax

    valid = labels != -100  # 找到有效 label 位置，也就是 answer token 部分
    target = labels[valid]  # 取出有效目标 token id
    token_log_probs = log_probs[valid, :].gather(1, target.unsqueeze(1)).squeeze(1)  # 取出每个目标 token 对应的 log-prob

    return token_log_probs.sum()  # 把回答部分的 token log-prob 求和，得到整个回答的 log-prob


def main():  # 主函数
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
    tokenizer = CharTokenizer.load("data/vocab.json")  # 加载词表

    dataset = DPODataset("data/dpo.jsonl")  # 构建 DPO 数据集
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  # DPO 这里为了简单，batch 先设为 1

    policy_model = MiniGPT(  # 创建 policy 模型
        vocab_size=tokenizer.vocab_size(),  # 词表大小
        max_seq_len=128,  # 最大长度
        d_model=256,  # hidden dim
        n_heads=4,  # 头数
        n_layers=4,  # 层数
        dropout=0.1,  # dropout
    ).to(device)  # 放到设备上
    policy_model.load_state_dict(torch.load("sft_ckpt.pt", map_location=device))  # 加载 SFT 模型参数

    ref_model = MiniGPT(  # 创建 reference 模型
        vocab_size=tokenizer.vocab_size(),  # 词表大小
        max_seq_len=128,  # 最大长度
        d_model=256,  # hidden dim
        n_heads=4,  # 头数
        n_layers=4,  # 层数
        dropout=0.1,  # dropout
    ).to(device)  # 放到设备上
    ref_model.load_state_dict(torch.load("sft_ckpt.pt", map_location=device))  # reference 一般初始化为同一个 SFT 模型
    ref_model.eval()  # 参考模型固定，不训练

    optimizer = optim.AdamW(policy_model.parameters(), lr=5e-6)  # DPO 通常学习率较小
    beta = 0.1  # DPO 温度超参数

    policy_model.train()  # 切换 policy 模型到训练模式
    for epoch in range(5):  # 训练 5 个 epoch
        total_loss = 0.0  # 初始化总损失

        for step, batch in enumerate(loader):  # 遍历每个样本
            prompt = batch["prompt"][0]  # 取 prompt 文本
            chosen = batch["chosen"][0]  # 取偏好更好的回答
            rejected = batch["rejected"][0]  # 取偏好更差的回答

            full_prompt = f"{SYSTEM_PROMPT}\n用户：{prompt}\n助手："  # 拼出完整 prompt

            c_input_ids, c_labels, c_mask = build_input_and_labels(tokenizer, full_prompt, chosen, 128)  # 构造 chosen 输入
            r_input_ids, r_labels, r_mask = build_input_and_labels(tokenizer, full_prompt, rejected, 128)  # 构造 rejected 输入

            c_input_ids = c_input_ids.to(device)  # 把 chosen 输入放到设备上
            c_labels = c_labels.to(device)  # 把 chosen 标签放到设备上
            c_mask = c_mask.to(device)  # 把 chosen mask 放到设备上

            r_input_ids = r_input_ids.to(device)  # 把 rejected 输入放到设备上
            r_labels = r_labels.to(device)  # 把 rejected 标签放到设备上
            r_mask = r_mask.to(device)  # 把 rejected mask 放到设备上

            pi_c = seq_logprob(policy_model, c_input_ids, c_labels, c_mask)  # 计算 policy 对 chosen 的总 log-prob
            pi_r = seq_logprob(policy_model, r_input_ids, r_labels, r_mask)  # 计算 policy 对 rejected 的总 log-prob

            with torch.no_grad():  # 参考模型不需要梯度
                ref_c = seq_logprob(ref_model, c_input_ids, c_labels, c_mask)  # 计算 ref 对 chosen 的 log-prob
                ref_r = seq_logprob(ref_model, r_input_ids, r_labels, r_mask)  # 计算 ref 对 rejected 的 log-prob

            logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))  # 按 DPO 公式构造 logit
            loss = -F.logsigmoid(logits)  # 计算 DPO 损失

            optimizer.zero_grad()  # 清空旧梯度

            loss.backward()  # 反向传播
            optimizer.step()  # 更新 policy 参数

            total_loss += loss.item()  # 累加损失

            if step % 10 == 0:  # 每 10 步打印一次
                print(f"[DPO] epoch={epoch} step={step} loss={loss.item():.4f}")  # 打印当前损失

        avg_loss = total_loss / max(len(loader), 1)  # 计算平均损失
        print(f"[DPO] epoch={epoch} avg_loss={avg_loss:.4f}")  # 打印 epoch 平均损失

    torch.save(policy_model.state_dict(), "dpo_ckpt.pt")  # 保存 DPO 后的模型参数
    print("saved dpo checkpoint to dpo_ckpt.pt")  # 打印保存提示


if __name__ == "__main__":  # 如果作为主程序运行
    main()  # 执行主函数