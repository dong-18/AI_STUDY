import json  # 导入 json，用于读取 jsonl 数据
from tokenizer import CharTokenizer  # 导入我们自己写的字符级 tokenizer

texts = []  # 初始化文本列表，用来收集所有文本

with open("data/pretrain.txt", "r", encoding="utf-8") as f:  # 打开预训练文本
    for line in f:  # 遍历每一行
        #line = line.strip()  # 去掉首尾空白
        if line:  # 如果这一行非空
            texts.append(line)  # 加入文本列表

with open("data/sft.jsonl", "r", encoding="utf-8") as f:  # 打开 SFT 数据
    for line in f:  # 遍历每一行
        obj = json.loads(line)  # 解析一条 json
        texts.append(obj["question"])  # 加入问题文本
        texts.append(obj["answer"])  # 加入答案文本

with open("data/dpo.jsonl", "r", encoding="utf-8") as f:  # 打开 DPO 数据
    for line in f:  # 遍历每一行
        obj = json.loads(line)  # 解析一条 json
        texts.append(obj["prompt"])  # 加入 prompt 文本
        texts.append(obj["chosen"])  # 加入 chosen 文本
        texts.append(obj["rejected"])  # 加入 rejected 文本

tokenizer = CharTokenizer(texts=texts)  # 用所有文本构建字符级词表
tokenizer.save("data/vocab.json")  # 保存词表到文件

print("vocab size =", tokenizer.vocab_size())  # 打印词表大小
print("saved tokenizer to data/vocab.json")  # 打印保存提示