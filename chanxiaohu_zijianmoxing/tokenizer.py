import json  # 导入 json，用于保存和加载词表


class CharTokenizer:  # 定义一个最简单的字符级分词器
    def __init__(self, texts=None, vocab=None):  # 构造函数，支持从文本构建词表或直接加载词表
        self.pad_token = "<pad>"  # 定义 padding token
        self.bos_token = "<bos>"  # 定义句子开始 token
        self.eos_token = "<eos>"  # 定义句子结束 token
        self.unk_token = "<unk>"  # 定义未知字符 token

        if vocab is not None:  # 如果外部已经传入了词表
            self.id2token = vocab  # 直接保存 id 到 token 的映射
            self.token2id = {t: i for i, t in enumerate(vocab)}  # 构造 token 到 id 的映射
        else:  # 如果没有传入现成词表
            assert texts is not None  # 必须提供文本列表
            chars = set()  # 用集合收集所有出现过的字符
            for text in texts:  # 遍历所有文本
                for ch in text:  # 遍历文本中的每个字符
                    chars.add(ch)  # 把字符加入集合

            vocab = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]  # 先放特殊 token
            vocab += sorted(list(chars))  # 再追加排序后的普通字符词表

            self.id2token = vocab  # 保存 id 到 token 的映射
            self.token2id = {t: i for i, t in enumerate(vocab)}  # 保存 token 到 id 的映射

        self.pad_token_id = self.token2id[self.pad_token]  # 缓存 pad 的 id
        self.bos_token_id = self.token2id[self.bos_token]  # 缓存 bos 的 id
        self.eos_token_id = self.token2id[self.eos_token]  # 缓存 eos 的 id
        self.unk_token_id = self.token2id[self.unk_token]  # 缓存 unk 的 id

    def encode(self, text, add_bos=False, add_eos=False):  # 把字符串编码成 token id 列表
        ids = []  # 初始化结果列表
        if add_bos:  # 如果需要加 bos
            ids.append(self.bos_token_id)  # 先加 bos id
        for ch in text:  # 遍历文本中每个字符
            ids.append(self.token2id.get(ch, self.unk_token_id))  # 查词表，没有就用 unk
        if add_eos:  # 如果需要加 eos
            ids.append(self.eos_token_id)  # 末尾加 eos id
        return ids  # 返回编码结果

    def decode(self, ids, skip_special_tokens=True):  # 把 token id 列表解码回字符串
        special = {  # 定义特殊 token id 集合
            self.pad_token_id,  # pad id
            self.bos_token_id,  # bos id
            self.eos_token_id,  # eos id
            self.unk_token_id,  # unk id
        }
        chars = []  # 初始化字符结果列表
        for i in ids:  # 遍历每个 id
            if skip_special_tokens and i in special:  # 如果跳过特殊 token 且当前是特殊 token
                continue  # 直接跳过
            chars.append(self.id2token[i])  # 否则把对应 token 加入结果
        return "".join(chars)  # 拼接成字符串返回

    def vocab_size(self):  # 返回词表大小
        return len(self.id2token)  # 词表长度就是大小

    def save(self, path):  # 保存词表到文件
        with open(path, "w", encoding="utf-8") as f:  # 以写模式打开文件
            json.dump(self.id2token, f, ensure_ascii=False, indent=2)  # 保存 id2token 列表

    @classmethod
    def load(cls, path):  # 从文件加载词表
        with open(path, "r", encoding="utf-8") as f:  # 以读模式打开文件
            vocab = json.load(f)  # 读取词表列表
        return cls(vocab=vocab)  # 用词表直接构造 tokenizer