import torch  # 导入 PyTorch
from tokenizer import CharTokenizer  # 导入 tokenizer
from model import MiniGPT  # 导入模型

SYSTEM_PROMPT = "你是公司制度助手，请严格根据制度回答，不知道就明确说不知道。"  # 定义系统提示语


def load_model(ckpt_path, device):  # 定义模型加载函数
    tokenizer = CharTokenizer.load("data/vocab.json")  # 加载 tokenizer

    model = MiniGPT(  # 创建模型结构
        vocab_size=tokenizer.vocab_size(),  # 词表大小
        max_seq_len=128,  # 最大长度
        d_model=256,  # hidden dim
        n_heads=4,  # 头数
        n_layers=4,  # 层数
        dropout=0.1,  # dropout
    ).to(device)  # 放到设备上

    model.load_state_dict(torch.load(ckpt_path, map_location=device))  # 加载参数
    model.eval()  # 切换到评估模式

    return tokenizer, model  # 返回 tokenizer 和 model


def main():  # 主函数
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

    ckpt_path = "sft_ckpt.pt"  # 默认使用 SFT 模型，也可以改成 dpo_ckpt.pt
    tokenizer, model = load_model(ckpt_path, device)  # 加载模型和 tokenizer

    question = "公司禁止员工做哪些损害公司事情？"  # 你可以改这里测试不同问题
    prompt = f"{SYSTEM_PROMPT}\n用户：{question}\n助手："  # 构造输入 prompt

    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)  # 编码 prompt
    x = torch.tensor([input_ids], dtype=torch.long).to(device)  # 增加 batch 维并放到设备上

    y = model.generate(  # 调用生成函数
        x,  # 输入上下文
        max_new_tokens=50,  # 最多生成 50 个 token
        eos_token_id=tokenizer.eos_token_id,  # 遇到 eos 提前停止
        temperature=1.0,  # 温度参数
    )

    output_text = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)  # 解码生成结果
    print("===== MODEL OUTPUT =====")  # 打印分隔线
    print(output_text)  # 打印完整输出


if __name__ == "__main__":  # 如果作为主程序运行
    main()  # 执行主函数