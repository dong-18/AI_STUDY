from pathlib import Path

import torch
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# ============================================================
# 基本配置
# ============================================================

# 所有本地路径都以 main.py 所在目录为基准，避免切换工作目录后找不到文件。
APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "company_docs"

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = (
    APP_DIR.parent
    / "qwn-0.5b-chanxiaohu-weitiao-lora"
    / "company_lora"
    / "output"
    / "company-qwen-lora"
)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# LoRA 训练时完整序列最大长度为 512。这里限制检索上下文，尽量让
# 推理输入长度与训练阶段接近，同时始终保留员工问题。
MAX_CONTEXT_TOKENS = 320
RETRIEVAL_TOP_K = 2
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = """你是公司制度问答助手。

回答要求：
1. 只能根据用户提供的制度资料回答。
2. 不得编造制度中没有的金额、日期、条件或流程。
3. 回答时尽量说明制度名称或条款。
4. 如果资料中没有答案，明确说明“现有制度资料中没有找到相关规定”。
5. 回答应准确、简洁，不要加入与问题无关的内容。
"""


# ============================================================
# 1. 读取并切分公司制度
# ============================================================

if not DOCS_DIR.exists():
    raise FileNotFoundError(f"找不到公司制度目录：{DOCS_DIR}")

loader = DirectoryLoader(
    str(DOCS_DIR),
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs = loader.load()
if not docs:
    raise ValueError(f"公司制度目录中没有找到 txt 文件：{DOCS_DIR}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=220,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "；", "，"],
)
chunks = splitter.split_documents(docs)
if not chunks:
    raise ValueError("公司制度文档没有生成有效切片。")

print(f"制度文档数：{len(docs)}")
print(f"制度切片数：{len(chunks)}")


# ============================================================
# 2. 创建 Embedding 和 FAISS 索引
# ============================================================

# Embedding 放在 CPU，给 3GB 显卡尽量保留模型推理显存。
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vector_db = FAISS.from_documents(chunks, embedding_model)


# ============================================================
# 3. 加载基础模型和 LoRA 适配器
# ============================================================

if not LORA_PATH.exists():
    raise FileNotFoundError(f"找不到 LoRA 适配器目录：{LORA_PATH}")
if not (LORA_PATH / "adapter_config.json").exists():
    raise FileNotFoundError(f"LoRA 目录缺少 adapter_config.json：{LORA_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
print(f"推理设备：{device}")
print(f"LoRA 路径：{LORA_PATH}")

# 使用 LoRA 目录中的 tokenizer，确保聊天模板与训练保存结果一致。
tokenizer = AutoTokenizer.from_pretrained(str(LORA_PATH), use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

# 在基础模型上挂载训练得到的 PEFT LoRA 权重。
model = PeftModel.from_pretrained(
    base_model,
    str(LORA_PATH),
).to(device)
model.eval()
model.config.use_cache = True


# ============================================================
# 4. 检索与生成
# ============================================================

def limit_context_tokens(context: str) -> str:
    """限制检索上下文长度，避免大量条款挤占问题和回答空间。"""
    context_ids = tokenizer(
        context,
        add_special_tokens=False,
    )["input_ids"]
    if len(context_ids) <= MAX_CONTEXT_TOKENS:
        return context

    return tokenizer.decode(
        context_ids[:MAX_CONTEXT_TOKENS],
        skip_special_tokens=True,
    )


def ask_company(question: str) -> str:
    """检索相关制度，并使用 LoRA 模型生成仅依据制度的答案。"""
    question = question.strip()
    if not question:
        return "问题不能为空。"

    # 第一步：从制度库中检索最相关的条款。
    results = vector_db.similarity_search(
        question,
        k=min(RETRIEVAL_TOP_K, len(chunks)),
    )
    context = "\n\n".join(
        f"【资料 {index}】\n{document.page_content}"
        for index, document in enumerate(results, start=1)
    )
    context = limit_context_tokens(context)

    # 第二步：使用与 LoRA 训练阶段一致的消息结构和 Chat Template。
    user_content = (
        "请根据下面的公司制度资料回答问题。\n\n"
        f"【制度资料】\n{context}\n\n"
        f"【员工问题】\n{question}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = {
        key: value.to(device)
        for key, value in model_inputs.items()
    }

    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 第三步：生成答案。generate 的输出包含输入，因此解码前移除输入 token。
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            generation_config=generation_config,
        )

    input_length = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return answer or "现有制度资料中没有找到相关规定。"


# ============================================================
# 5. 命令行问答
# ============================================================

def main() -> None:
    print("\n公司制度问答已启动，输入 exit 退出。")
    while True:
        question = input("\n问题：").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("\n回答：")
        print(ask_company(question))


if __name__ == "__main__":
    main()
