import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate


# =========================
# 1. 加载公司制度
# =========================

loader = DirectoryLoader(
    "./company_docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding":"utf-8"
    }
)

docs = loader.load()


# =========================
# 2. 文档切片
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        "。",
        "；",
        "，"
    ]
)

chunks = splitter.split_documents(docs)


print("切片数量:",len(chunks))


# =========================
# 3. 创建Embedding
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name=
    "BAAI/bge-small-zh-v1.5"
)


# =========================
# 4. 建立FAISS
# =========================

vector_db = FAISS.from_documents(
    chunks,
    embedding_model
)


# =========================
# 5. 加载0.5B模型
# =========================


model_name = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to("cuda")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)



# =========================
# 6. Prompt模板
# =========================

prompt = PromptTemplate(
    template="""

你是公司的制度助手。

只能根据下面制度回答。

制度资料:
{context}


员工问题:
{question}


如果资料没有答案，请说：
没有找到相关制度。

回答:
""",

input_variables=[
    "context",
    "question"
]
)



# =========================
# 7. LoopChain
# =========================

def ask_company(question):


    # step1 检索
    results = vector_db.similarity_search(
        question,
        k=3
    )


    context="\n".join(
        [
            x.page_content
            for x in results
        ]
    )


    # step2 拼prompt

    final_prompt = prompt.format(
        context=context,
        question=question
    )


    # step3 LLM生成

    output = generator(
        final_prompt,
        max_new_tokens=512,
        temperature=0.1
    )


    answer = output[0]["generated_text"]


    return answer



# =========================
# 8. 测试
# =========================


while True:

    q=input("\n问题:")

    if q=="exit":
        break


    print(
        ask_company(q)
    )