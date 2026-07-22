import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

VECTORSTORE_DIR = "vectorstore"

if not DASHSCOPE_API_KEY:
    raise ValueError("请在 .env 中配置 DASHSCOPE_API_KEY")


app = FastAPI(title="公司制度问答机器人-LangChain版")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


class ChatResponse(BaseModel):
    question: str
    answer: str
    references: List[str]


def get_llm():
    return ChatOpenAI(
        model=QWEN_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


SYSTEM_TEMPLATE = """
你是一个公司制度问答助手，职责是根据公司制度内容回答员工问题。

请严格遵守以下规则：
1. 必须仅依据提供的制度片段回答，不得编造。
2. 如果制度片段中没有明确依据，请回答：
“根据当前提供的制度内容，未查询到明确规定，建议咨询HR或直属主管。”
3. 回答应简洁、准确、正式。
4. 若制度中包含时间、条件、审批流程，请优先分点说明。
"""

USER_TEMPLATE = """
员工问题：
{question}

参考制度内容：
{context}

请严格根据参考制度内容回答。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", USER_TEMPLATE),
    ]
)


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = get_llm()


@app.get("/")
def home():
    return {"message": "公司制度聊天机器人 LangChain 版运行中"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    docs = vectorstore.similarity_search(req.question, k=req.top_k)
    references = [doc.page_content for doc in docs]

    context = "\n\n".join(
        [f"【参考{i+1}】{doc.page_content}" for i, doc in enumerate(docs)]
    )

    chain = prompt | llm
    result = chain.invoke(
        {
            "question": req.question,
            "context": context if context else "无"
        }
    )

    answer = result.content if hasattr(result, "content") else str(result)

    return ChatResponse(
        question=req.question,
        answer=answer,
        references=references
    )