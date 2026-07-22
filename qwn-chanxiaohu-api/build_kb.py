import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

KB_DIR = "kb"
VECTORSTORE_DIR = "vectorstore"


def load_documents():
    loader = DirectoryLoader(
        KB_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " "]
    )
    return text_splitter.split_documents(documents)


def build_vectorstore():
    documents = load_documents()
    print(f"加载文档数: {len(documents)}")

    split_docs = split_documents(documents)
    print(f"切分后片段数: {len(split_docs)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
#HuggingFaceEmbeddings这种模型是怎么训练的
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"向量库已保存到: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()