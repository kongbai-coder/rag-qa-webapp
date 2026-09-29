import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"


def load_documents(data_dir):
    """遍历 data 目录，按文件类型加载，返回 Document 列表"""
    docs = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(root, name)
            lower = name.lower()
            if lower.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
            elif lower.endswith((".txt", ".md")):
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif lower.endswith(".docx"):
                docs.extend(Docx2txtLoader(path).load())
    return docs


# 1. 加载
raw_docs = load_documents(DATA_DIR)
print(f"加载文档页数/段落数：{len(raw_docs)}")

# 2. 切片：按段落→句子→空格的优先级递归切，尽量不切断语义
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
)
chunks = splitter.split_documents(raw_docs)
print(f"切片总数：{len(chunks)}")

# 3. 向量化并入库
embeddings = OpenAIEmbeddings(
    model="qwen3.7-text-embedding",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,  # 百炼不接受 token 数字，必须直接传文字
    chunk_size=20,                    # qwen3.7 每批最多 20 条
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR,
    collection_name="knowledge_base",
)
print("入库完成，向量库保存在：", DB_DIR)
