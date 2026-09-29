import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
app = FastAPI(title="RAG 知识库问答")

# 允许前端跨域访问（开发阶段放开，生产环境应限定来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 复用向量库与检索器 ----
embeddings = OpenAIEmbeddings(
    model="qwen3.7-text-embedding",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
    chunk_size=20,
)
db = Chroma(persist_directory="chroma_db",
            embedding_function=embeddings,
            collection_name="knowledge_base")
retriever = db.as_retriever(search_kwargs={"k": 5})

# ---- 大模型：开启流式 ----
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2,
    streaming=True,        # 流式开关：让模型一边生成一边往外吐
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

CONV_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是严谨的知识库问答助手，答案必须来自参考资料，"
     "资料不足时回答：抱歉，知识库中没有相关信息；"
     "回答末尾用[1][2]标注引用。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "参考资料：\n{context}\n\n问题：{question}"),
])



def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {d.page_content}"
        for i, d in enumerate(docs))


rag_chain = (
    {"context": itemgetter("question") | retriever | format_docs,
     "question": itemgetter("question"),
     "history": itemgetter("history")}
    | CONV_PROMPT | llm
)


# ---- 请求体的数据结构 ----
class ChatReq(BaseModel):
    question: str
    history: list = []      


def to_messages(history):
    """把前端传来的历史转成 LangChain 消息对象，只保留最近 4 轮（8 条）"""
    msgs = []
    for h in history[-8:]:
        if h["role"] == "user":
            msgs.append(HumanMessage(content=h["content"]))
        else:
            msgs.append(AIMessage(content=h["content"]))
    return msgs


def event_stream(question, history):
    """把链的流式输出包装成 SSE 格式：每条 data: ... 空行结尾"""
    input_data = {
        "question": question,
        "history": to_messages(history),
    }
    for chunk in rag_chain.stream(input_data):
        if chunk.content:
            data = json.dumps(chunk.content, ensure_ascii=False)
            yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat")
def chat(req: ChatReq):
    return StreamingResponse(
        event_stream(req.question, req.history),
        media_type="text/event-stream")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = os.path.join("data", file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # 在这里调用 ingest 的增量入库逻辑，上传即可问答
    return {"filename": file.filename, "status": "saved"}


@app.get("/")
def index():
    return FileResponse("../frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
