import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ---- 1. 连接向量库与检索器 ----
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
retriever = db.as_retriever(search_type="similarity",
                            search_kwargs={"k": 3})

# ---- 2. 连接大模型 ----
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2,   # 问答场景调低，让答案稳定、少发挥
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ---- 3. 设计 Prompt（面试可直接展示）----
PROMPT = ChatPromptTemplate.from_template(
"""你是一个严谨的知识库问答助手。请严格根据【参考资料】回答用户问题。
要求：
1. 答案必须完全来自参考资料，禁止编造资料中没有的内容；
2. 如果参考资料不足以回答问题，请直接回答：抱歉，知识库中没有相关信息；
3. 回答末尾用 [1][2] 的形式标注引用了哪几条参考资料。

【参考资料】
{context}

【用户问题】
{question}""")


def format_docs(docs):
    """给每个片段编号 [1][2]，方便模型引用"""
    return "\n\n".join(
        f"[{i+1}] {d.page_content}"
        for i, d in enumerate(docs))


# ---- 4. 用 LCEL 把各环节串成链（| 表示数据依次流过）----
chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
    | PROMPT
    | llm
)

if __name__ == "__main__":
    SCORE_THRESHOLD = 0.9   # L2 距离阈值：分数越小越相关，大于它就判定知识库没有
    while True:
        q = input("\n请输入问题（输入 q 退出）：")
        if q.strip().lower() == "q":
            break

        # 1. 检索，同时拿到每个片段的“距离分数”（L2，越小越相关）
        scored = db.similarity_search_with_score(q, k=3)
        sources = [doc for doc, _ in scored]
        top_score = scored[0][1]

        # 2. 第一道防线：最相关片段都太远 → 直接拒答，不调用大模型、不展示来源
        if top_score > SCORE_THRESHOLD:
            print("\n回答： 抱歉，知识库中没有相关信息。")
            print("（未检索到足够相关的内容，无引用来源）")
            continue

        # 3. 通过阈值，才调用问答链生成答案
        answer = chain.invoke(q)
        print("\n回答：", answer.content)

        # 4. 第二道防线：模型若仍判断资料不足 → 也不展示来源
        if "知识库中没有" in answer.content:
            print("（无有效引用来源）")
            continue

        # 5. 真正作答了才打印来源；PDF 显示页码，md 不显示空页码
        print("\n引用来源：")
        for i, d in enumerate(sources):
            src = d.metadata.get("source", "")
            page = d.metadata.get("page")
            if page is None:
                print(f"[{i+1}] {src}")
            else:
                print(f"[{i+1}] {src} 第{page + 1}页")
