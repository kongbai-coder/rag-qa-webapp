"""一次性脚本：抓取 LangChain 官方中文镜像的教程页面，转成 markdown 存入 data 目录。"""
import os
import re
import urllib.request

import html2text
from bs4 import BeautifulSoup

BASE = "https://python.langchain.ac.cn/docs/tutorials/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# slug -> 中文文件名
PAGES = {
    "llm_chain": "LangChain教程-构建简单LLM应用",
    "chatbot": "LangChain教程-构建聊天机器人",
    "retrievers": "LangChain教程-检索器",
    "rag": "LangChain教程-构建检索增强生成RAG应用",
    "qa_chat_history": "LangChain教程-带聊天历史的问答",
    "agents": "LangChain教程-构建智能体Agent",
    "documents": "LangChain教程-文档加载",
    "text_splitters": "LangChain教程-文本切分器",
    "embeddings": "LangChain教程-文本嵌入",
    "vectorstores": "LangChain教程-向量存储",
    "local_rag": "LangChain教程-本地RAG应用",
    "query_analysis": "LangChain教程-查询分析",
    "sql-qa": "LangChain教程-基于SQL的问答",
    "extraction": "LangChain教程-信息提取",
}


def make_converter():
    h = html2text.HTML2Text()
    h.body_width = 0          # 不按宽度强行换行
    h.ignore_images = True    # 知识库只取文字
    h.protect_links = True
    h.wrap_links = False
    return h


def fetch_html(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")


def extract_article(html):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if article is None:
        article = soup.find("main")
    # 去掉与正文无关的元素
    for sel in ["nav", "footer", "script", "style", "button", "svg",
                ".pagination-nav", ".theme-edit-this-page", ".tocCollapsible",
                ".table-of-contents", "header"]:
        for el in article.select(sel):
            el.decompose()
    return article


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    h = make_converter()
    for slug, cn_name in PAGES.items():
        url = BASE + slug + "/"
        try:
            html = fetch_html(url)
            article = extract_article(html)
            md = h.handle(str(article))
            # 清理多余空行
            md = re.sub(r"\n{3,}", "\n\n", md).strip()
            header = f"# {cn_name}\n\n> 资料来源：LangChain 官方中文文档 {url}\n\n"
            out_path = os.path.join(DATA_DIR, cn_name + ".md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(header + md + "\n")
            print(f"已生成: {cn_name}.md  ({len(md)} 字)")
        except Exception as e:
            print(f"失败: {slug} -> {e}")


if __name__ == "__main__":
    main()
