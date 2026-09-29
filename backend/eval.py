# -*- coding: utf-8 -*-
"""
- 评测脚本：批量跑 50 条问题，生成 Excel 评测表。
- 自动填入：编号 / 问题 / 类型 / 期望要点 / 实际回答 / 检索来源
- 人工填写：G列「答案正确?」、H列「引用正确?」（下拉选 是/否）
- 「指标汇总」表用公式自动算三个指标，填完即更新。
运行（在 backend 目录、已激活虚拟环境）：python eval.py
"""
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

load_dotenv()

# ---- 组件（与 app.py 保持一致，评测单轮、history 为空）----
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
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2,
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
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


rag_chain = (
    {"context": itemgetter("question") | retriever | format_docs,
     "question": itemgetter("question"),
     "history": itemgetter("history")}
    | CONV_PROMPT | llm
)

# ---- 评测集：(问题, 类型, 期望要点)。应答 35 条、拒答 15 条 ----
QUESTIONS = [
    # ===== 应答类（资料里有答案）35 条 =====
    ("LangChain中一个最简单的LLM应用由哪几部分组成？", "应答",
     "提示词模板、语言模型（聊天模型）、输出解析器"),
    ("什么是提示词模板（PromptTemplate）？", "应答",
     "含模板字符串和输入变量，把用户输入填入固定提示词"),
    ("LangChain如何让大模型流式输出？", "应答",
     "调用.stream()逐块/逐token获取输出（streaming）"),
    ("什么是检索增强生成（RAG）？", "应答",
     "先从外部知识库检索相关资料、再交给大模型生成答案；结合检索与生成、减少幻觉"),
    ("RAG的索引阶段包含哪些步骤？", "应答",
     "加载文档、切分文档、嵌入并存入向量库"),
    ("RAG的检索与生成阶段大致流程是什么？", "应答",
     "问题检索相关片段→把片段和问题填入提示词→大模型生成答案"),
    ("文档为什么要切分成小块（chunk）？", "应答",
     "长文档无法整体嵌入/检索；切块提高检索精度、适配上下文长度"),
    ("RAG应用如何让答案返回引用来源？", "应答",
     "检索文档带元数据（source来源、page页码），随答案一并返回"),
    ("本地RAG应用通常用什么加载本地文档？", "应答",
     "文档加载器Document Loader（如目录加载器、PDF/文本加载器）"),
    ("多轮对话中如何让检索器听懂带代词的追问？", "应答",
     "历史感知检索器；先用对话历史把追问改写成独立问题再检索"),
    ("聊天历史的状态管理主要解决什么问题？", "应答",
     "按会话保存/读取历史消息、按会话ID隔离、控制历史长度"),
    ("什么是智能体（Agent）？", "应答",
     "以LLM为推理引擎，自主决定调用哪些工具、执行行动并按结果循环直到完成任务"),
    ("Agent中的工具（tool）是什么？", "应答",
     "可供模型调用的函数/能力（搜索、计算等），用@tool定义、含名称和描述"),
    ("智能体和普通聊天机器人的核心区别是什么？", "应答",
     "Agent能自主决策并调用外部工具、多步执行；普通聊天机器人主要做端到端对话"),
    ("什么是工具调用（tool calling）？", "应答",
     "模型输出要调用的工具名与参数，程序执行后把结果回传模型继续作答"),
    ("如何给智能体添加记忆？", "应答",
     "用检查点/记忆机制保存对话历史，运行时按会话/线程传入"),
    ("LangChain中有哪几种聊天消息类型？", "应答",
     "HumanMessage人类消息、AIMessage的AI消息、SystemMessage系统消息"),
    ("什么是聊天消息的持久化？", "应答",
     "把对话历史保存到存储（内存/数据库等），跨会话或重启后仍可读取"),
    ("为什么要管理和裁剪对话历史？", "应答",
     "历史太长会超出上下文窗口、增加成本；只保留最近/相关消息"),
    ("什么是文本嵌入（embedding）？", "应答",
     "把文本转成语义向量；语义相近的文本向量距离更近"),
    ("什么是向量存储（vector store）？", "应答",
     "存储文档向量并支持相似度检索的数据库，如Chroma"),
    ("MMR（最大边际相关性）检索解决什么问题？", "应答",
     "平衡相关性与多样性，避免召回片段重复，选出既相关又有差异的内容"),
    ("文档分割常用什么分割器？", "应答",
     "文本分割器TextSplitter，如递归字符文本分割器"),
    ("为什么需要查询分析（query analysis）？", "应答",
     "原始问题可能含混或含多个条件，直接检索效果差，需先改写或结构化"),
    ("查询分析可以生成什么来改善检索？", "应答",
     "结构化查询/元数据过滤条件，或改写后的检索语句"),
    ("信息提取（information extraction）是做什么的？", "应答",
     "从非结构化文本中抽取结构化信息，如实体及其属性"),
    ("信息提取时为什么要定义模式（schema）？", "应答",
     "规定要抽取的字段和类型，让模型按统一结构输出"),
    ("如何一次提取多个实体？", "应答",
     "在模式中定义多个实体/字段，模型一次抽取并返回列表"),
    ("LangChain主要用来做什么？", "应答",
     "构建基于大语言模型的应用，提供模型、提示词、链、检索、Agent等组件与编排"),
    ("什么是线性回归？", "应答",
     "用线性函数拟合输入与输出，最小化预测值与真实值的误差（损失函数）"),
    ("Softmax回归用于解决什么问题？", "应答",
     "多类别分类；softmax把输出转换为各类别的概率分布"),
    ("什么是过拟合？如何缓解？", "应答",
     "训练集好、泛化差；缓解：权重衰减、暂退dropout、更多数据、正则化"),
    ("卷积神经网络的卷积层有什么特点？", "应答",
     "局部连接、权值共享，用卷积核提取局部特征"),
    ("循环神经网络（RNN）适合处理什么数据？", "应答",
     "序列数据（文本、时间序列）；含隐状态、按时间步共享参数"),
    ("什么是注意力机制（attention）？", "应答",
     "为输入各部分动态分配权重、聚焦相关信息；自注意力在序列内部计算关联"),
    # ===== 拒答类（资料里没有，应拒答）15 条 =====
    ("今天郑州天气怎么样？", "拒答", "应回答没有相关信息（实时天气，知识库无）"),
    ("明天贵州茅台的股票会涨吗？", "拒答", "应拒答（实时行情/预测，知识库无）"),
    ("红烧肉的家常做法是什么？", "拒答", "应拒答（菜谱，知识库无）"),
    ("周杰伦今年多少岁？", "拒答", "应拒答（人物时效信息，知识库无）"),
    ("2026年世界杯足球赛冠军是谁？", "拒答", "应拒答（时效/体育，知识库无）"),
    ("我感冒发烧38度应该吃什么药？", "拒答", "应拒答（医疗建议，知识库无）"),
    ("现在人民币兑美元的汇率是多少？", "拒答", "应拒答（实时汇率，知识库无）"),
    ("帮我写一首关于春天的七言绝句。", "拒答", "应拒答（自由创作，非知识库问答；直接写算未守住边界）"),
    ("从郑州坐高铁到北京要多长时间？", "拒答", "应拒答（出行时刻，知识库无）"),
    ("最新款苹果手机是什么型号、售价多少？", "拒答", "应拒答（时效产品，知识库无）"),
    ("我国个人所得税的税率是多少？", "拒答", "应拒答（法律条文，知识库无）"),
    ("怎么用Python写一个爬取网页的爬虫？", "拒答", "应拒答（知识库无爬虫教程）"),
    ("推荐几家郑州好吃的火锅店。", "拒答", "应拒答（本地生活，知识库无）"),
    ("请解释量子力学的不确定性原理。", "拒答", "应拒答（物理学科，知识库无）"),
    ("帮我算一下23乘以17再加5等于几。", "拒答", "应拒答（纯计算，知识库无；直接算出结果算未守边界）"),
]


def sources_of(question):
    """单独检索一次，取召回片段的来源文件名（去重），供核对引用。"""
    docs = retriever.invoke(question)
    names = []
    for d in docs:
        name = os.path.basename(d.metadata.get("source", "未知"))
        if name not in names:
            names.append(name)
    return "、".join(names)


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "评测明细"

    headers = ["编号", "问题", "类型", "期望要点", "实际回答",
               "检索来源", "答案正确?(是/否)", "引用正确?(是/否/—)"]
    ws.append(headers)

    # 表头样式
    head_fill = PatternFill("solid", fgColor="16406B")
    head_font = Font(name="微软雅黑", bold=True, color="FFFFFF", size=11)
    head_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    head_border = Border(bottom=Side(style="medium", color="16406B"))
    for c in ws[1]:
        c.fill = head_fill
        c.font = head_font
        c.alignment = head_align
        c.border = head_border

    body_font = Font(name="微软雅黑", size=10)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)


    n = len(QUESTIONS)
    for i, (q, typ, expect) in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{n}] {q}")
        try:
            ans = rag_chain.invoke({"question": q, "history": []}).content
            src = sources_of(q)
        except Exception as e:
            ans, src = f"【调用失败】{e}", ""
        ws.append([i, q, typ, expect, ans, src, "", ""])
        r = ws.max_row
        for col in range(1, 9):
            cell = ws.cell(row=r, column=col)
            cell.font = body_font
            cell.alignment = center


    # 列宽（中文按2字符估算，限制在 8~50，超出靠换行）
    widths = {"A": 7, "B": 34, "C": 8, "D": 40, "E": 50, "F": 26, "G": 15, "H": 16}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"  # 冻结表头

    # G/H 列下拉，方便统一填写
    dv_g = DataValidation(type="list", formula1='"是,否"', allow_blank=True)
    dv_h = DataValidation(type="list", formula1='"是,否,—"', allow_blank=True)
    ws.add_data_validation(dv_g)
    ws.add_data_validation(dv_h)
    last = n + 1
    dv_g.add(f"G2:G{last}")
    dv_h.add(f"H2:H{last}")

    # ---- 指标汇总（公式驱动，填完 G/H 自动更新）----
    sm = wb.create_sheet("指标汇总")
    D = "'评测明细'"
    rng_t, rng_g, rng_h = f"{D}!$C$2:$C${last}", f"{D}!$G$2:$G${last}", f"{D}!$H$2:$H${last}"

    sm.merge_cells("A1:C1")
    sm["A1"] = "RAG 问答评测指标汇总"
    sm["A1"].fill = head_fill
    sm["A1"].font = Font(name="微软雅黑", bold=True, color="FFFFFF", size=13)
    sm["A1"].alignment = Alignment(horizontal="center", vertical="center")

    sm.append([])
    sm.append(["指标", "数值", "口径说明"])
    rows = [
        ("应答类问题总数", f'=COUNTIF({rng_t},"应答")', "资料里有答案、应答对的问题数"),
        ("应答答对条数", f'=COUNTIFS({rng_t},"应答",{rng_g},"是")', "G列填“是”的应答题数"),
        ("答案正确率", "=IF(B4=0,\"\",B5/B4)", "应答答对 ÷ 应答总数"),
        ("拒答类问题总数", f'=COUNTIF({rng_t},"拒答")', "资料里没有、应拒答的问题数"),
        ("正确拒答条数", f'=COUNTIFS({rng_t},"拒答",{rng_g},"是")', "G列填“是”的拒答题数（确实拒答）"),
        ("正确拒答率", "=IF(B7=0,\"\",B8/B7)", "正确拒答 ÷ 拒答总数"),
        ("引用命中条数", f'=COUNTIFS({rng_t},"应答",{rng_h},"是")', "H列填“是”的应答题数"),
        ("引用命中率", "=IF(B4=0,\"\",B10/B4)", "引用命中 ÷ 应答总数"),
    ]
    for name, formula, note in rows:
        sm.append([name, formula, note])

    # 汇总表样式
    for c in sm[3]:
        c.fill = PatternFill("solid", fgColor="5B9BD5")
        c.font = Font(name="微软雅黑", bold=True, color="FFFFFF", size=11)
        c.alignment = Alignment(horizontal="center", vertical="center")
    for r in range(4, 12):
        sm.cell(row=r, column=1).font = Font(name="微软雅黑", size=11)
        sm.cell(row=r, column=3).font = Font(name="微软雅黑", size=10)
        sm.cell(row=r, column=3).alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        sm.cell(row=r, column=2).alignment = Alignment(horizontal="center", vertical="center")
    for r in (6, 9, 11):  # 三个比率设百分比
        sm.cell(row=r, column=2).number_format = "0.0%"
    sm.column_dimensions["A"].width = 20
    sm.column_dimensions["B"].width = 14
    sm.column_dimensions["C"].width = 46
    sm["A13"] = ("填写说明：请在「评测明细」G、H 两列用下拉选择。应答类：G 选答案是否正确、"
                 "H 选引用来源是否正确；拒答类：G 选是否正确拒答（是=确实拒答），H 选“—”。"
                 "填写过程中本表数值自动更新；未填完前比率显示 0% 属正常。")
    sm.merge_cells("A13:C18")
    sm["A13"].alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    sm["A13"].font = Font(name="微软雅黑", size=10, color="808080")
    for rr in range(13, 19):
        sm.row_dimensions[rr].height = 22

    out = "RAG评测结果.xlsx"
    wb.save(out)
    print(f"\n完成，共 {n} 条，已生成：{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
