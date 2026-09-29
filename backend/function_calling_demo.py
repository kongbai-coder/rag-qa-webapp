# -*- coding: utf-8 -*-
"""
Function Calling（工具调用 / Agent 雏形）
给 qwen-plus 挂一个"计算器"工具，观察它遇到算术题时自主决定：
  1) 先返回工具调用请求（工具名 + 参数），而不是自己心算；
  2) 程序真正执行工具；
  3) 把工具结果回传，模型再组织最终自然语言答案。
非算术题则不调用工具、直接回答。
运行（backend 目录、已激活虚拟环境）：python function_calling_demo.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()


# ---- 1. 用 @tool 定义一个计算器工具（docstring 会发给模型，告诉它工具用途）----
@tool
def calculator(expression: str) -> str:
    """数学计算工具。输入一个只包含数字、小数点和 + - * / ( ) 的合法算式，
    例如 23*17+5 或 (128+256)*2，返回计算结果字符串。"""
    allowed = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
    }
    # 教学用受限沙箱：禁用所有内建函数，命名空间里只暴露 + - * / 四个运算。
    # 注意：仅限本地教学演示；生产环境请换用成熟的安全表达式解析库，勿直接用 eval。
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"计算失败：{e}"


# ---- 2. 绑定工具：模型从此"看得到"计算器，并可自主决定是否调用 ----
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
llm_with_tools = llm.bind_tools([calculator])
tool_map = {"calculator": calculator}


def ask(question: str):
    print("=" * 64)
    print("用户问题：", question)

    # 第 1 轮：模型决定要不要调用工具
    ai_msg = llm_with_tools.invoke([HumanMessage(content=question)])

    if not ai_msg.tool_calls:
        print("→ 模型判断【不需要工具】，直接回答：")
        print(ai_msg.content)
        return

    # 第 2 步：程序按模型给的工具名和参数，真正执行工具
    tool_messages = []
    for tc in ai_msg.tool_calls:
        name, args, call_id = tc["name"], tc["args"], tc["id"]
        print(f"→ 模型决定【调用工具】{name}，参数：{args}")
        result = tool_map[name].invoke(args)
        print(f"→ 程序执行工具，返回结果：{result}")
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    # 第 3 步：把工具结果回传，模型据此组织最终答案
    final = llm_with_tools.invoke(
        [HumanMessage(content=question), ai_msg, *tool_messages]
    )
    print("→ 模型结合工具结果，最终回答：")
    print(final.content)


if __name__ == "__main__":
    ask("帮我算一下23乘以17再加5等于几")
    ask("(128+256)*2 等于多少")
    ask("什么是RAG（检索增强生成）？")
