import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 自动读取 .env 里的 Key

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user",
               "content": "你好，请用一句话介绍你自己"}],
)
print(resp.choices[0].message.content)
