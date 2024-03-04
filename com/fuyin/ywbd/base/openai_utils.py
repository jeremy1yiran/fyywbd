import openai
import os
# 加载环境变量
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
# 加载环境变量


_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY
api_key = os.getenv('OPENAI_API_KEY')
_httpx = os.getenv('HTTPS_PROXY')
os.environ["OPENAI_BASE_URL"] = "https://api.fe8.cn/v1"
# openai.api_key = 'sk-c34ZNPkWBBeO2prhzvetPoinqcDoxyM27h8GeQIhyVolFPQW'
os.environ["HTTPS_PROXY"] = _httpx
os.environ["openai_api_key"] = api_key
client = OpenAI()



def get_completion(prompt, context, model="gpt-3.5-turbo"):
    """封装 openai 接口"""
    messages = context + [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )

    return response.choices[0].message.content

# 封装 embedding 接口 这段代码适用于0.28.0版本
def demo_get_embedding(text, model="text-embedding-ada-002"):
    """封装 OpenAI 的 Embedding 模型接口"""
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def get_embedding(texts, model="text-embedding-ada-002", dimensions=None):
    '''封装 OpenAI 的 Embedding 模型接口'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    # return [x.embedding for x in data]
    return data[0].embedding
