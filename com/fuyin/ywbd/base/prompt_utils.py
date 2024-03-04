prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。

已知信息:
__INFO__

用户问：
__QUERY__

请用中文回答用户问题。
"""


def build_prompt(template=prompt_template, **kwargs):
    """将 Prompt 模板赋值"""
    prompt = template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt
