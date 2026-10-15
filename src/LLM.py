'''
这部分写了本地部署密钥+掉用大模型
后面每次使用的时候不用重复写密钥/请求/流式响应
'''
import os
import math
from pathlib import Path
from typing import Any
from openai.types.chat import ChatCompletionMessage
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用兼容 OpenAI Chat Completions 格式的服务，当前使用非流式响应。
    """

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = float(timeout if timeout is not None else os.getenv("LLM_TIMEOUT", 120))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("LLM_TIMEOUT 必须是正数")
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(
            api_key=apiKey,
            base_url=baseUrl,
            timeout=timeout,
            max_retries=0,
        )

        print(
            f"LLM 配置：model={self.model}，"
            f"timeout={self.client.timeout}，"
            f"max_retries={self.client.max_retries}"
        )

    def think(self, messages: list[dict[str, Any]], temperature: float = 0, tools: list | None = None,json_mode: bool = False) -> ChatCompletionMessage:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"正在调用 {self.model} 模型...")
        try:
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,

            }

            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "auto"
            if json_mode:
                request_kwargs["response_format"] = {
                    "type": "json_object"
                }
            response = self.client.chat.completions.create(**request_kwargs)


                #处理流式响应,如果stream是True的话，需要这段代码
            # print("✅ 大语言模型响应成功:")
            # collected_content = []
            # for chunk in response:
            #     if not chunk.choices:
            #         continue
            #     content = chunk.choices[0].delta.content or ""
            #     print(content, end="", flush=True)
            #     collected_content.append(content)
            # print()  # 在流式输出结束后换行
            return response.choices[0].message
        #大模型返回的回答是个列表，里面可能有多条回答，一般是1条，所以是response.choices[0]，回答中有message和其他模型，消耗的token等信息，有用的回答在message中，但message中又有role（角色）和content(需要的回答)
        except Exception as e:
            # 异常正文可能带请求内容或凭据；控制台只输出类型。
            print(f"调用模型失败：{type(e).__name__}")
            raise


# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)

