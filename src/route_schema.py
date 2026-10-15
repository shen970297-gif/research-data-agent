from typing import Literal

from pydantic import BaseModel, Field

'''限定大模型返回的路由格式，如果不符合的话就会报错'''
class RouteDecision(BaseModel):#括号里的 BaseModel 不是传入参数，而是表示继承类的用法，因此 RouteDecision 创建出的对象能直接使用 BaseModel 提供的方法
    route: Literal["knowledge", "data", "direct"]#是在给 route 字段规定：它只能是这三个固定字符串之一
    reason: str = Field(min_length=1, max_length=200)
    data_action: Literal["profile", "filter", "none"] = "none"
    file_name: str | None = None