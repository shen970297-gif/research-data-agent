import json

from data_schema import FilterRequest
from tools import filter_dataset

#工具管理的代码，保存函数名字描述和代码
TOOL_REGISTRY = {
    "filter_dataset": {
        "description": (
            "按 time 数值区间筛选 CSV，包含上下边界，"
            "返回指定字段、匹配记录数和有限条数的记录。"
            "必须明确文件名、字段和时间上下限，不换算时间单位。"
            "时间下限不能大于上限，返回字段不能重复。"
            "此工具不计算均值等统计量。"
        ),
        "args_model": FilterRequest,#参数检验（大模型返回的）的函数代码本体
        "handler": filter_dataset,#工具函数代码本体
    },
}
#发给LLM的工具描述，不需要代码本身，只需要发工具名称、描述、需要返回的参数以及参数格式。
def tool_describe_for_LLM(registry: dict | None = None) -> list[dict]:
    """将工具注册表转换为发给模型的工具说明。"""

    tool_schemas = []

    for tool_name, tool_info in (TOOL_REGISTRY if registry is None else registry).items():
        args_model = tool_info["args_model"]

        tool_schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": args_model.model_json_schema(),#model_json_schema() 不仅会提取类型、默认值和数值限制，也会提取 Field() 中的 description
            },
        }

        tool_schemas.append(tool_schema)

    return tool_schemas

def execute_tool(tool_name: str, raw_json: str, registry: dict | None = None) -> dict:
    """根据工具名称，校验模型返回，参数并执行已注册的工具。"""

    registry = TOOL_REGISTRY if registry is None else registry
    if tool_name not in registry:
        raise ValueError(f"未注册的工具：{tool_name}")

    tool_info = registry[tool_name]

    args_model = tool_info["args_model"]
    request = args_model.model_validate_json(raw_json)
    arguments = request.model_dump()

    handler = tool_info["handler"]
    result = handler(**arguments)

    return result


def create_research_registry(runtime) -> dict:
    """给每次研究绑定自己的文件、数据集和证据，避免跨会话混用。"""
    from research_schema import SelectRequest, GroupRequest, CompareRequest, SearchRequest, ExportRequest, ClarificationRequest
    entries = [
        ("request_clarification", ClarificationRequest, "必要信息缺失或统计设计未确认时，向用户提问并暂停本次任务。"),
        ("filter_dataset", FilterRequest, "兼容原有 time 区间筛选，返回完整数据集 ID 和有限展示记录。"),
        ("select_rows", SelectRequest, "从原始 CSV 按 AND 条件筛选，返回完整 dataset_id。修改条件时重新读取原始文件。"),
        ("summarize_groups", GroupRequest, "对完整数据集分组计算记录/单位数、均值、样本标准差和中位数，不进行显著性检验。"),
        ("compare_groups", CompareRequest, "仅在用户明确确认独立组设计和 Welch 方法时，比较两个独立组。不能猜测统计设计。"),
        ("plot_groups", GroupRequest, "绘制完整数据的分组散点和均值线；每个点表示记录或聚合后的单位，返回图片产物。"),
        ("export_dataset", ExportRequest, "导出完整筛选数据为 CSV，不受展示行数限制。"),
        ("search_literature", SearchRequest, "检索本地文献，返回原文片段、证据 ID 和页码；无命中可改写查询后再搜。"),
    ]
    return {name: {"description": description, "args_model": schema, "handler": getattr(runtime, name)}
            for name, schema, description in entries}

if __name__ == "__main__":
    for tool_name, tool_info in TOOL_REGISTRY.items():
        args_model = tool_info["args_model"]
        schema = args_model.model_json_schema()

        print("工具名称：", tool_name)
        print("执行函数：", tool_info["handler"].__name__)
        print("参数字段：", list(schema["properties"]))
        print()
