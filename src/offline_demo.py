"""可重复的离线演示：脚本化模型 + 合成小鼠数据 + 明确标注的模拟文献。"""
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
from openai.types.chat import ChatCompletionMessage

from research_agent import PROJECT_ROOT, run_research


def response(content="", calls=None):
    return ChatCompletionMessage(role="assistant", content=content, tool_calls=calls)


def tool(name, arguments, number):
    return {"id": "demo_" + str(number), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def result_for(messages, name):
    names = {tool["id"]: tool["function"]["name"] for message in messages for tool in message.get("tool_calls", [])}
    for message in reversed(messages):
        if message["role"] == "tool" and names.get(message["tool_call_id"]) == name:
            return json.loads(message["content"])["result"]
    raise ValueError("演示中未找到对应工具结果")


class DemoModel:
    """只输出预先规定的调用步骤，用于验证程序，不用于真实科研问答。"""
    def __init__(self):
        self.index = 0

    def think(self, messages, **kwargs):
        stage = self.index
        self.index += 1
        if stage == 0:
            return response(json.dumps({"steps": ["选择30分钟记录", "分组统计、比较与绘图", "整理模拟证据并保存报告"],
                "needs_data": True, "needs_literature": True, "save_report": True, "clarification": None}))
        if stage == 1:
            return response(calls=[tool("select_rows", {"file_name":"synthetic_mice.csv",
                "conditions":[{"column":"time", "operator":"eq", "value":30}], "max_rows":2}, 1),
                tool("search_literature", {"query":"synthetic demonstration"}, 2)])
        if stage == 2:
            dataset_id = result_for(messages, "select_rows")["dataset_id"]
            args = {"dataset_id": dataset_id, "group_by":"age_group", "value_column":"signal", "unit_column":"mouse_id"}
            return response(calls=[tool("summarize_groups", args, 3), tool("plot_groups", args, 4),
                tool("compare_groups", {**args, "group_a":"young", "group_b":"old", "design":"independent", "method":"welch"}, 5),
                tool("export_dataset", {"dataset_id":dataset_id}, 6)])
        if stage == 3:
            statistics = result_for(messages, "compare_groups")
            evidence = result_for(messages, "search_literature")["evidence"][0]
            return response(
                "**这是合成数据与模拟文献的离线演示，不是真实实验结论。**\n\n"
                f"分析全部 8 条记录，每组 4 个合成样本。young 均值为 {statistics['mean_a']:.2f}，"
                f"old 均值为 {statistics['mean_b']:.2f}；差值为 {statistics['mean_difference_a_minus_b']:.2f}。"
                f"双侧 Welch 检验 p={statistics['p_value']:.6g}，数值由 SciPy 实际计算。\n\n"
                f"模拟文献仅用于展示来源关联 [{evidence['evidence_id']}]。不能据此推断任何真实生物学结论。")
        return response('{"supported": true, "issues": []}')


def run_offline_demo():
    root = PROJECT_ROOT / "reports" / "offline_demo" / uuid4().hex[:12]
    uploads = root / "data" / "uploads"
    uploads.mkdir(parents=True)
    rows = [{"mouse_id": group+str(i), "age_group":group, "time":30, "signal":value}
            for group, values in [("young", [10,12,14,16]), ("old", [3,5,7,9])]
            for i, value in enumerate(values)]
    pd.DataFrame(rows).to_csv(uploads / "synthetic_mice.csv", index=False)
    def retrieve(**kwargs):
        return [(SimpleNamespace(page_content="模拟文献：此片段只用于验证引用链路，没有真实研究来源，不能支持生物学结论。",
                                 metadata={"source":"模拟文献-非真实论文", "page":0}), 1.0)]
    result = run_research("【离线演示】分析30分钟的独立组合成数据，执行Welch检验、绘图、导出并生成带模拟引用的报告。",
                          DemoModel(), root=root, session_id="offline-demo", retriever=retrieve,
                          progress=lambda name: print("执行节点：", name))
    print(result["answer"])
    print("\n状态：", result["status"], "（仅验证代码流程，模型输出为脚本模拟）")
    for artifact in result["artifacts"]:
        print(artifact["kind"], str(root / artifact["path"]))
    return result


if __name__ == "__main__":
    run_offline_demo()
