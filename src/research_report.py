"""引用检查与报告保存。原始工具输出独立附录，便于复核模型文字。"""
import re

from research_storage import json_text


def check_evidence(answer, plan, workspace, evidence=None):
    issues = []
    allowed_evidence = workspace.evidence if evidence is None else {item["evidence_id"]: item for item in evidence}
    citations = set(re.findall(r"\[(E[a-zA-Z0-9_-]+)\]", answer))
    unknown = citations - allowed_evidence.keys()
    if unknown:
        issues.append("答案引用了不存在的证据 ID：" + ", ".join(sorted(unknown)))
    if plan.get("needs_literature") and not citations:
        issues.append("任务需要文献依据，但答案未提供可追溯的证据 ID")
    if plan.get("needs_literature") and not allowed_evidence:
        issues.append("没有可用文献证据，应报告资料不足")
    data_tools = {"filter_dataset", "select_rows", "summarize_groups", "compare_groups", "plot_groups", "export_dataset"}
    if plan.get("needs_data") and not any(r["tool"] in data_tools for r in workspace.results):
        issues.append("本次任务没有成功执行数据工具，不能声称已完成分析")
    return issues


def write_report(workspace, question, answer, status, issues, plan):
    sections = ["# 科研分析报告", "## 问题", question, "## 任务状态", status,
                "模型审查是辅助检查，不代表经过科研人员复核。", "## 任务计划",
                "\n".join(f"{i}. {step}" for i, step in enumerate(plan.get("steps", []), 1)),
                "## 回答", answer]
    if issues:
        sections += ["## 待解决问题", "\n".join("- " + issue for issue in issues)]
    sections += ["## 本次计算记录（由程序写入）"]
    for record in workspace.results:
        if record["tool"] in {"search_literature", "request_clarification"}:
            continue
        value = {**record, "result": {k: v for k, v in record["result"].items() if k != "rows"}}
        sections += ["### " + record["tool"], "```json\n" + json_text(value) + "\n```"]
    sections += ["## 文献证据"]
    cited_ids = set(re.findall(r"\[(E[a-zA-Z0-9_-]+)\]", answer))
    for evidence in workspace.evidence.values():
        if evidence["evidence_id"] not in cited_ids:
            continue
        sections += [f"### [{evidence['evidence_id']}] {evidence['source']} / 页码：{evidence['page']}",
                     evidence["excerpt"]]
    sections += ["## 文件产物"]
    for item in workspace.artifacts:
        name = item["path"].replace("\\", "/").split("/")[-1]
        sections.append(f"![分析图]({name})" if item["kind"] == "figure" else f"[下载 {name}]({name})")
    path = workspace.directory / "report.md"
    path.write_text("\n\n".join(sections), encoding="utf-8")
    return workspace.register_artifact(path, "report")
