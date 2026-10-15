"""离线测试：使用脚本化模型，不读取 .env，也不访问模型 API。"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, Field

from research_agent import run_research
from research_report import check_evidence
from research_storage import RunWorkspace, SessionStore
from research_tools import ResearchTools
from tool_agent import ToolExecutor, run_tool_agent
from tool_registry import create_research_registry, execute_tool, tool_describe_for_LLM


def reply(content="", calls=None):
    return ChatCompletionMessage(role="assistant", content=content, tool_calls=calls)


def call(name, args, number=1):
    return {"id": f"call_{number}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def plan(**overrides):
    return reply(json.dumps({"steps": ["执行任务并整理结果"], "needs_data": False,
                             "needs_literature": False, "save_report": False, "clarification": None, **overrides}))


class ScriptedModel:
    def __init__(self, script):
        self.script = iter(script)
        self.requests = []

    def think(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        step = next(self.script)
        if isinstance(step, Exception):
            raise step
        return step(messages) if callable(step) else step


def last_result(messages, tool_name):
    names = {}
    for message in messages:
        for tool in message.get("tool_calls", []):
            names[tool["id"]] = tool["function"]["name"]
    for message in reversed(messages):
        if message["role"] == "tool" and names.get(message["tool_call_id"]) == tool_name:
            payload = json.loads(message["content"])
            if payload.get("ok"):
                return payload["result"]
    raise AssertionError(f"missing result {tool_name}")


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.uploads = self.root / "data" / "uploads"
        self.uploads.mkdir(parents=True)
        rows = []
        for group, values in [("young", [10, 12, 14, 16]), ("old", [3, 5, 7, 9])]:
            for i, value in enumerate(values):
                rows.append({"mouse_id": f"{group}{i}", "age_group": group, "time": 30, "signal": value})
                rows.append({"mouse_id": f"{group}{i}", "age_group": group, "time": 60, "signal": value + 1})
        self.frame = pd.DataFrame(rows)
        self.frame.to_csv(self.uploads / "mice.csv", index=False)
        self.workspace = RunWorkspace(self.root, "test")
        self.runtime = ResearchTools(self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def select(self, **kwargs):
        return self.runtime.select_rows("mice.csv", [{"column": "time", "operator": "eq", "value": 30}], **kwargs)

    def group_args(self, dataset_id):
        return {"dataset_id": dataset_id, "group_by": "age_group", "value_column": "signal", "unit_column": "mouse_id", "repeated": "error"}


class DataToolsTests(Fixture):
    def test_statistics_use_all_rows_not_preview(self):
        selected = self.select(max_rows=1)
        self.assertTrue(selected["truncated"])
        self.assertEqual(selected["matched_row_count"], 8)
        result = self.runtime.summarize_groups(**self.group_args(selected["dataset_id"]))
        groups = {r["group"]: r for r in result["groups"]}
        self.assertEqual(groups["young"]["n"], 4)
        self.assertEqual(groups["young"]["mean"], 13)
        self.assertEqual(groups["old"]["mean"], 6)
        self.assertAlmostEqual(groups["young"]["sd"], (20/3)**0.5)

    def test_welch_matches_scipy(self):
        from scipy.stats import ttest_ind
        args = self.group_args(self.select()["dataset_id"])
        result = self.runtime.compare_groups(**args, group_a="young", group_b="old", design="independent", method="welch")
        expected = ttest_ind([10, 12, 14, 16], [3, 5, 7, 9], equal_var=False)
        self.assertAlmostEqual(result["p_value"], float(expected.pvalue))
        self.assertEqual(result["mean_difference_a_minus_b"], 7)
        self.assertAlmostEqual(result["difference_ci95"][0], float(expected.confidence_interval().low))

    def test_repeated_units_require_explicit_policy(self):
        selected = self.runtime.select_rows("mice.csv")
        args = self.group_args(selected["dataset_id"])
        with self.assertRaisesRegex(ValueError, "多条记录"):
            self.runtime.summarize_groups(**args)
        args["repeated"] = "mean"
        result = self.runtime.summarize_groups(**args)
        self.assertEqual(result["quality"]["analysis_rows"], 8)

    def test_cross_group_units_rejected(self):
        frame = self.frame.copy()
        frame["mouse_id"] = frame["mouse_id"].str[-1]
        frame.to_csv(self.uploads / "mice.csv", index=False)
        args = self.group_args(self.select()["dataset_id"])
        with self.assertRaisesRegex(ValueError, "跨两组"):
            self.runtime.compare_groups(**args, group_a="young", group_b="old", design="independent", method="welch")

    def test_missing_values_counted(self):
        self.frame.loc[0, "signal"] = float("nan")
        self.frame.to_csv(self.uploads / "mice.csv", index=False)
        result = self.runtime.summarize_groups(**self.group_args(self.select()["dataset_id"]))
        self.assertEqual(result["quality"]["excluded_missing_rows"], 1)
        self.assertEqual(sum(g["n"] for g in result["groups"]), 7)

    def test_empty_selection_not_fake_zero_statistics(self):
        selected = self.runtime.select_rows("mice.csv", [{"column": "time", "operator": "gt", "value": 1000}])
        self.assertEqual(selected["matched_row_count"], 0)
        with self.assertRaisesRegex(ValueError, "没有可分析"):
            self.runtime.summarize_groups(**self.group_args(selected["dataset_id"]))

    def test_changed_original_rejected(self):
        dataset_id = self.select()["dataset_id"]
        with (self.uploads / "mice.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "已变化"):
            self.workspace.load_dataset(dataset_id)

    def test_tampered_snapshot_rejected(self):
        dataset_id = self.select()["dataset_id"]
        path = self.root / self.workspace.datasets[dataset_id]["path"]
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "被修改"):
            self.workspace.load_dataset(dataset_id)

    def test_unknown_id_and_path_escape(self):
        with self.assertRaises(ValueError):
            self.workspace.load_dataset("ds_invented")
        with self.assertRaises(ValueError):
            self.runtime.select_rows("../../secret.csv")
        with self.assertRaises(ValueError):
            RunWorkspace(self.root, "../../escape")

    def test_export_and_plot(self):
        dataset_id = self.select(max_rows=1)["dataset_id"]
        exported = self.runtime.export_dataset(dataset_id)
        self.assertEqual(len(pd.read_csv(self.root / exported["artifact"]["path"])), 8)
        result = self.runtime.plot_groups(**self.group_args(dataset_id))
        path = self.root / result["artifact"]["path"]
        self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))

    def test_search_metadata_and_missing_evidence(self):
        self.runtime.retriever = lambda **kwargs: [(SimpleNamespace(page_content="Evidence text", metadata={"source": "paper.pdf", "page": 2, "chunk_id": 7}), .9)]
        result = self.runtime.search_literature("question")
        self.assertEqual(result["evidence"][0]["page"], 3)
        self.runtime.retriever = lambda **kwargs: []
        self.assertTrue(self.runtime.search_literature("missing")["insufficient_evidence"])

    def test_registry_no_profile_and_schema_validation(self):
        registry = create_research_registry(self.runtime)
        self.assertNotIn("profile_dataset", registry)
        self.assertEqual(len(tool_describe_for_LLM(registry)), 8)
        with self.assertRaises(ValueError):
            execute_tool("select_rows", json.dumps({"file_name":"mice.csv", "columns":["time","time"]}), registry)


class LoopTests(unittest.TestCase):
    def setUp(self):
        class Args(BaseModel):
            value: int = Field(ge=0)
        self.registry = {"identity": {"args_model": Args, "description": "test", "handler": lambda value: {"value": value}}}

    def test_feedback_repairs_invalid_json(self):
        model = ScriptedModel([reply(calls=[call("identity", {"value": -1})]),
                               reply(calls=[call("identity", {"value": 2}, 2)]), reply("done")])
        self.assertEqual(run_tool_agent("test", model, registry=self.registry), "done")
        response = json.loads(model.requests[1][-1]["content"])
        self.assertFalse(response["ok"])
        self.assertEqual(response["repairs_remaining"], 3)

    def test_exactly_three_repairs_then_stop(self):
        model = ScriptedModel([reply(calls=[call("identity", {"value": -1}, i)]) for i in range(4)])
        with self.assertRaisesRegex(ValueError, "3 次修正"):
            run_tool_agent("test", model, registry=self.registry)
        self.assertEqual(len(model.requests), 4)

    def test_batch_pairing_on_budget_limit(self):
        executor = ToolExecutor(self.registry, max_tool_calls=1)
        messages = []
        executor.execute_batch(reply(calls=[call("identity", {"value": 1}, i) for i in range(3)]), messages)
        self.assertEqual([m["tool_call_id"] for m in messages[1:]], ["call_0", "call_1", "call_2"])
        self.assertEqual(executor.calls, 1)
        self.assertFalse(json.loads(messages[-1]["content"])["executed"])

    def test_last_round_does_not_execute(self):
        invoked = []
        self.registry["identity"]["handler"] = lambda value: invoked.append(value)
        model = ScriptedModel([reply(calls=[call("identity", {"value": 1})])])
        with self.assertRaisesRegex(ValueError, "未执行"):
            run_tool_agent("test", model, max_rounds=1, registry=self.registry)
        self.assertEqual(invoked, [])


class WorkflowTests(Fixture):
    def test_full_research_report(self):
        def computations(messages):
            dataset_id = last_result(messages, "select_rows")["dataset_id"]
            args = self.group_args(dataset_id)
            return reply(calls=[call("summarize_groups", args, 3), call("plot_groups", args, 4),
                call("compare_groups", {**args, "group_a":"young", "group_b":"old", "design":"independent", "method":"welch"}, 5),
                call("export_dataset", {"dataset_id":dataset_id}, 6)])
        def answer(messages):
            evidence = last_result(messages, "search_literature")["evidence"][0]
            return reply("young 均值 13，old 均值 6。文献背景需结合实验条件 [" + evidence["evidence_id"] + "]。")
        model = ScriptedModel([plan(needs_data=True, needs_literature=True, save_report=True),
            reply(calls=[call("select_rows", {"file_name":"mice.csv", "conditions":[{"column":"time","operator":"eq","value":30}], "max_rows":1}, 1),
                         call("search_literature", {"query":"age tracer"}, 2)]), computations, answer,
            reply('{"supported":true,"issues":[]}')])
        retrieve = lambda **kwargs: [(SimpleNamespace(page_content="Synthetic literature fixture: age groups were compared.", metadata={"source":"fixture.pdf", "page":0}), .8)]
        result = run_research("比较30分钟数据并检索文献生成报告，使用独立组Welch检验", model, root=self.root, retriever=retrieve)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_calls"], 5)
        self.assertEqual(result["steps"], ["plan","agent","tools","agent","tools","agent","review","finalize"])
        report = self.root / next(a["path"] for a in result["artifacts"] if a["kind"] == "report")
        self.assertIn("mean_difference_a_minus_b", report.read_text(encoding="utf-8"))
        self.assertTrue((self.root / "logs" / "research_runs.jsonl").exists())
        saved = SessionStore(self.root / "data" / "sessions.sqlite").load("default")
        self.assertEqual(len(saved["datasets"]), 1)

    def test_session_memory_and_isolation(self):
        for session_id, question in [("one", "记住指标 signal"), ("one", "继续上个问题"), ("two", "你好")]:
            model = ScriptedModel([plan(), reply("收到"), reply('{"supported":true,"issues":[]}')])
            result = run_research(question, model, root=self.root, session_id=session_id)
            self.assertEqual(result["status"], "completed")
            context = model.requests[0][1]["content"]
            if question == "继续上个问题":
                self.assertIn("记住指标 signal", context)
            if session_id == "two":
                self.assertNotIn("记住指标 signal", context)

    def test_clarification_stops_without_tools(self):
        model = ScriptedModel([plan(clarification="请提供信号指标列名")])
        result = run_research("帮我分析", model, root=self.root)
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(len(model.requests), 1)

    def test_tool_clarification_skips_later_batch_calls(self):
        model = ScriptedModel([plan(needs_data=True), reply(calls=[
            call("request_clarification", {"question":"如何聚合重复记录？"}, 1),
            call("select_rows", {"file_name":"mice.csv"}, 2)])])
        result = run_research("分析", model, root=self.root)
        self.assertEqual(result["status"], "needs_input")
        manifest = json.loads((Path(result["run_directory"]) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["datasets"], {})

    def test_insufficient_evidence_is_not_completed(self):
        model = ScriptedModel([plan(needs_literature=True)])
        result = run_research("文献依据", model, root=self.root, retriever=lambda **kwargs: [], max_reviews=1)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(len(model.requests), 1)

    def test_timeout_preserves_run_without_exception_secret(self):
        model = ScriptedModel([plan(), TimeoutError("secret-token-should-not-appear")])
        result = run_research("hello", model, root=self.root)
        self.assertEqual(result["status"], "partial")
        self.assertNotIn("secret-token", json.dumps(result))

    def test_unknown_citation_detected(self):
        issues = check_evidence("结果[E000000000000]", {}, self.workspace)
        self.assertTrue(issues)

    def test_plan_invalid_json_stops(self):
        model = ScriptedModel([reply("not-json") for _ in range(4)])
        result = run_research("hello", model, root=self.root)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["model_calls"], 4)

    def test_review_failure_not_marked_complete(self):
        model = ScriptedModel([plan(), reply("answer"), reply("invalid")])
        result = run_research("hello", model, root=self.root, max_reviews=1)
        self.assertEqual(result["status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
