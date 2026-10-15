"""RAG 分支的离线契约测试：检索器、模型均注入，不调用外部 API。"""
import json
from types import SimpleNamespace

from test_research_core import Fixture, ScriptedModel, plan, reply, call
from research_agent import run_research
from research_storage import SessionStore


def retrieved(**kwargs):
    return [(SimpleNamespace(page_content="Synthetic evidence: the study compares tracer signals at 30 minutes.",
                             metadata={"source": "example.pdf", "page": 3, "chunk_id": 2}), 0.85)]


def grounded_reply(messages):
    context = json.loads(messages[-1]["content"])
    evidence = context["evidence"][0]
    return reply("该研究比较了30分钟的示踪剂信号 [" + evidence["evidence_id"] + "]。")


class RAGTests(Fixture):
    def test_rag_has_own_nodes_and_injects_evidence(self):
        model = ScriptedModel([plan(needs_literature=True), grounded_reply, reply('{"supported":true,"issues":[]}')])
        result = run_research("根据本地文献解释测量时间", model, root=self.root, retriever=retrieved)
        self.assertEqual(result["route"], "rag")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"], ["plan", "rag_retrieve", "rag_generate", "review", "finalize"])
        self.assertEqual(result["model_calls"], 3)
        self.assertIn("Synthetic evidence", model.requests[1][-1]["content"])
        self.assertEqual(result["sources"][0]["page"], 4)
        self.assertEqual(result["sources"][0]["source"], "example.pdf")

    def test_resolved_followup_query_reaches_retriever(self):
        queries = []
        def retrieve(**kwargs):
            queries.append(kwargs["query"])
            return retrieved()
        model = ScriptedModel([plan(needs_literature=True, retrieval_query="论文中 tracer signal 采集时间"),
                               grounded_reply, reply('{"supported":true,"issues":[]}')])
        run_research("它什么时候测量？", model, root=self.root, retriever=retrieve)
        self.assertEqual(queries, ["论文中 tracer signal 采集时间"])

    def test_bad_citation_revised_without_retrieving_again(self):
        queries = []
        def retrieve(**kwargs):
            queries.append(kwargs)
            return retrieved()
        model = ScriptedModel([plan(needs_literature=True), reply("错误引用 [E000000000000]"),
                               grounded_reply, reply('{"supported":true,"issues":[]}')])
        result = run_research("问文献", model, root=self.root, retriever=retrieve)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(queries), 1)
        self.assertEqual(result["steps"].count("rag_generate"), 2)
        self.assertNotIn("agent", result["steps"])
        self.assertIn("revision_issues", json.loads(model.requests[2][-1]["content"]))

    def test_repeated_bad_citation_is_not_marked_complete(self):
        model = ScriptedModel([plan(needs_literature=True), reply("错误 [E000000000000]"), reply("仍错 [E000000000000]")])
        result = run_research("问文献", model, root=self.root, retriever=retrieved)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["sources"], [])

    def test_empty_results_do_not_reuse_old_session_evidence(self):
        store = SessionStore(self.root / "data" / "sessions.sqlite")
        old = {"evidence_id":"E000000000000", "source":"old.pdf", "page":1, "excerpt":"Old content"}
        store.save("default", {"turns": [], "datasets": {}, "evidence": {old["evidence_id"]:old}})
        model = ScriptedModel([plan(needs_literature=True)])
        result = run_research("新文献问题", model, root=self.root, retriever=lambda **kwargs: [])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("rag_generate", result["steps"])
        self.assertEqual(result["sources"], [])

    def test_old_citation_cannot_pass_current_evidence_check(self):
        store = SessionStore(self.root / "data" / "sessions.sqlite")
        old = {"evidence_id":"E000000000000", "source":"old.pdf", "page":1, "excerpt":"Old content"}
        store.save("default", {"turns": [], "datasets": {}, "evidence": {old["evidence_id"]:old}})
        model = ScriptedModel([plan(needs_literature=True), reply("结论 [E000000000000]")])
        result = run_research("新文献问题", model, root=self.root, retriever=retrieved, max_reviews=1)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["sources"], [])

    def test_retrieval_failure_preserved_without_secret(self):
        def fail(**kwargs):
            raise RuntimeError("private-provider-error")
        model = ScriptedModel([plan(needs_literature=True)])
        result = run_research("问文献", model, root=self.root, retriever=fail)
        self.assertEqual(result["status"], "partial")
        self.assertNotIn("private-provider-error", json.dumps(result))
        self.assertNotIn("rag_generate", result["steps"])

    def test_generation_budget_exhausted_leaves_unreviewed_status(self):
        model = ScriptedModel([plan(needs_literature=True), grounded_reply])
        result = run_research("问文献", model, root=self.root, retriever=retrieved, max_rounds=2)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["model_calls"], 2)

    def test_empty_or_tool_call_generation_is_rejected(self):
        for message in [reply(""), reply(calls=[call("select_rows", {"file_name":"mice.csv"})])]:
            model = ScriptedModel([plan(needs_literature=True), message])
            result = run_research("问文献", model, root=self.root, retriever=retrieved)
            self.assertEqual(result["status"], "partial")
            self.assertNotIn("tools", result["steps"])

    def test_blank_and_duplicate_chunks_are_filtered(self):
        blank = (SimpleNamespace(page_content="  ", metadata={"source":"blank.pdf"}), .9)
        doc = retrieved()[0]
        self.runtime.retriever = lambda **kwargs: [blank, doc, doc]
        result = self.runtime.search_literature("question")
        self.assertEqual(len(result["evidence"]), 1)

    def test_rag_report_contains_sources(self):
        model = ScriptedModel([plan(needs_literature=True, save_report=True), grounded_reply,
                               reply('{"supported":true,"issues":[]}')])
        result = run_research("查文献并保存报告", model, root=self.root, retriever=retrieved)
        artifact = next(a for a in result["artifacts"] if a["kind"] == "report")
        text = (self.root / artifact["path"]).read_text(encoding="utf-8")
        self.assertIn("example.pdf", text)
        self.assertIn("Synthetic evidence", text)
