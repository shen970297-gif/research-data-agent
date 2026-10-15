"""保存会话和本次任务产物；模型只获得 ID，不能指定任意磁盘路径。"""
import hashlib
import io
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pandas as pd


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


class SessionStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def load(self, session_id: str) -> dict:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            row = connection.execute("SELECT payload FROM sessions WHERE id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else {"turns": [], "datasets": {}, "evidence": {}}

    def save(self, session_id: str, payload: dict):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("INSERT INTO sessions VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (session_id, json_text(payload)))


class RunWorkspace:
    def __init__(self, root: Path, session_id: str, datasets=None, evidence=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("session_id 只能包含 1-64 位字母、数字、下划线或连字符")
        self.root = root.resolve()
        self.session_id = session_id
        self.run_id = uuid4().hex
        self.directory = self.root / "reports" / "runs" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.datasets = dict(datasets or {})
        self.evidence = dict(evidence or {})
        self.results = []
        self.artifacts = []

    def save_json(self, name: str, value):
        (self.directory / name).write_text(json_text(value), encoding="utf-8")

    def save_dataset(self, frame, provenance):
        dataset_id = "ds_" + uuid4().hex[:16]
        path = self.directory / (dataset_id + ".json")
        content = frame.to_json(orient="table", index=False, force_ascii=False, double_precision=15)
        path.write_text(content, encoding="utf-8")
        self.datasets[dataset_id] = {
            "path": str(path.relative_to(self.root)), "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "rows": len(frame), "columns": frame.columns.tolist(), "provenance": provenance,
        }
        return dataset_id

    def load_dataset(self, dataset_id):
        if dataset_id not in self.datasets:
            raise ValueError("数据集 ID 不属于当前会话；请先筛选数据并使用真实返回的 ID")
        metadata = self.datasets[dataset_id]
        provenance = metadata.get("provenance", {})
        source = (self.root / "data" / "uploads" / provenance.get("file_name", "")).resolve()
        if not source.is_relative_to((self.root / "data" / "uploads").resolve()):
            raise ValueError("原始数据路径越界")
        if source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() != provenance.get("source_sha256"):
            raise ValueError("原始 CSV 已变化，请重新筛选，避免混用旧结果")
        path = (self.root / metadata["path"]).resolve()
        if not path.is_relative_to(self.root / "reports" / "runs"):
            raise ValueError("数据集路径越界")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
            raise ValueError("保存的数据集已被修改，请重新筛选")
        return pd.read_json(io.StringIO(content.decode("utf-8")), orient="table")

    def record_result(self, tool_name, arguments, result):
        self.results.append({"tool": tool_name, "arguments": arguments, "result": result})
        self.save_json("tool_results.json", self.results)

    def register_artifact(self, path, kind):
        item = {"kind": kind, "path": str(path.relative_to(self.root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.artifacts.append(item)
        return item

    def manifest(self):
        return {"run_id": self.run_id, "session_id": self.session_id,
                "datasets": self.datasets, "evidence": self.evidence, "artifacts": self.artifacts}
