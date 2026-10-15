"""完整数据计算与文献检索。所有文件访问都限制在项目目录。"""
import hashlib
import io
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from research_schema import Condition, GroupRequest


class ResearchTools:
    def __init__(self, workspace, retriever=None):
        self.workspace = workspace
        self.upload_dir = workspace.root / "data" / "uploads"
        self.retriever = retriever
        self.clarification = None

    def request_clarification(self, question):
        self.clarification = question
        return {"status": "needs_input", "question": question}

    def catalog(self):
        """内部字段目录，供模型理解可用文件；不是数据预览工具。"""
        files = []
        for path in sorted(self.upload_dir.glob("*.csv"))[:30]:
            if not path.resolve().is_relative_to(self.upload_dir.resolve()):
                continue
            try:
                columns = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
                files.append({"file_name": path.name, "columns": columns})
            except (ValueError, OSError):
                files.append({"file_name": path.name, "error": "无法读取 CSV 表头"})
        return files

    def select_rows(self, file_name, conditions=None, columns=None, max_rows=5):
        path = (self.upload_dir / file_name).resolve()
        if not path.is_relative_to(self.upload_dir.resolve()) or path.suffix.lower() != ".csv":
            raise ValueError("只能读取 data/uploads 中的 CSV")
        if not path.is_file():
            raise ValueError("CSV 文件不存在，请检查文件名")
        content = path.read_bytes()
        frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", low_memory=False)
        conditions = [Condition.model_validate(c) for c in (conditions or [])]
        requested = [c.column for c in conditions] + (columns or [])
        if any(c not in frame.columns for c in requested):
            raise ValueError("指定字段不存在。可用字段：" + ", ".join(frame.columns))
        mask = pd.Series(True, index=frame.index)
        excluded_missing = pd.Series(False, index=frame.index)
        for condition in conditions:
            series = frame[condition.column]
            value = condition.value
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                series = pd.to_numeric(series, errors="raise")
                if np.isinf(series.dropna().to_numpy(dtype=float)).any():
                    raise ValueError("筛选列包含无穷大")
            elif condition.operator not in {"eq", "ne"}:
                raise ValueError("大小比较需要数值条件；文本只支持 eq/ne")
            if isinstance(value, str):
                series = series.astype("string")
            excluded_missing |= series.isna()
            comparison = getattr(series, condition.operator)(value)
            mask &= series.notna() & comparison.fillna(False)
        selected = frame.loc[mask, columns] if columns is not None else frame.loc[mask]
        numeric = selected.select_dtypes(include="number")
        if not numeric.empty and np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("保留的数值列包含无穷大，请先处理数据，不能将无穷大静默转成缺失值")
        provenance = {"file_name": file_name, "source_sha256": hashlib.sha256(content).hexdigest(),
                      "conditions": [c.model_dump() for c in conditions], "columns": columns}
        dataset_id = self.workspace.save_dataset(selected, provenance)
        rows = __import__("json").loads(selected.head(max_rows).to_json(orient="records", force_ascii=False, double_precision=15))
        return {"dataset_id": dataset_id, "file_name": file_name, "columns": selected.columns.tolist(),
                "matched_row_count": len(selected), "returned_row_count": len(rows),
                "truncated": len(selected) > len(rows), "missing_filter_row_count": int(excluded_missing.sum()),
                "rows": rows, "provenance": provenance}

    def filter_dataset(self, file_name, columns, time_min, time_max, max_rows=20):
        """兼容旧筛选参数，并新增可用于完整计算的 dataset_id。"""
        return self.select_rows(file_name, [
            {"column": "time", "operator": "ge", "value": time_min},
            {"column": "time", "operator": "le", "value": time_max},
        ], columns, max_rows)

    def _group_data(self, request):
        frame = self.workspace.load_dataset(request.dataset_id)
        columns = [request.group_by, request.value_column] + ([request.unit_column] if request.unit_column else [])
        if len(columns) != len(set(columns)):
            raise ValueError("分组、指标和实验单位必须使用不同的列")
        if any(column not in frame.columns for column in columns):
            raise ValueError("数据集中缺少分组/指标/单位列，请重新筛选并保留这些列")
        frame = frame[columns].copy()
        frame[request.value_column] = pd.to_numeric(frame[request.value_column], errors="raise")
        if np.isinf(frame[request.value_column].dropna().to_numpy(dtype=float)).any():
            raise ValueError("指标包含无穷大")
        missing = frame.isna().any(axis=1)
        discarded = int(missing.sum())
        frame = frame.loc[~missing].copy()
        if frame.empty:
            raise ValueError("筛选后没有可分析的完整记录")
        frame[request.group_by] = frame[request.group_by].astype(str)
        if frame[request.group_by].nunique() > 50:
            raise ValueError("分组超过 50 组，请缩小分析范围")
        if request.unit_column:
            keys = [request.group_by, request.unit_column]
            if frame.duplicated(keys).any():
                if request.repeated == "error":
                    raise ValueError("同一单位在同组有多条记录。请确认统计单位或明确同意 repeated='mean'")
                frame = frame.groupby(keys, as_index=False)[request.value_column].mean()
        return frame, {"excluded_missing_rows": discarded, "input_rows": len(missing),
                       "analysis_rows": len(frame), "unit_column": request.unit_column,
                       "repeated_policy": request.repeated,
                       "n_meaning": "实验单位数" if request.unit_column else "记录数，不能直接视为独立样本数"}

    def summarize_groups(self, **kwargs):
        request = GroupRequest.model_validate(kwargs)
        frame, quality = self._group_data(request)
        groups = []
        for label, group in frame.groupby(request.group_by, sort=True):
            values = group[request.value_column]
            groups.append({"group": str(label), "n": len(values), "mean": float(values.mean()),
                           "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                           "median": float(values.median()), "min": float(values.min()), "max": float(values.max())})
        return {"dataset_id": request.dataset_id, "group_by": request.group_by,
                "value_column": request.value_column, "quality": quality, "groups": groups}

    def compare_groups(self, **kwargs):
        from scipy.stats import ttest_ind
        from research_schema import CompareRequest
        request = CompareRequest.model_validate(kwargs)
        frame, quality = self._group_data(request)
        frame = frame[frame[request.group_by].isin([request.group_a, request.group_b])]
        if frame.groupby(request.unit_column)[request.group_by].nunique().gt(1).any():
            raise ValueError("同一实验单位跨两组出现，不能使用独立组 Welch 检验")
        a = frame.loc[frame[request.group_by] == request.group_a, request.value_column].to_numpy(dtype=float)
        b = frame.loc[frame[request.group_by] == request.group_b, request.value_column].to_numpy(dtype=float)
        if min(len(a), len(b)) < 2:
            raise ValueError("每组至少需要两个有效独立单位；请检查组标签与样本量")
        if np.var(a, ddof=1) == 0 and np.var(b, ddof=1) == 0:
            raise ValueError("两组都没有组内变异，无法报告可靠的 Welch 检验结果")
        test = ttest_ind(a, b, equal_var=False, alternative="two-sided")
        interval = test.confidence_interval(confidence_level=0.95)
        numbers = [test.statistic, test.pvalue, test.df, interval.low, interval.high]
        if not np.isfinite(numbers).all():
            raise ValueError("检验产生非有限数值，请检查数据")
        return {"dataset_id": request.dataset_id, "method": "Welch two-sided t-test",
                "group_a": request.group_a, "group_b": request.group_b, "n_a": len(a), "n_b": len(b),
                "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_difference_a_minus_b": float(a.mean()-b.mean()),
                "difference_ci95": [float(interval.low), float(interval.high)], "statistic": float(test.statistic),
                "df": float(test.df), "p_value": float(test.pvalue), "quality": quality,
                "limitations": "依赖独立单位等设计假设；未校正多重比较；不自动作因果解释。"}

    def plot_groups(self, **kwargs):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        request = GroupRequest.model_validate(kwargs)
        frame, quality = self._group_data(request)
        labels = sorted(frame[request.group_by].unique())
        with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False}):
            figure, axis = plt.subplots(figsize=(max(6, min(14, len(labels))), 4.5))
            rng = np.random.default_rng(0)
            for index, label in enumerate(labels):
                values = frame.loc[frame[request.group_by] == label, request.value_column].to_numpy()
                axis.scatter(index + rng.uniform(-0.12, 0.12, len(values)), values, alpha=0.65, s=24)
                axis.plot([index-.22, index+.22], [values.mean()]*2, color="black", linewidth=2)
            axis.set_xticks(range(len(labels)), labels, rotation=20)
            axis.set_xlabel(request.group_by)
            axis.set_ylabel(request.value_column)
            axis.set_title("Points: units/records; black line: mean")
            figure.tight_layout()
            path = self.workspace.directory / ("groups_" + uuid4().hex[:8] + ".png")
            try:
                figure.savefig(path, dpi=160)
            finally:
                plt.close(figure)
        return {"dataset_id": request.dataset_id, "quality": quality,
                "artifact": self.workspace.register_artifact(path, "figure")}

    def export_dataset(self, dataset_id):
        frame = self.workspace.load_dataset(dataset_id)
        path = self.workspace.directory / (dataset_id + "_" + uuid4().hex[:8] + ".csv")
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return {"dataset_id": dataset_id, "row_count": len(frame), "artifact": self.workspace.register_artifact(path, "csv")}

    def search_literature(self, query, k=5, score_threshold=0.5):
        if self.retriever is None:
            from retriever import retrieve_docs
            retrieve = retrieve_docs
        else:
            retrieve = self.retriever
        documents = retrieve(query=query, k=k, score_threshold=score_threshold)
        evidence = []
        seen = set()
        for document, score in documents:
            if not document.page_content.strip():
                continue
            metadata = document.metadata
            key = str(metadata.get("source", "unknown")) + str(metadata.get("page")) + document.page_content
            evidence_id = "E" + hashlib.sha256(key.encode()).hexdigest()[:12]
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            page = metadata.get("page")
            item = {"evidence_id": evidence_id, "source": str(metadata.get("source", "未知来源")),
                    "page": page + 1 if isinstance(page, int) else None, "chunk_id": metadata.get("chunk_id"),
                    "score": float(score), "excerpt": document.page_content[:2400]}
            self.workspace.evidence[evidence_id] = item
            evidence.append(item)
        return {"query": query, "evidence": evidence, "insufficient_evidence": not evidence,
                "note": "相似度不代表结论成立；请检查实验条件和原文。无命中时可改写查询，不能编造出处。"}
