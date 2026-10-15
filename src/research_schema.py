"""新工作流与工具的输入合同。这里不读取文件、不调用模型。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ResearchPlan(StrictRequest):
    steps: list[str] = Field(min_length=1, max_length=8)
    needs_data: bool = False
    needs_literature: bool = False
    save_report: bool = False
    clarification: str | None = None
    retrieval_query: str | None = Field(default=None, min_length=1, max_length=2000,
        description="需要查文献时，把当前问题结合历史改写为可独立理解的检索问题；不添加未经用户提供的事实")


class Condition(StrictRequest):
    column: str = Field(min_length=1, description="CSV 原始列名")
    operator: Literal["eq", "ne", "gt", "ge", "lt", "le"]
    value: str | float | bool = Field(description="比较值；数值条件用数字。多个条件按 AND 合并")


class SelectRequest(StrictRequest):
    file_name: str = Field(min_length=1, description="data/uploads 中的 CSV 文件名")
    conditions: list[Condition] = Field(default_factory=list, max_length=12)
    columns: list[str] | None = Field(default=None, description="保留列；null 保留所有列，包括样本 ID 和分组列")
    max_rows: int = Field(default=5, ge=1, le=100, strict=True, description="只限制展示，不限制保存和统计的数据")

    @model_validator(mode="after")
    def unique_columns(self):
        if self.columns is not None and (not self.columns or len(self.columns) != len(set(self.columns))):
            raise ValueError("columns 应是非空且不重复的列名列表，或 null")
        return self


class GroupRequest(StrictRequest):
    dataset_id: str = Field(min_length=1, description="筛选工具实际返回的数据集 ID，不能编造")
    group_by: str = Field(min_length=1, description="分组列名")
    value_column: str = Field(min_length=1, description="数值指标列名")
    unit_column: str | None = Field(default=None, description="独立实验单位 ID，如 mouse_id；不提供时只描述记录，不能称为独立样本")
    repeated: Literal["error", "mean"] = Field(default="error", description="同一单位同组重复测量：默认报错；只有用户明确同意先求均值时才选 mean")


class CompareRequest(GroupRequest):
    unit_column: str = Field(min_length=1, description="必填的独立实验单位 ID 列")
    group_a: str = Field(min_length=1, description="第一组标签，差值为 A-B")
    group_b: str = Field(min_length=1)
    design: Literal["independent"] = Field(description="仅支持用户已确认的两独立组设计；配对/重复测量不适用")
    method: Literal["welch"] = Field(description="用户明确要求或确认的双侧 Welch t 检验")

    @model_validator(mode="after")
    def distinct_groups(self):
        if self.group_a == self.group_b:
            raise ValueError("比较组必须不同")
        return self


class SearchRequest(StrictRequest):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=10, strict=True)
    score_threshold: float = Field(default=0.5, ge=0, le=1)


class ExportRequest(StrictRequest):
    dataset_id: str = Field(min_length=1)


class ClarificationRequest(StrictRequest):
    question: str = Field(min_length=1, max_length=1500, description="需要用户补充或确认的具体问题")


class ReviewResult(StrictRequest):
    supported: bool
    issues: list[str] = Field(default_factory=list, max_length=10)
