import ast
import csv
import operator
from pathlib import Path
import pandas as pd
import json
import math


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
REPORT_DIR = PROJECT_ROOT / "reports"


def profile_dataset(file_name: str) -> dict:
    """检查整份 CSV 的行列数、字段类型和缺失情况。"""
    file_path = (UPLOAD_DIR / file_name).resolve()

    if UPLOAD_DIR.resolve() not in file_path.parents:
        raise ValueError("只能读取 data/uploads 目录中的文件")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("目前只支持 CSV 文件")

    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在或不是文件：{file_name}")

    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV 文件为空或没有可读取的字段") from error

    if df.empty:
        raise ValueError("CSV 文件没有数据行")

    column_profiles = {}

    for column_name in df.columns:
        column = df[column_name]

        column_profiles[column_name] = {
            "dtype": str(column.dtype),
            "missing_count": int(column.isna().sum()),
            "unique_count": int(column.nunique(dropna=True)),
        }

    preview_text = df.head(5).to_json(
        orient="records",
        force_ascii=False,
        double_precision=15,
    )
    preview_rows = json.loads(preview_text)

    return {
        "file_name": file_name,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "column_profiles": column_profiles,
        "preview_rows": preview_rows,
        "row_count_preview": len(preview_rows),
    }
#读取特定时间区间的数值，输入的是有格式要求的参数，这个格式可以有大模型分析用户自然语言得出。
def filter_dataset(
    file_name: str,
    columns: list[str],
    time_min: float,
    time_max: float,
    max_rows: int = 20,
) -> dict:
    """按 time 区间筛选记录，并返回指定字段。"""
    if not columns:
        raise ValueError("至少需要指定一个返回字段")

    if not math.isfinite(time_min) or not math.isfinite(time_max):
        raise ValueError("时间上下限必须是有限数值")

    if time_min > time_max:
        raise ValueError("时间下限不能大于上限")

    if type(max_rows) is not int or not 1 <= max_rows <= 100:
        raise ValueError("max_rows 必须是1到100之间的整数")

    file_path = (UPLOAD_DIR / file_name).resolve()

    if UPLOAD_DIR.resolve() not in file_path.parents:
        raise ValueError("只能读取 data/uploads 目录中的文件")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("目前只支持 CSV 文件")

    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在或不是文件：{file_name}")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    for column_name in ["time"] + columns:
        if column_name not in df.columns:
            raise ValueError(f"字段不存在：{column_name}")

    df["time"] = pd.to_numeric(
        df["time"],
        errors="raise",
    )

    if df["time"].isin([math.inf, -math.inf]).any():
        raise ValueError("time 字段包含无穷大")

    mask = (
        df["time"].notna()
        & (df["time"] >= time_min)
        & (df["time"] <= time_max)
    )

    filtered = df.loc[mask, columns]
    returned = filtered.head(max_rows)

    rows = json.loads(
        returned.to_json(
            orient="records",
            force_ascii=False,
            double_precision=15,
        )
    )

    return {
        "file_name": file_name,
        "filters": {
            "time_min": time_min,
            "time_max": time_max,
            "include_bounds": True,
        },
        "columns": columns,
        "matched_row_count": int(len(filtered)),
        "returned_row_count": int(len(returned)),
        "truncated": len(returned) < len(filtered),
        "missing_time_count": int(df["time"].isna().sum()),
        "rows": rows,
    }

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}    #允许使用的运算符白名单


def _safe_calculate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_calculate(node.left)
        right = _safe_calculate(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_calculate(node.operand))

    raise ValueError("表达式中包含不支持的内容")


def calculate(expression: str) -> dict:
    """安全计算简单数学表达式，不执行任意 Python 代码。"""
    if len(expression) > 100:
        raise ValueError("表达式过长")

    tree = ast.parse(expression, mode="eval")
    result = _safe_calculate(tree.body)

    return {
        "expression": expression,
        "result": result,
    }


def save_report(title: str, content: str) -> dict:
    """将报告保存为 Markdown 文件。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    safe_title = "".join(
        character for character in title if character.isalnum() or character in " _-"
    ).strip()
    '''文件命名规则⬆️'''

    if not safe_title:
        safe_title = "untitled_report"
        '''如果没解析出来有效文件名，就用默认的'''
    report_path = REPORT_DIR / f"{safe_title}.md"
    report_path.write_text(content, encoding="utf-8")

    return {
        "title": title,
        "path": str(report_path),
        "status": "saved",
    }

if __name__ == "__main__":
    result = filter_dataset(
        file_name="定量-总表.csv",
        columns=["file", "time", "target_mean", "sbr"],
        time_min=14,
        time_max=45,
        max_rows=5,
    )

    print("匹配记录数：", result["matched_row_count"])
    print("返回记录数：", result["returned_row_count"])
    print("是否截取展示：", result["truncated"])
    print("时间缺失数：", result["missing_time_count"])

    print("\n返回数据：")
    for row in result["rows"]:
        print(row)

