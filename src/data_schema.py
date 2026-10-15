
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pydantic import (
    model_validator,
)


class FilterRequest(BaseModel):
    """规定筛选工具的参数结构，并检查参数之间的关系。"""


    model_config = ConfigDict(extra="forbid")
    file_name: str = Field(
        min_length=1,
        description="用户指定的 CSV 文件名，文件位于 data/uploads 目录。",
    )

    columns: list[str] = Field(
        min_length=1,
        description=(
            "用户希望返回的原始列名列表，不允许重复。"
            "例如用户说只返回file、time、sbr，"
            "则填写列表[\"file\", \"time\", \"sbr\"]。"
        ),
    )

    time_min: float = Field(
        allow_inf_nan=False,
        description=(
            "固定列名 time 的筛选下限，包含此值。"
            "用户说time在14到45之间时，此参数为14。"
        ),
    )

    time_max: float = Field(
        allow_inf_nan=False,
        description=(
            "固定列名 time 的筛选上限，包含此值。"
            "用户说time在14到45之间时，此参数为45。"
            "必须大于或等于time_min。"
        ),
    )

    max_rows: int = Field(
        default=20,
        ge=1,
        le=100,
        strict=True,
        description="最多展示的匹配记录条数，未指定时使用20。",
    )

    @model_validator(mode="after")#装饰器。你暂时可以把它理解为：给下面的函数做标记，让 Pydantic 在校验参数时自动调用它
    def check_filter_rules(self) -> "FilterRequest":
        """检查时间上下限和返回字段。"""

        if self.time_min > self.time_max:
            raise ValueError(
                "time_min 不能大于 time_max，请依据原始问题检查时间范围"
            )

        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                "columns 不能包含重复字段，请检查返回字段列表"
            )

        return self



if __name__ == "__main__":
    raw_json = """
    {
        "file_name": "定量-总表.csv",
        "columns": ["time", "sbr"],
        "time_min": 14,
        "time_max": 45,
        "max_rows": 5
    }
    """

    try:
        request = FilterRequest.model_validate_json(raw_json)
        print("校验通过：", request)

    except ValidationError as error:
        print("参数校验失败")

        errors = error.errors(
            include_input=False,
            include_context=False,
            include_url=False,
        )

        for item in errors:
            print("错误字段：", item["loc"])
            print("错误类型：", item["type"])
            print("错误原因：", item["msg"])