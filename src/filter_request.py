#如果返回参数有误，重新发给LLM进行修改，定义一个最大次数
import json
from openai import APITimeoutError
from pydantic import ValidationError

from LLM import HelloAgentsLLM
from data_schema import FilterRequest
from tools import filter_dataset


def extract_filter_request(
    llmclient: HelloAgentsLLM,
    messages: list[dict],
    max_repairs: int = 3,
) -> FilterRequest:
    """提取参数，校验失败后最多修正指定次数。"""
    if type(max_repairs) is not int or max_repairs < 0:
        raise ValueError("max_repairs 必须是非负整数")

    conversation = messages.copy()

    for attempt in range(max_repairs + 1):
        print(
            f"参数提取第 {attempt + 1} 次，"
            f"最多 {max_repairs + 1} 次"
        )

        assistant_message = llmclient.think(
            messages=conversation,
            json_mode=True,
        )

        raw_json = assistant_message.content or "{}"

        try:
            request = FilterRequest.model_validate_json(raw_json)
            print("参数校验通过")
            return request

        except ValidationError as error:#把 Pydantic 的校验错误整理成 JSON 字符串，保存到 feedback，方便后续反馈给模型。
            feedback = json.dumps(#封装字典为JSON的函数
                error.errors(
                    include_input=False,
                    include_context=False,
                    include_url=False,
                ),
                ensure_ascii=False,#中文用中文显示
                indent=2,#indent=2：使用两个空格缩进，方便阅读
            )

            if attempt == max_repairs:
                raise ValueError(
                    f"首次提取及 {max_repairs} 次修正均失败。"
                    f"最后的校验错误：\n{feedback}"
                ) from error

            print("校验失败，将错误反馈给模型修正")

            conversation.append(
                {
                    "role": "assistant",
                    "content": raw_json,
                }
            )

            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"上一次JSON未通过校验：\n{feedback}\n\n"
                        "请依据原始问题修正，重新返回完整JSON。"
                        "保留正确参数，不要编造缺失的必要信息。"
                        "不要返回解释或Markdown代码块。"
                    ),
                }
            )


if __name__ == "__main__":
    llmclient = HelloAgentsLLM()

    question = (
        "读取定量-总表.csv，筛选 time 在14到45之间的记录，"
        "包含两端，只返回 file、time、target_mean、sbr，"
        "最多展示5条。"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你负责从用户问题中提取CSV筛选参数。"
                "只返回合法JSON，不要返回解释或Markdown代码块。"
                "字段为file_name、columns、time_min、time_max、max_rows。"
                "file_name是文件名，columns是原始字段名组成的列表。"
                "time_min和time_max是time字段的数值边界，包含两端。"
                "max_rows未指定时使用20。"
                "不要猜测文件名、字段名或时间边界，不要换算时间单位。"
                "如果缺少必要信息，或要求当前不支持的筛选条件，"
                "返回空JSON对象 {}。"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    try:
        request = extract_filter_request(
            llmclient=llmclient,
            messages=messages,
            max_repairs=3,
        )

        arguments = request.model_dump()

        print("\n校验后的参数：")
        print(json.dumps(arguments, ensure_ascii=False, indent=2))

        result = filter_dataset(**arguments)

        print("\n匹配记录数：", result["matched_row_count"])
        print("返回记录数：", result["returned_row_count"])
        print("是否截取展示：", result["truncated"])

    except APITimeoutError:
        print("参数提取超时，本次没有执行筛选工具。")

    except ValueError as error:
        print("参数提取或筛选失败：", error)

    except FileNotFoundError as error:
        print("文件检查失败：", error)