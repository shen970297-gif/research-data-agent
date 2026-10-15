# 科研文献与实验数据分析 Agent

在原 seven_day 项目上升级，目标是完成“用户问题 → 本地文献/数据工具 → 证据整理 → 可追溯报告”。

## 当前状态

- LangGraph：规划 → 多轮工具调用 → 审查 → 保存，支持一条任务同时查询文献和分析 CSV。
- 独立 RAG 分支：纯文献问题走规划 → 本地检索 → 证据增强生成 → 引用审查 → 保存；混合任务仍走工具协作。
- 工具管理、Pydantic 校验、错误反馈；同一工具连续错误允许首次调用 + 3 次修正。
- CSV 条件筛选，完整数据快照和 ID，分组统计、两独立组双侧 Welch 检验、分组散点图、CSV 导出。
- 本地 Qdrant 文献检索、片段 ID/文件/页码引用；保留原知识库，不自动重建。
- SQLite 跨轮会话，JSONL 运行索引，每个任务独立保存参数、结果、来源和报告。
- Markdown 报告；不足或失败时标记状态，保留已完成产物。

**离线流程已测试；真实智谱模型和真实 Embedding 联调由用户运行，未通过代理自动调用。** 模型辅助审查不是科研专家复核。`completed` 表示程序检查和辅助审查通过，不保证科学结论正确。

## 运行

### 首次下载后的准备

本仓库不包含 API key、真实实验 CSV、论文 PDF、向量索引、会话数据库或运行报告。三份 Markdown 文档用于演示；`--demo` 会自行生成合成数据，无需真实文件或 API key。

Windows PowerShell（先安装 Python 3.11）：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
# 仅在 .env 不存在时复制，避免覆盖已有配置
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
New-Item -ItemType Directory -Force data/uploads | Out-Null
& .\.venv\Scripts\python.exe .\src\main.py --demo
```

调用真实模型前，在本机 `.env` 中填写配置，将自己的 CSV 放到 `data/uploads`，文献放到 `data/reference`。首次文献检索前运行 `src/build_vector_store.py` 建库，会调用配置的 Embedding API；已有索引需要替换时才使用后文的 `--rebuild`。

### 已有环境

在项目目录的 PowerShell 中运行。现有 `.venv` 继续使用 Python 3.11。

```powershell
# 无需密钥：合成数据 + 脚本化模型，不调用任何模型 API
& .\.venv\Scripts\python.exe .\src\main.py --demo

# 离线测试，不读取真实数据或调用模型
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 真实单次任务，会调用 .env 配置的模型
& .\.venv\Scripts\python.exe .\src\main.py "筛选定量-总表.csv中time在14到45之间的记录，只返回file、time、sbr，展示5条。" --session lesson1

# 连续对话；输入 exit 退出
& .\.venv\Scripts\python.exe .\src\main.py --session lesson1
```

PyCharm 也可继续运行 `src/day4_research_agent.py` 或 `src/tool_agent.py`，二者均进入统一命令行。直接运行无参数时，等待你输入问题。换 `--session` 值开启独立会话；默认 `default` 会持续保存历史。

新环境可安装 `requirements-lock.txt`。`.env.example` 给出配置名，真实密钥只放本机 `.env`。

## 代码阅读顺序

| 文件 | 作用 |
|---|---|
| `src/main.py` | 接收用户问题、会话 ID、显示结果 |
| `src/research_agent.py` | LangGraph 总流程、RAG 分支和调用预算 |
| `src/rag_pipeline.py` | RAG 专用生成提示词，绑定本次检索原文与引用 ID |
| `src/tool_registry.py` | 工具名、说明、参数模型、执行函数的对应关系 |
| `src/research_schema.py` | 规划、分析、检索和审查的参数规范 |
| `src/tool_agent.py` | Function Calling 消息配对、参数错误反馈与停止条件 |
| `src/research_tools.py` | Pandas/SciPy/Matplotlib 真实计算和文献检索适配 |
| `src/research_storage.py` | 完整数据快照、文件哈希、SQLite 会话 |
| `src/research_report.py` | 引用 ID 检查、原始计算附录、Markdown 报告 |

原来的 `data_schema.py` / `filter_request.py` / `tools.py` 保留兼容和学习用途。旧 profile 函数未注册，也不作为主流程的数据概览功能。原互斥路由被统一图接替，备份中保留修改前代码。

## 数据和证据约定

- 用户 CSV 位于 `data/uploads`，PDF/Markdown 位于 `data/reference`。
- `select_rows` 多个条件按 AND 合并，只支持数值大小比较和文本/数值相等比较；不执行模型生成的 Python/SQL。
- `max_rows` 只限制给模型展示多少条。计算使用 `dataset_id` 指向的完整快照。
- 修改时间范围时，从原始 CSV 重新选择；CSV 内容变化后，拒绝继续使用旧快照。
- CSV 采用 Pandas 默认类型推断和缺失值规则；不会猜测年龄组、时间单位或样本 ID。
- 分组统计默认排除分组/指标/指定单位列缺失的记录，并报告数量；不自动删除异常值。
- 指定 `unit_column` 时，同单位同组重复记录默认报错；只有用户确认后使用 `repeated=mean` 聚合。
- Welch 仅支持已确认的两独立组、每组至少 2 个有效单位。跨组重复 ID 会被拒绝。暂不支持配对、混合模型、多组/多重比较校正。
- 文献相似度不代表科学正确性。引用格式为 `[E...]`，与保存的原文片段关联。

## 状态、预算与输出

`needs_input`：等待补充；`completed`：流程检查通过；`insufficient_evidence`：RAG 检索无有效证据、未生成科研结论；`needs_review`：证据/审查未通过；`partial`：预算或运行错误中止；`failed`：规划失败。

## 独立 RAG 路径

问题示例：“根据本地文献，这篇研究如何量化脑脊液清除？请给出出处。”

规划结果为 `needs_literature=true`、`needs_data=false` 时，代码直接执行检索，模型不需要先选择工具。`retrieval_query` 把历史指代改写为自包含问题；未提供则用原问题。

`rag_retrieve` 调用已有 Qdrant 检索，默认最多 5 个片段、阈值 0.5，过滤空白与重复片段。没有有效证据时立即返回资料不足，不用历史证据凑答案。

`rag_generate` 把当前问题、检索问题和证据原文放入上下文，要求关键结论标注 `[E...]`。审查只允许引用本次检索的 ID；失败时在同一批证据上修订，最多两次审查，且受全局调用上限限制。当前不会自动反复重检索或降低阈值。

命令行显示引用文件和页码，`result.json` 中的 `route` 标识分支、`sources` 给出实际引用来源。只有 ID 有效不代表内容支持结论，仍保留辅助语义审查和人工复核边界。

默认所有规划、执行、审查合计最多 12 次模型调用；最多 32 次工具调用；最多 2 次报告审查。修正次数同样受全局预算限制。最后一次模型调用仍要求工具时不执行这些工具。工具网络异常停止，不无限重试。

输出在 `reports/runs/<run_id>/`：

- `manifest.json`：数据来源、哈希、证据和产物路径。
- `tool_results.json`：本次成功调用的参数和结果。
- `trace.json`：节点顺序、模型次数、工具成功/失败、未解决问题。
- `result.json`：最终回答与状态。
- 用户要求报告时生成 `report.md`；图和导出 CSV 保存到同一目录。

会话保存在 `data/sessions.sqlite`；模型读取最近 6 轮任务摘要和最近 20 个数据集元数据，完整历史仍在数据库。该版本是单用户串行 CLI，没有多人鉴权、并发会话写锁或任务崩溃后的自动断点续跑。

## 知识库

现有 Qdrant 可继续查询。本次未重建，也没有调用 Embedding API。新分块参数 1000/150 只在下一次建库时生效。

```powershell
# 仅在明确要重建时运行；替换原 collection，消耗 Embedding API
& .\.venv\Scripts\python.exe .\src\build_vector_store.py --rebuild
```

本地 Qdrant 同进程复用连接；不要在两个进程同时打开同一嵌入式数据库目录。

## 开源参考与实现边界

参考 [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 的研究规划、执行、来源整理与审查分工。参考版本：`3582a6841fe7dc801e2a34e34ee408641aea9bce`。

按最新要求直接扩展现有代码：**当前不是该仓库的 fork，也没有安装或整合其完整运行时**。实验数据工具和本项目的流程适配独立实现。这样保留现有 Python 3.11 环境；后续确需上游特定模块时再做独立适配。

后续功能：专业术语表与混合检索、联网/MCP、更多实验设计、前端、真正的检索与任务成功率评估。已有离线测试不能替代真实模型和真实科研任务评估。
