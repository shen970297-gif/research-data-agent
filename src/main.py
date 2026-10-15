"""命令行入口；只有用户运行本文件时才会调用真实模型。"""
import argparse

from research_agent import run_research


def main():
    parser = argparse.ArgumentParser(description="科研文献与数据分析 Agent")
    parser.add_argument("question", nargs="?", help="自然语言任务；不填则进入连续对话")
    parser.add_argument("--session", default="default", help="同一会话沿用此 ID；换 ID 开启新会话")
    parser.add_argument("--max-rounds", type=int, default=12, help="规划、执行和审查的模型调用总上限")
    parser.add_argument("--demo", action="store_true", help="使用合成数据运行离线演示，不调用模型 API")
    args = parser.parse_args()
    if args.demo:
        from offline_demo import run_offline_demo
        run_offline_demo()
        return
    def run(question):
        try:
            result = run_research(question, session_id=args.session, max_rounds=args.max_rounds,
                                  progress=lambda stage: print("执行节点：", stage))
            print("\n状态：", result["status"])
            print(result["answer"])
            for source in result["sources"]:
                page = f"第 {source['page']} 页" if source['page'] is not None else "无页码"
                print(f"[{source['evidence_id']}] {source['source']}，{page}")
            print("\n本次记录：", result["run_directory"])
            for artifact in result["artifacts"]:
                print(artifact["kind"], artifact["path"])
        except Exception as error:
            print("任务未完成，错误类型：", type(error).__name__)
    if args.question:
        run(args.question)
    else:
        print("输入 exit 退出。同一个 session 会记住已保存的任务上下文。")
        while True:
            question = input("\n问题：").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                run(question)


if __name__ == "__main__":
    main()
