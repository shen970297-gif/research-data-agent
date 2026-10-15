import os
from pathlib import Path
from functools import lru_cache
import atexit

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

#这部分代码用于建完向量数据库之后，进行查询用的
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
COLLECTION_NAME = "research_knowledge"


def create_embedding_model():
    """查询时必须使用与建库时相同的 Embedding 模型。"""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "embedding-3"),
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )


@lru_cache(maxsize=1)
def load_vector_store():
    """连接到已经创建好的 Qdrant 本地集合。"""
    if not QDRANT_PATH.exists():
        raise FileNotFoundError("Qdrant 数据库不存在，请先运行 build_vector_store.py")

    return QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION_NAME,
        embedding=create_embedding_model(),
        path=str(QDRANT_PATH),
    )


def close_vector_store():
    """同一进程复用本地 Qdrant，退出时释放文件锁。"""
    if load_vector_store.cache_info().currsize:
        load_vector_store().client.close()
        load_vector_store.cache_clear()


atexit.register(close_vector_store)


def retrieve_docs(query: str, k: int = 3,score_threshold: float = 0.50):
    """把用户问题转换成向量，并返回最相关的文档片段。"""
    vector_store = load_vector_store()
    return vector_store.similarity_search_with_score(query, k=k,score_threshold=score_threshold)


if __name__ == "__main__":
        questions = [
            "这篇研究如何量化脑脊液清除？",
            "清除参数与年龄有什么关系？",
            "研究中的 PET 扫描持续多长时间？",
            "量子计算机的工作原理是什么？",
        ]

        for question in questions:
            print(f"\n{'=' * 20}")
            print(f"问题：{question}")

            results = retrieve_docs(
                query=question,
                k=3,
            )

            if not results:
                print("没有检索到达到相似度阈值的资料。")
                continue

            for index, (document, score) in enumerate(
                    results,
                    start=1,
            ):
                print(f"\n----- 结果 {index} -----")
                print(f"相似度分数：{score:.3f}")
                print(f"来源：{document.metadata['source']}")
                print(f"PDF 页码：{document.metadata.get('page')}")
                print("内容前 300 字：")
                print(document.page_content[:300])
