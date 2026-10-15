import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore#这是 LangChain 对 Qdrant 的封装。它负责把文档、向量和 Qdrant 连接起来

from split_documents import split_documents


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
COLLECTION_NAME = "research_knowledge"

#创建一个调用embedding模型的函数，把相关的参数写进去。后续通过该函数创建embedding对象。写为函数格式是为了方便管理
# 本身只是完成模型对象的初始化，不会自动把 Document 转换成向量，也不会进行相似度计算。
def create_embedding_model():
    """创建用于建库和查询的同一个 Embedding 模型。"""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "embedding-3"),
        api_key=os.environ["LLM_API_KEY"],#如果LLM_API_KEY不存在会直接报错，但是os.getenv("LLM_API_KEY"，默认值),这个不存在只会返回none或者默认值，不会直接报错
        base_url=os.environ["LLM_BASE_URL"],
    )

#创建一个chunks通过embedding模型转换成向量，并存储为一个向量数据库对象，的函数
def build_vector_store(rebuild: bool = False):
    if QDRANT_PATH.exists() and any(QDRANT_PATH.iterdir()) and not rebuild:
        raise ValueError("已有本地知识库。仅查询无需重建；确认重建时运行本文件并传入 --rebuild。")
    chunks = split_documents()

    embedding_model = create_embedding_model()
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,#刚刚创建的embedding模型对象，有自己的方法
        path=str(QDRANT_PATH),#数据保存到哪个本地目录
        collection_name=COLLECTION_NAME,
        force_recreate=rebuild,
    )
    #QdrantVectorStore.from_documents（）会把填进去的embedding模型当做工具调用，生成向量数据库对象

    return vector_store, embedding_model#函数返回了两个对象，第一个对象是已经建立好的向量数据库对象。不是普通的列表或字典，而是一个可以调用方法的对象
#第二个是返回了embedding_model类，作用是向量转换，方便后续用同一个embedding模型转换


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="明确重建现有 collection，将调用 Embedding API")
    args = parser.parse_args()
    vector_store, embedding_model = build_vector_store(rebuild=args.rebuild)

    print("Qdrant 向量数据库创建成功")
    print(f"存储位置：{QDRANT_PATH}")
    print(f"集合名称：{COLLECTION_NAME}")

    results = vector_store.similarity_search(
        query="这篇研究如何量化脑脊液清除？",
        k=3,
    )
    #similarity_search是向量数据库对象可以调用的方法，会自动完成接受用户问题——转换为向量——与已有向量进行对比相似度——用列表返回最相关的 Document 对象。
    # K代表从高到低返回的个数


    print("\n检索结果：")

    for index, document in enumerate(results, start=1):
        print(f"\n===== 结果 {index} =====")
        print("内容：")
        print(document.page_content)
        print("来源：")
        print(document.metadata)
