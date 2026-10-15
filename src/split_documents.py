from langchain_text_splitters import RecursiveCharacterTextSplitter

from load_documents import load_documents


def split_documents():
    documents = load_documents()#这里是从之前的load_documents导入的类，定义的对象，是文件转换成的document对象

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )#定义一个分割器，这个分割器是已经配置好的工具。add_start_index=True 会在 metadata 中增加起始位置

    chunks = splitter.split_documents(documents)#split_documents是RecursiveCharacterTextSplitter 类定义好的方法
    #splitter是前面我们自己创建的类的对象，所以它有split_documents方法，返回的是以document对象为元素的列表
    #splitter.split_text(text)如果用这个得到的是字符串列表，分割的是text文件

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    return chunks


if __name__ == "__main__":
    chunks = split_documents()

    print(f"切分后的 Chunk 数量：{len(chunks)}")
    print('chunks:', chunks)
    for index, chunk in enumerate(chunks):
        print(f"\n===== Chunk {index + 1} =====")
        print("正文：")
        print(chunk.page_content)

        print("\nMetadata：")
        print(chunk.metadata)
