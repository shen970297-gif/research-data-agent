import re
from collections import Counter
from math import ceil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "reference"

PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?$",
    re.IGNORECASE,
)

REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(references|bibliography|参考文献)\s*$"
)


def normalize_line(line: str) -> str:
    """统一空格和大小写，方便比较不同页面的同一行文字。"""
    return re.sub(r"\s+", " ", line).strip().lower()


def find_repeated_page_lines(
    page_documents: list[Document],
) -> set[str]:
    """找出在多数页面重复出现的页眉或页脚文本。"""
    line_counter = Counter()

    for document in page_documents:
        page_lines = set()

        for line in document.page_content.splitlines():
            normalized_line = normalize_line(line)

            if len(normalized_line) >= 12:
                page_lines.add(normalized_line)

        # 同一页中重复出现的文字，只计数一次。
        line_counter.update(page_lines)

    min_repeat_count = max(
        2,
        ceil(len(page_documents) * 0.5),
    )

    return {
        line
        for line, count in line_counter.items()
        if count >= min_repeat_count
    }


def clean_pdf_text(
    text: str,
    repeated_lines: set[str],
) -> str:
    """删除重复页眉、页脚和单独页码。"""
    clean_lines = []

    for line in text.splitlines():
        normalized_line = normalize_line(line)

        if not normalized_line:
            clean_lines.append("")
            continue

        if normalized_line in repeated_lines:
            continue

        if PAGE_NUMBER_PATTERN.fullmatch(normalized_line):
            continue

        clean_lines.append(line.strip())

    cleaned_text = "\n".join(clean_lines)

    # 避免 PDF 产生过多空行。
    return re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()


def load_markdown_documents() -> list[Document]:
    """加载 docs 目录及其子目录中的 Markdown 文档。"""
    documents = []

    for file_path in sorted(DOCS_DIR.rglob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "file_type": "markdown",
                },
            )
        )

    return documents


def load_pdf_documents() -> list[Document]:
    """逐页加载 PDF，自动清理重复页眉页脚，并保留页码来源。"""
    documents = []

    for file_path in sorted(DOCS_DIR.rglob("*.pdf")):
        reader = PdfReader(str(file_path))
        paper_title = (
            reader.metadata.title
            if reader.metadata and reader.metadata.title
            else file_path.stem
        )

        loader = PyPDFLoader(str(file_path))
        page_documents = loader.load()

        repeated_lines = find_repeated_page_lines(
            page_documents
        )

        reached_references = False

        for document in page_documents:
            content = clean_pdf_text(
                text=document.page_content,
                repeated_lines=repeated_lines,
            )

            if reached_references:
                continue
            heading = REFERENCE_HEADING_PATTERN.search(content)
            if heading:
                # 保留同页参考文献标题之前的正文，避免整页结论被误删。
                content = content[:heading.start()].strip()
                reached_references = True
            if not content:
                continue

            metadata = {
                **document.metadata,
                "source": file_path.name,
                "file_path": str(file_path),
                "file_type": "pdf",
                "paper_title": paper_title,
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    return documents


def load_documents() -> list[Document]:
    """统一加载 Markdown 和 PDF 文档。"""
    markdown_documents = load_markdown_documents()
    pdf_documents = load_pdf_documents()

    return markdown_documents + pdf_documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"加载 Document 总数：{len(documents)}")

    for index, document in enumerate(documents, start=1):
        print(f"\n===== Document {index} =====")
        print("正文前 200 个字符：")
        print(document.page_content[:200])

        print("Metadata：")
        print(document.metadata)
