from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_text_documents(folder_path: str) -> list[Document]:
    docs = []
    folder = Path(folder_path)

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name, "type": "txt"}
            )
        )

    return docs


def load_pdf_documents(folder_path: str) -> list[Document]:
    docs = []
    folder = Path(folder_path)

    for file_path in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        pdf_docs = loader.load()

        for doc in pdf_docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["type"] = "pdf"

        docs.extend(pdf_docs)

    return docs


def load_all_documents(folder_path: str) -> list[Document]:
    docs = []
    docs.extend(load_text_documents(folder_path))
    docs.extend(load_pdf_documents(folder_path))
    return docs