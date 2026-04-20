from pathlib import Path
from langchain_core.documents import Document


def load_text_documents(folder_path: str) -> list[Document]:
    docs = []
    folder = Path(folder_path)

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name}
            )
        )

    return docs