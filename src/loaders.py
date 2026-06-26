import re
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

_LOREM_IPSUM_PATTERN = re.compile(r"(lorem ipsum|mauris facilisis|integer nec|pellentesque)", re.IGNORECASE)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def _is_placeholder(text: str) -> bool:
    return bool(_LOREM_IPSUM_PATTERN.search(text))


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


def load_wordpress_documents(posts_csv: str, postmeta_csv: str) -> list[Document]:
    posts = pd.read_csv(posts_csv)
    postmeta = pd.read_csv(postmeta_csv)

    posts = posts[
        (posts["post_status"] == "publish") &
        (posts["post_type"].isin(["post", "page"]))
    ].copy()

    useful_keys = ["_yoast_wpseo_focuskw", "_yoast_wpseo_metadesc", "_yoast_wpseo_title", "desc_page"]
    meta_wide = (
        postmeta[postmeta["meta_key"].isin(useful_keys)]
        .pivot_table(index="post_id", columns="meta_key", values="meta_value", aggfunc="first")
        .reset_index()
    )

    merged = posts.merge(meta_wide, left_on="ID", right_on="post_id", how="left")

    docs = []
    for _, row in merged.iterrows():
        content = _strip_html(str(row.get("post_content", "") or ""))
        if not content:
            continue

        parts = []

        title = str(row.get("_yoast_wpseo_title") or row.get("post_title", "")).strip()
        if title and title != "nan":
            parts.append(f"Tytuł: {title}")

        focuskw = str(row.get("_yoast_wpseo_focuskw", "") or "").strip()
        if focuskw and focuskw != "nan":
            parts.append(f"Słowa kluczowe: {focuskw}")

        metadesc = str(row.get("_yoast_wpseo_metadesc", "") or "").strip()
        if metadesc and metadesc != "nan":
            parts.append(f"Opis: {metadesc}")

        desc_page = str(row.get("desc_page", "") or "").strip()
        if desc_page and desc_page != "nan" and not _is_placeholder(desc_page):
            parts.append(f"Opis strony: {desc_page}")

        parts.append(content)
        full_text = "\n".join(parts)

        docs.append(Document(
            page_content=full_text,
            metadata={
                "source": str(row.get("post_name", row["ID"])),
                "type": "wordpress",
                "post_id": int(row["ID"]),
                "post_type": str(row.get("post_type", "")),
            }
        ))

    return docs


def load_all_documents(folder_path: str) -> list[Document]:
    docs = []
    docs.extend(load_text_documents(folder_path))
    docs.extend(load_pdf_documents(folder_path))
    return docs