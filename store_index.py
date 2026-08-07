from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import create_embeddings, text_split


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY was not found in the .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


INDEX_NAME = "medical-chatbot"

MEDLINEPLUS_DIRECTORY = Path(
    "data/processed/medlineplus"
)

CURATED_DIRECTORY = Path(
    "data/curated/ready_documents"
)

LANGUAGES_TO_INDEX = {"English"}

# True deletes and rebuilds the existing Pinecone index.
RECREATE_INDEX = True


def load_medlineplus_documents(
    directory: Path,
    allowed_languages: set[str],
) -> list[Document]:
    """Load English MedlinePlus JSON files."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}. "
            "Run scripts/parse_medlineplus.py first."
        )

    documents: list[Document] = []
    skipped_language = 0
    skipped_empty = 0

    for json_path in sorted(directory.glob("*.json")):
        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            item = json.load(file)

        language = str(
            item.get("language", "")
        ).strip()

        if language not in allowed_languages:
            skipped_language += 1
            continue

        title = str(
            item.get("title", "")
        ).strip()

        summary = str(
            item.get("summary", "")
        ).strip()

        meta_description = str(
            item.get("meta_description", "")
        ).strip()

        synonyms = [
            str(value).strip()
            for value in item.get("synonyms", [])
            if str(value).strip()
        ]

        groups = [
            str(value).strip()
            for value in item.get("groups", [])
            if str(value).strip()
        ]

        if not title or not summary:
            skipped_empty += 1
            continue

        page_content_parts = [
            f"Title: {title}",
            f"Language: {language}",
        ]

        if synonyms:
            page_content_parts.append(
                f"Alternative names: {', '.join(synonyms)}"
            )

        if groups:
            page_content_parts.append(
                f"Categories: {', '.join(groups)}"
            )

        if meta_description:
            page_content_parts.append(
                f"Description: {meta_description}"
            )

        page_content_parts.extend(
            [
                "",
                summary,
            ]
        )

        page_content = "\n".join(
            page_content_parts
        ).strip()

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "topic_id": str(
                        item.get("id", "")
                    ),
                    "title": title,
                    "language": language,
                    "source": str(
                        item.get(
                            "source",
                            "MedlinePlus",
                        )
                    ),
                    "source_url": str(
                        item.get(
                            "source_url",
                            "",
                        )
                    ),
                    "date_created": str(
                        item.get(
                            "date_created",
                            "",
                        )
                    ),
                    "categories": ", ".join(groups),
                    "document_type": "medlineplus_topic",
                    "file": str(json_path),
                },
            )
        )

    print(
        f"MedlinePlus documents loaded: "
        f"{len(documents)}"
    )

    print(
        f"MedlinePlus documents skipped "
        f"because of language: {skipped_language}"
    )

    print(
        f"MedlinePlus documents skipped "
        f"because content was empty: {skipped_empty}"
    )

    return documents


def load_curated_documents(
    directory: Path,
) -> list[Document]:
    """Load curated everyday-topic JSON files."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}. "
            "Run scripts/build_curated_documents.py first."
        )

    documents: list[Document] = []
    skipped_empty = 0

    for json_path in sorted(directory.glob("*.json")):
        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            item = json.load(file)

        title = str(
            item.get("title", "")
        ).strip()

        category = str(
            item.get("category", "")
        ).strip()

        mapped_title = str(
            item.get("mapped_title", "")
        ).strip()

        page_content = str(
            item.get("page_content", "")
        ).strip()

        source = str(
            item.get(
                "source",
                "MedlinePlus",
            )
        ).strip()

        source_url = str(
            item.get(
                "source_url",
                "",
            )
        ).strip()

        if not title or not page_content:
            skipped_empty += 1
            continue

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "topic_id": str(
                        item.get("id", "")
                    ),
                    "title": title,
                    "category": category,
                    "mapped_title": mapped_title,
                    "language": "English",
                    "source": source,
                    "source_url": source_url,
                    "document_type": "curated_topic",
                    "mapping_type": str(
                        item.get(
                            "mapping_type",
                            "",
                        )
                    ),
                    "mapping_score": float(
                        item.get(
                            "mapping_score",
                            0,
                        )
                    ),
                    "file": str(json_path),
                },
            )
        )

    print(
        f"Curated documents loaded: "
        f"{len(documents)}"
    )

    print(
        f"Curated documents skipped "
        f"because content was empty: {skipped_empty}"
    )

    return documents


def prepare_pinecone_index(
    pc: Pinecone,
    index_name: str,
    recreate: bool,
) -> None:
    """Create or recreate the Pinecone index."""

    existing_indexes = pc.list_indexes().names()

    if recreate and index_name in existing_indexes:
        print(
            f"Deleting existing index: "
            f"{index_name}"
        )

        pc.delete_index(index_name)

        while index_name in pc.list_indexes().names():
            time.sleep(2)

    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        print(
            f"Creating index: "
            f"{index_name}"
        )

        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while not pc.describe_index(
            index_name
        ).status["ready"]:
            print(
                "Waiting for Pinecone index..."
            )
            time.sleep(5)

    else:
        print(
            "Pinecone index already exists."
        )


def main() -> None:
    print(
        "1. Loading MedlinePlus documents..."
    )

    medlineplus_documents = (
        load_medlineplus_documents(
            directory=MEDLINEPLUS_DIRECTORY,
            allowed_languages=LANGUAGES_TO_INDEX,
        )
    )

    print(
        "2. Loading curated documents..."
    )

    curated_documents = load_curated_documents(
        directory=CURATED_DIRECTORY,
    )

    all_documents = (
        medlineplus_documents
        + curated_documents
    )

    print(
        f"Total documents before chunking: "
        f"{len(all_documents)}"
    )

    if not all_documents:
        raise ValueError(
            "No documents were loaded."
        )

    print(
        "3. Splitting documents into chunks..."
    )

    chunks = text_split(
        all_documents
    )

    print(
        f"Total text chunks: "
        f"{len(chunks)}"
    )

    print(
        "4. Loading embedding model..."
    )

    embeddings = create_embeddings()

    print(
        "Embedding model loaded."
    )

    print(
        "5. Preparing Pinecone index..."
    )

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    prepare_pinecone_index(
        pc=pc,
        index_name=INDEX_NAME,
        recreate=RECREATE_INDEX,
    )

    print(
        "6. Uploading chunks to Pinecone..."
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    print(
        "7. Upload completed."
    )

    index = pc.Index(
        INDEX_NAME
    )

    stats = index.describe_index_stats()

    print(
        "8. Pinecone index statistics:"
    )

    print(stats)


if __name__ == "__main__":
    main()