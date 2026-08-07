from __future__ import annotations

import json
import re
from pathlib import Path


MAPPED_CATALOG_FILE = Path(
    "data/curated/topic_catalog_mapped.json"
)

OUTPUT_DIR = Path(
    "data/curated/ready_documents"
)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)

    return value.strip("-") or "topic"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_document(topic: dict) -> dict | None:
    medlineplus_file = str(
        topic.get("medlineplus_file", "")
    ).strip()

    if not medlineplus_file:
        return None

    source_path = Path(medlineplus_file)

    if not source_path.exists():
        print(
            f"Missing MedlinePlus file for "
            f"{topic.get('title')}: {source_path}"
        )
        return None

    medlineplus_data = load_json(source_path)

    topic_title = str(
        topic.get("title", "")
    ).strip()

    category = str(
        topic.get("category", "")
    ).strip()

    mapped_title = str(
        topic.get(
            "related_medlineplus_topic",
            "",
        )
    ).strip()

    summary = str(
        medlineplus_data.get("summary", "")
    ).strip()

    source = str(
        medlineplus_data.get(
            "source",
            "MedlinePlus",
        )
    ).strip()

    source_url = str(
        medlineplus_data.get(
            "source_url",
            "",
        )
    ).strip()

    synonyms = [
        str(value).strip()
        for value in medlineplus_data.get(
            "synonyms",
            [],
        )
        if str(value).strip()
    ]

    groups = [
        str(value).strip()
        for value in medlineplus_data.get(
            "groups",
            [],
        )
        if str(value).strip()
    ]

    aliases = [
        str(value).strip()
        for value in topic.get("aliases", [])
        if str(value).strip()
    ]

    search_phrases = list(
        dict.fromkeys(
            [
                topic_title,
                mapped_title,
                *aliases,
                *synonyms,
            ]
        )
    )

    page_content_parts = [
        f"User topic: {topic_title}",
        f"Category: {category}",
        f"Trusted medical topic: {mapped_title}",
    ]

    if search_phrases:
        page_content_parts.append(
            "Search phrases: "
            + ", ".join(search_phrases)
        )

    if groups:
        page_content_parts.append(
            "Medical groups: "
            + ", ".join(groups)
        )

    page_content_parts.extend(
        [
            "",
            "Medical information:",
            summary,
        ]
    )

    page_content = "\n".join(
        page_content_parts
    ).strip()

    if not topic_title or not summary:
        return None

    return {
        "id": str(topic.get("id", "")),
        "title": topic_title,
        "category": category,
        "mapped_title": mapped_title,
        "page_content": page_content,
        "search_phrases": search_phrases,
        "source": source,
        "source_url": source_url,
        "medlineplus_id": str(
            topic.get("medlineplus_id", "")
        ),
        "mapping_type": str(
            topic.get("mapping_type", "")
        ),
        "mapping_score": topic.get(
            "mapping_score",
            0,
        ),
    }


def main() -> None:
    if not MAPPED_CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {MAPPED_CATALOG_FILE}. "
            "Run map_topics_to_medlineplus.py first."
        )

    catalog = load_json(
        MAPPED_CATALOG_FILE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old generated JSON files.
    for old_file in OUTPUT_DIR.glob("*.json"):
        old_file.unlink()

    created = 0
    skipped = 0

    for category in catalog.get(
        "categories",
        [],
    ):
        for topic in category.get(
            "topics",
            [],
        ):
            status = str(
                topic.get("status", "")
            )

            if not status.startswith("mapped"):
                skipped += 1
                continue

            document = create_document(topic)

            if document is None:
                skipped += 1
                continue

            filename = (
                f"{slugify(document['category'])}-"
                f"{slugify(document['title'])}.json"
                )

            output_path = OUTPUT_DIR / filename

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    document,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            created += 1

    print(f"Curated documents created: {created}")
    print(f"Topics skipped: {skipped}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()