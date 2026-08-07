from __future__ import annotations

import json
import re
from pathlib import Path


INPUT_FILE = Path("data/curated/topics_master.txt")
OUTPUT_FILE = Path("data/curated/topic_catalog.json")


def clean_category_name(value: str) -> str:
    value = re.sub(r"^\d+\.\s*", "", value)
    value = re.sub(r"\s*\(\d+\+?\)\s*$", "", value)
    return value.strip()


def is_category_heading(line: str) -> bool:
    return bool(re.match(r"^\d+\.\s+\S+", line))


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    lines = INPUT_FILE.read_text(
        encoding="utf-8"
    ).splitlines()

    categories = []
    current_category = None

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        if is_category_heading(line):
            category_name = clean_category_name(line)

            current_category = {
                "id": slugify(category_name),
                "category": category_name,
                "topics": [],
            }

            categories.append(current_category)
            continue

        if current_category is None:
            print(f"Skipping line outside category: {line}")
            continue

        existing_titles = {
            topic["title"].lower()
            for topic in current_category["topics"]
        }

        if line.lower() in existing_titles:
            continue

        current_category["topics"].append(
            {
                "id": slugify(line),
                "title": line,
                "category": current_category["category"],
                "aliases": [],
                "keywords": [],
                "related_topics": [],
                "related_medlineplus_topic": "",
                "status": "unmapped",
            }
        )

    total_topics = sum(
        len(category["topics"])
        for category in categories
    )

    output = {
        "dataset_name": "MedicalBot Common Topics",
        "version": "1.0",
        "total_categories": len(categories),
        "total_topics": total_topics,
        "categories": categories,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Categories created: {len(categories)}")
    print(f"Topics created: {total_topics}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()