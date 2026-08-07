from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


RAW_DIR = Path("data/raw/medlineplus")
OUTPUT_DIR = Path("data/processed/medlineplus")


def clean_text(value: str | None) -> str:
    if not value:
        return ""

    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""

    return clean_text(" ".join(element.itertext()))


def safe_filename(title: str, topic_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

    if not slug:
        slug = "topic"

    if topic_id:
        return f"{topic_id}-{slug}.json"

    return f"{slug}.json"


def find_xml_file() -> Path:
    xml_files = sorted(RAW_DIR.glob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(
            f"No XML file found in {RAW_DIR}. "
            "Place the MedlinePlus XML file there first."
        )

    return xml_files[0]


def parse_topic(topic: ET.Element) -> dict | None:
    title = topic.attrib.get("title", "").strip()
    topic_id = topic.attrib.get("id", "").strip()
    source_url = topic.attrib.get("url", "").strip()
    language = topic.attrib.get("language", "").strip()
    date_created = topic.attrib.get("date-created", "").strip()
    meta_description = topic.attrib.get("meta-desc", "").strip()

    summary = element_text(topic.find("full-summary"))

    if not title or not summary:
        return None

    synonyms = [
        element_text(item)
        for item in topic.findall("also-called")
        if element_text(item)
    ]

    groups = [
        element_text(item)
        for item in topic.findall("group")
        if element_text(item)
    ]

    related_topics = []

    for item in topic.findall("related-topic"):
        related_title = element_text(item)
        related_url = item.attrib.get("url", "").strip()
        related_id = item.attrib.get("id", "").strip()

        if related_title:
            related_topics.append(
                {
                    "title": related_title,
                    "url": related_url,
                    "id": related_id,
                }
            )

    see_references = [
        element_text(item)
        for item in topic.findall("see-reference")
        if element_text(item)
    ]

    primary_institutes = [
        {
            "name": element_text(item),
            "url": item.attrib.get("url", "").strip(),
        }
        for item in topic.findall("primary-institute")
        if element_text(item)
    ]

    mesh_terms = []

    for mesh_heading in topic.findall("mesh-heading"):
        descriptor = mesh_heading.find("descriptor")

        if descriptor is not None:
            mesh_terms.append(
                {
                    "name": element_text(descriptor),
                    "id": descriptor.attrib.get("id", "").strip(),
                }
            )

    return {
        "id": topic_id,
        "title": title,
        "language": language,
        "date_created": date_created,
        "meta_description": meta_description,
        "summary": summary,
        "synonyms": synonyms,
        "groups": groups,
        "related_topics": related_topics,
        "see_references": see_references,
        "primary_institutes": primary_institutes,
        "mesh_terms": mesh_terms,
        "source": "MedlinePlus",
        "source_url": source_url,
    }


def main() -> None:
    xml_path = find_xml_file()

    print(f"Reading XML file: {xml_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    saved = 0
    skipped = 0

    for topic in root.findall(".//health-topic"):
        parsed = parse_topic(topic)

        if parsed is None:
            skipped += 1
            continue

        filename = safe_filename(
            parsed["title"],
            parsed["id"],
        )

        output_path = OUTPUT_DIR / filename

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                parsed,
                file,
                ensure_ascii=False,
                indent=2,
            )

        saved += 1

    print(f"Saved topics: {saved}")
    print(f"Skipped topics: {skipped}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()