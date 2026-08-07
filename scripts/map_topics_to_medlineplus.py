from __future__ import annotations

import difflib
import json
import re
from pathlib import Path


CATALOG_FILE = Path("data/curated/topic_catalog.json")
MEDLINEPLUS_DIR = Path("data/processed/medlineplus")
OUTPUT_FILE = Path("data/curated/topic_catalog_mapped.json")
REPORT_FILE = Path("data/curated/topic_mapping_report.json")

MIN_FUZZY_SCORE = 0.90



# Manual corrections for common everyday phrases.
MANUAL_MAPPINGS = {
    # Fever
    "high fever": "Fever",
    "low-grade fever": "Fever",
    "chills": "Fever",

    # Respiratory
    "productive cough": "Cough",
    "dry cough": "Cough",
    "persistent cough": "Cough",
    "blocked nose": "Nose Injuries and Disorders",

    # Musculoskeletal
    "pain between shoulder blades": "Back Pain",
    "upper back pain": "Back Pain",
    "lower back pain": "Back Pain",
    "neck pain": "Neck Injuries and Disorders",
    "shoulder pain": "Shoulder Injuries and Disorders",
    "poor posture": "Back Pain",
    "work-related muscle pain": "Muscle Disorders",
    "heavy lifting injuries": "Back Injuries",
    "muscle pain": "Muscle Disorders",
    "muscle stiffness": "Muscle Disorders",
    "body aches": "Muscle Aches",
    "joint pain": "Joint Disorders",
    "heel pain": "Heel Injuries and Disorders",
    "ankle pain": "Ankle Injuries and Disorders",

    # General symptoms
    "weakness": "Fatigue",
    "confusion": "Delirium",
    "weight loss": "Weight Control",
    "weight gain": "Weight Control",

    # Men's health
    "penis pain": "Penis Disorders",
    "penis swelling": "Penis Disorders",
    "penis redness": "Penis Disorders",
    "penis itching": "Penis Disorders",
    "penis discharge": "Sexually Transmitted Infections",
    "weak erection": "Erectile Dysfunction",
    "morning erections": "Erectile Dysfunction",
    "pain during erection": "Penis Disorders",
    "pain after ejaculation": "Male Reproductive System",
    "low libido": "Sexual Problems",
    "high libido": "Sexual Health",
    "low sexual desire": "Sexual Problems",
    "loss of libido": "Sexual Problems",
    "no desire for sex": "Sexual Problems",
    "testicular pain": "Testicular Disorders",
    "prostatitis": "Prostate Diseases",
    "varicocele": "Testicular Disorders",
    "hydrocele": "Testicular Disorders",

    # Women's health
    "menstrual pain": "Menstruation",
    "heavy periods": "Menstruation",
    "light periods": "Menstruation",
    "irregular periods": "Menstruation",
    "missed period": "Menstruation",
    "late period": "Menstruation",
    "vaginal discharge": "Vaginal Diseases",
    "vaginal itching": "Vaginal Diseases",
    "vaginal odor": "Vaginal Diseases",
    "pain during sex": "Sexual Problems in Women",
    "pregnancy symptoms": "Pregnancy",
    "morning sickness": "Morning Sickness",
    "ovarian cyst": "Ovarian Cysts",
    "breast pain": "Breast Diseases",

    # Sexual relationships
    "different sex drives": "Sexual Health",
    "partner wants more sex": "Sexual Health",
    "how often is sex normal?": "Sexual Health",
    "is sex every day normal?": "Sexual Health",
    "is sex once a week normal?": "Sexual Health",
    "is sex three times a week normal?": "Sexual Health",
    "stress and libido": "Sexual Problems",
    "depression and libido": "Sexual Problems",
    "anxiety and sex": "Sexual Health",
    "pornography and sexual function": "Sexual Health",
    "relationship intimacy": "Sexual Health",
    "communication with partner": "Sexual Health",
    "masturbation": "Sexual Health",
    "can stress reduce sexual desire?": "Sexual Problems",
    "why don't i want sex anymore?": "Sexual Problems",
    "can i have sex during pregnancy?": "Sexual Health",
    "can i have sex during my period?": "Sexual Health",

    # Urinary
    "painful urination": "Urination - Painful",
    "burning urination": "Urination - Painful",
    "frequent urination": "Urination and Urination Problems",
    "urgent urination": "Urination and Urination Problems",
    "blood in urine": "Urine and Urination",

    # Medicines
    "painkillers": "Pain Relievers",
    "paracetamol": "Pain Relievers",
    "ibuprofen": "Pain Relievers",
    "missed dose": "Medicines",
    "medicine safety": "Medicines",

    # Lifestyle
    "gym injuries": "Sports Injuries",
    "hydration": "Dehydration",
    "protein": "Dietary Proteins",

    # FAQ
    "can stress cause fever?": "Fever",
    "why do i have back pain after work?": "Back Pain",
    "can lifting heavy objects hurt my back?": "Back Injuries",

    # STI
    "sti prevention": "Sexually Transmitted Infections",
    "emergency contraception": "Birth Control",

    # Hormones
    "adrenal disorders": "Adrenal Gland Disorders",
    "vaccinations": "Vaccines",

    # Eyes
    "dry eyes": "Tears",
}


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9\s'-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_catalog() -> dict:
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Missing catalog: {CATALOG_FILE}"
        )

    with CATALOG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_medlineplus_topics() -> list[dict]:
    if not MEDLINEPLUS_DIR.exists():
        raise FileNotFoundError(
            f"Missing directory: {MEDLINEPLUS_DIR}"
        )

    topics = []

    for path in sorted(MEDLINEPLUS_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            item = json.load(file)

        if item.get("language") != "English":
            continue

        title = str(item.get("title", "")).strip()

        if not title:
            continue

        search_names = {title}

        for synonym in item.get("synonyms", []):
            if synonym:
                search_names.add(str(synonym).strip())

        for reference in item.get("see_references", []):
            if reference:
                search_names.add(str(reference).strip())

        topics.append(
            {
                "id": str(item.get("id", "")),
                "title": title,
                "source": item.get("source", "MedlinePlus"),
                "source_url": item.get("source_url", ""),
                "summary": item.get("summary", ""),
                "search_names": sorted(search_names),
                "file": str(path),
            }
        )

    return topics


def build_lookup(topics: list[dict]) -> dict[str, dict]:
    lookup = {}

    for topic in topics:
        for name in topic["search_names"]:
            normalized = normalize(name)

            if normalized and normalized not in lookup:
                lookup[normalized] = topic

    return lookup


def find_topic_by_title(
    requested_title: str,
    topics: list[dict],
) -> dict | None:
    normalized_requested = normalize(requested_title)

    for topic in topics:
        if normalize(topic["title"]) == normalized_requested:
            return topic

    return None


def fuzzy_match(
    query: str,
    lookup: dict[str, dict],
) -> tuple[dict | None, float, str]:
    normalized_query = normalize(query)

    if not normalized_query:
        return None, 0.0, ""

    candidates = list(lookup.keys())

    matches = difflib.get_close_matches(
        normalized_query,
        candidates,
        n=1,
        cutoff=MIN_FUZZY_SCORE,
    )

    if not matches:
        return None, 0.0, ""

    matched_name = matches[0]

    score = difflib.SequenceMatcher(
        None,
        normalized_query,
        matched_name,
    ).ratio()

    return lookup[matched_name], score, matched_name


def map_single_topic(
    topic_title: str,
    medline_topics: list[dict],
    lookup: dict[str, dict],
) -> dict:
    normalized_title = normalize(topic_title)

    manual_target = MANUAL_MAPPINGS.get(normalized_title)

    if manual_target:
        matched = find_topic_by_title(
            manual_target,
            medline_topics,
        )

        if matched:
            return {
                "status": "mapped_manual",
                "match_type": "manual",
                "score": 1.0,
                "matched_name": manual_target,
                "medlineplus": matched,
            }

    if normalized_title in lookup:
        return {
            "status": "mapped_exact",
            "match_type": "exact",
            "score": 1.0,
            "matched_name": normalized_title,
            "medlineplus": lookup[normalized_title],
        }

    matched, score, matched_name = fuzzy_match(
        topic_title,
        lookup,
    )

    if matched:
        return {
            "status": "mapped_fuzzy",
            "match_type": "fuzzy",
            "score": round(score, 3),
            "matched_name": matched_name,
            "medlineplus": matched,
        }

    return {
        "status": "unmapped",
        "match_type": "none",
        "score": 0.0,
        "matched_name": "",
        "medlineplus": None,
    }


def main() -> None:
    catalog = load_catalog()
    medline_topics = load_medlineplus_topics()
    lookup = build_lookup(medline_topics)

    print(f"MedlinePlus English topics: {len(medline_topics)}")
    print(f"Search names available: {len(lookup)}")

    report = {
        "mapped_manual": [],
        "mapped_exact": [],
        "mapped_fuzzy": [],
        "unmapped": [],
    }

    mapped_count = 0
    fuzzy_count = 0
    unmapped_count = 0

    for category in catalog.get("categories", []):
        for topic in category.get("topics", []):
            result = map_single_topic(
                topic["title"],
                medline_topics,
                lookup,
            )

            matched = result["medlineplus"]

            topic["status"] = result["status"]
            topic["mapping_type"] = result["match_type"]
            topic["mapping_score"] = result["score"]

            if matched:
                topic["related_medlineplus_topic"] = matched["title"]
                topic["medlineplus_id"] = matched["id"]
                topic["source"] = matched["source"]
                topic["source_url"] = matched["source_url"]
                topic["medlineplus_file"] = matched["file"]

                mapped_count += 1

                if result["status"] == "mapped_fuzzy":
                    fuzzy_count += 1
            else:
                topic["related_medlineplus_topic"] = ""
                topic["medlineplus_id"] = ""
                topic["source"] = ""
                topic["source_url"] = ""
                topic["medlineplus_file"] = ""

                unmapped_count += 1

            report[result["status"]].append(
                {
                    "topic": topic["title"],
                    "category": topic["category"],
                    "matched_topic": (
                        matched["title"] if matched else ""
                    ),
                    "matched_name": result["matched_name"],
                    "score": result["score"],
                    "source_url": (
                        matched["source_url"] if matched else ""
                    ),
                }
            )

    catalog["mapping_summary"] = {
        "medlineplus_topics_available": len(medline_topics),
        "mapped_topics": mapped_count,
        "fuzzy_mappings": fuzzy_count,
        "unmapped_topics": unmapped_count,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            catalog,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Total catalog topics: {catalog['total_topics']}")
    print(f"Mapped topics: {mapped_count}")
    print(f"Fuzzy mappings requiring review: {fuzzy_count}")
    print(f"Unmapped topics: {unmapped_count}")
    print(f"Mapped catalog: {OUTPUT_FILE}")
    print(f"Mapping report: {REPORT_FILE}")


if __name__ == "__main__":
    main()

