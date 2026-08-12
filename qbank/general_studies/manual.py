"""Manual post-processing step for QBank questions."""

import re
from typing import Any, Dict, List

OPTION_MAP = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "e": "E",
}

def normalize_correct_option(val: Any) -> str:
    if val is None:
        return ""
    val_str = str(val).strip()
    if not val_str:
        return val_str

    if val_str in OPTION_MAP:
        return OPTION_MAP[val_str]

    cleaned = re.sub(r"[()\.Option\s]", "", val_str, flags=re.IGNORECASE)
    if cleaned in OPTION_MAP:
        return OPTION_MAP[cleaned]

    if len(val_str) == 1 and val_str.isalpha():
        return val_str.upper()

    return val_str

def normalize_topic(topic: str) -> str:
    if not topic or not isinstance(topic, str):
        return topic

    topic_str = topic.replace("\n", " ").strip()
    topic_str = re.sub(r"\s+", " ", topic_str)
    topic_str = re.sub(r"\bBooks\s+and\s+Authors\b", "Books And Authors", topic_str, flags=re.IGNORECASE)
    topic_str = topic_str.title()
    return topic_str

def manual_process(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_topics = set()
    normalized_options_count = 0
    synced_text_count = 0

    for item in items:
        item["tenant"] = "comebackio"
        item["exam"] = "ssc"
        item["subject"] = "general_studies" 
        if "topic" in item and item["topic"]:
            item["topic"] = normalize_topic(str(item["topic"]))
            unique_topics.add(item["topic"])

        if "correct_option" in item and item["correct_option"] is not None:
            old_opt = item["correct_option"]
            new_opt = normalize_correct_option(old_opt)
            if old_opt != new_opt:
                item["correct_option"] = new_opt
                normalized_options_count += 1

            opt_key = f"option{new_opt}"
            if opt_key in item and item[opt_key]:
                if item.get("correct_option_text") != item[opt_key]:
                    item["correct_option_text"] = item[opt_key]
                    synced_text_count += 1

    print(f"Normalized correct_option in {normalized_options_count} questions.")
    print(f"Updated correct_option_text from correct option in {synced_text_count} questions.")
    print(f"Unique Topics ({len(unique_topics)}):")
    for t in sorted(unique_topics):
        print(f"  - {t}")

    return items
