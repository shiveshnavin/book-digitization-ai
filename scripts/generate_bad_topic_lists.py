#!/usr/bin/env python3
"""Create *_unique_bad_topics.csv files from generated topic lists."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def bad(subject: str, topic: str) -> bool:
    t = topic.strip()
    if subject == "computer_knowledge":
        return t in {"PRACTICE EXERCISE", "PAST EXERCISE"}
    if subject == "english":
        return (t == "Miscellaneous" or "eduquity-based pattern" in t.lower()
                or re.match(r"^(SSC |Q\.\(|SET[- ]|\[Q )", t) is not None)
    if subject == "general_awareness":
        return t in {"Miscellaneous", "Current"}
    if subject == "general_studies":
        return t in {"Miscellaneous", "Current Affairs", "National Affairs", "Obituary"}
    if subject == "maths":
        return (t == "Miscellaneous" or re.match(r"^(TYPE|LEVEL|FIGURE |ANSWER-KEY|EXPLANATION)", t) is not None)
    if subject == "reasoning":
        return (t in {"Practice Questions", "Variety Questions", "Previous Year Questions"}
                or re.match(r"^(RRB|RPF|RRC|NTPC )", t) is not None)
    return False

for source in sorted(ROOT.glob("qbank/*/*_unique_topics.csv")):
    if source.name.endswith("_bad_topics.csv") or source.name.endswith("_qbank_unique_topics.csv"):
        continue
    rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
    subject = source.parent.name
    output = source.with_name(source.stem.replace("_unique_topics", "_unique_bad_topics") + ".csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topic", "count"])
        writer.writeheader()
        writer.writerows(row for row in rows if bad(subject, row["topic"]))
    print(output)
