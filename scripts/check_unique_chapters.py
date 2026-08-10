#!/usr/bin/env python3
"""List unique chapter/topic names and their counts from a question bank JSON file and save to CSV."""

import argparse
import json
import csv
from collections import Counter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="List unique chapter names from a qbank JSON file and save to CSV.")
    parser.add_argument("qbank_file", type=Path, help="Path to the _qbank.json file")
    args = parser.parse_args()

    if not args.qbank_file.exists():
        print(f"Error: File '{args.qbank_file}' does not exist.")
        return

    # Determine pdfname and output path
    # Input: path/to/pdfname_qbank.json -> Output: path/to/pdfname_unique_topics.csv
    filename = args.qbank_file.name
    if filename.endswith("_qbank.json"):
        pdfname = filename[:-len("_qbank.json")]
    else:
        pdfname = args.qbank_file.stem

    output_path = args.qbank_file.parent / f"{pdfname}_unique_topics.csv"

    try:
        with open(args.qbank_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Error: '{args.qbank_file}' must contain a JSON array.")
            return

        topics = Counter(item.get("topic", "").strip() for item in data if isinstance(item, dict))

        # Write to CSV
        with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["topic", "count"])
            for topic, count in topics.most_common():
                writer.writerow([topic, count])

        print(f"Total entries: {len(data)}")
        print(f"Unique topics: {len(topics)}")
        print(f"Results saved to: {output_path}")
        print("\nTopics and counts:")
        for topic, count in topics.most_common():
            print(f"  {count}: {repr(topic)}")

    except Exception as e:
        print(f"Error reading or parsing file: {e}")

if __name__ == "__main__":
    main()
