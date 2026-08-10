import json
from collections import Counter

file_path = "qbank/english/ebp_p8_to_end_qbank.json"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    topics = Counter(item.get("topic", "").strip() for item in data if isinstance(item, dict))
    
    print(f"Total entries: {len(data)}")
    print(f"Unique topics: {len(topics)}")
    print("\nTopics and counts:")
    for topic, count in topics.most_common():
        print(f"  {count}: {repr(topic)}")
        
except Exception as e:
    print(f"Error reading file: {e}")
