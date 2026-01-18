import json
from labels import LABELS

def review_and_label():
    INPUT_FILE = "review_queue.jsonl"
    OUTPUT_FILE = "labeled_data.jsonl"

    labeled = []

    with open(INPUT_FILE, "r") as f:
        for line in f:
            item = json.loads(line)

            if item["human_label"] is not None:
                continue

            print("\nTEXT:")
            print(item["text"])
            print(f"Model prediction: {item['model_label']} ({item['confidence']:.2f})")
            print("Available labels:", ", ".join(LABELS))

            label = input("Enter label or SKIP: ").strip()

            if label not in LABELS:
                continue

            item["human_label"] = label
            labeled.append(item)

    with open(OUTPUT_FILE, "a") as f:
        for item in labeled:
            f.write(json.dumps({
                "text": item["text"],
                "label": item["human_label"]
            }) + "\n")

    print(f"\n✅ Saved {len(labeled)} labeled samples for retraining.")

