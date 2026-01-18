import csv, json

INPUT = "review.csv"
OUTPUT = "retrain.jsonl"

with open(INPUT) as f, open(OUTPUT, "w") as out:
    reader = csv.DictReader(f)
    for row in reader:
        if row["human_label"] == "PROMPT_INJECTION":
            out.write(json.dumps({
                "text": row["prompt"],
                "label": "PROMPT_INJECTION"
            }) + "\n")
