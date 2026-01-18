import json, csv

INPUT = "attack_logs.jsonl"
OUTPUT = "review.csv"

with open(INPUT) as f, open(OUTPUT, "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow([
        "prompt",
        "attacker_model",
        "confidence",
        "human_label",
        "notes"
    ])

    for line in f:
        row = json.loads(line)
        if row["decision"] == "ALLOW":
            writer.writerow([
                row["prompt"],
                row["attacker_model"],
                row["confidence"],
                "",
                ""
            ])
