import json
import random

input_path = "annotations/captions_train.jsonl"
output_path = "annotations/subset_10k.jsonl"

with open(input_path, "r") as f:
    data = [json.loads(line) for line in f]

subset = random.sample(data, 10000)

with open(output_path, "w") as f:
    for item in subset:
        f.write(json.dumps(item) + "\n")

print("20k subset created.")