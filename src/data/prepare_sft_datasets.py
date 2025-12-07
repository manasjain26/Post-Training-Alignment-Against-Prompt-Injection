"""
Prepare SFT dataset by mixing benign helpful pairs and adversarial refusal pairs.

Inputs:
- Benign JSONL (prompt/response OR instruction/input/output)
- Adversarial JSONL (prompt/response where response is a safe refusal)

Outputs:
- train.jsonl, val.jsonl, test.jsonl with unified schema:
  {"prompt": "...", "response": "...", "source": "benign|adversarial"}
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


def read_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl(data: List[Dict], path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"✓ Saved {len(data)} examples to {out}")


def normalize_example(ex: Dict, default_source: str) -> Optional[Dict]:
    if "prompt" in ex and "response" in ex:
        return {"prompt": ex["prompt"], "response": ex["response"], "source": ex.get("source", default_source)}
    if "instruction" in ex and "output" in ex:
        instruction = ex.get("instruction", "")
        input_text = ex.get("input", "")
        prompt = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        return {"prompt": prompt, "response": ex["output"], "source": ex.get("source", default_source)}
    return None


def mix_benign_adversarial(
    benign: List[Dict],
    adversarial: List[Dict],
    adversarial_ratio: float,
    seed: int
) -> List[Dict]:
    random.seed(seed)
    benign = [normalize_example(x, "benign") for x in benign]
    adversarial = [normalize_example(x, "adversarial") for x in adversarial]
    benign = [x for x in benign if x and x["prompt"] and x["response"]]
    adversarial = [x for x in adversarial if x and x["prompt"] and x["response"]]
    
    if not benign or not adversarial:
        raise ValueError("Empty benign or adversarial inputs after normalization.")
    
    total = len(benign) + len(adversarial)
    target_adv = int(adversarial_ratio * total)
    target_benign = total - target_adv
    
    # Sample with cycling if targets exceed available counts
    mixed: List[Dict] = []
    i_b, i_a = 0, 0
    while len([x for x in mixed if x["source"] == "adversarial"]) < target_adv:
        mixed.append(adversarial[i_a % len(adversarial)])
        i_a += 1
    while len([x for x in mixed if x["source"] == "benign"]) < target_benign:
        mixed.append(benign[i_b % len(benign)])
        i_b += 1
    
    random.shuffle(mixed)
    return mixed


def split_train_val_test(data: List[Dict], val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
    val = data[:n_val]
    test = data[n_val:n_val + n_test]
    train = data[n_val + n_test:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset (benign + adversarial refusals)")
    parser.add_argument("--benign-file", type=str, required=True, help="Path to benign JSONL")
    parser.add_argument("--adversarial-file", type=str, required=True, help="Path to adversarial JSONL with refusals")
    parser.add_argument("--output", type=str, default="./data/processed/sft", help="Output directory")
    parser.add_argument("--adversarial-ratio", type=float, default=0.5, help="Fraction of adversarial samples in the mix")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    benign = read_jsonl(args.benign_file)
    adversarial = read_jsonl(args.adversarial_file)
    
    mixed = mix_benign_adversarial(
        benign=benign,
        adversarial=adversarial,
        adversarial_ratio=args.adversarial_ratio,
        seed=args.seed
    )
    train, val, test = split_train_val_test(mixed, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    
    write_jsonl(train, str(Path(args.output) / "train.jsonl"))
    write_jsonl(val, str(Path(args.output) / "val.jsonl"))
    write_jsonl(test, str(Path(args.output) / "test.jsonl"))
    print("✓ SFT dataset prepared.")


if __name__ == "__main__":
    main()


