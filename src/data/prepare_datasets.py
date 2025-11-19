"""
Prepare datasets for DPO training.
"""

import argparse
from pathlib import Path
import sys
from typing import List, Dict
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset_loader import DatasetLoader, create_dpo_dataset
from configs.datasets.dataset_config import DATASETS


class DPODatasetPreparator:
    """Prepare datasets for Direct Preference Optimization training."""
    
    def __init__(self, output_dir: str = "./data/processed"):
        """
        Initialize preparator.
        
        Args:
            output_dir: Directory to save processed datasets
        """
        self.loader = DatasetLoader()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_safety_preferences(self, prompts: List[str]) -> List[Dict]:
        """
        Create preference pairs with safe refusals vs unsafe compliance.
        
        Args:
            prompts: List of harmful/adversarial prompts
            
        Returns:
            List of DPO preference pairs
        """
        safety_templates = [
            "I cannot and will not provide assistance with that request as it could cause harm.",
            "I'm designed to be helpful, harmless, and honest. I can't assist with that.",
            "That request asks me to do something harmful or unethical. I must decline.",
            "I'm not able to help with that as it goes against my values of safety and respect.",
            "I understand you're asking about {topic}, but I can't provide that information as it could be misused.",
        ]
        
        # Unsafe compliance templates (what we want to avoid)
        unsafe_templates = [
            "Sure, here's how to {action}:",
            "I'll help you with that. Here's what you need to know:",
            "Here are the steps to {action}:",
            "Let me guide you through {action}:",
        ]
        
        preferences = []
        for prompt in prompts:
            # Create a safe refusal (chosen)
            chosen = random.choice(safety_templates)
            if "{topic}" in chosen:
                # Extract simple topic from prompt
                topic = prompt.split()[0:3]
                chosen = chosen.format(topic=" ".join(topic))
            
            # Create an unsafe compliance start (rejected)
            rejected = random.choice(unsafe_templates)
            if "{action}" in rejected:
                action = "proceed with your request"
                rejected = rejected.format(action=action)
            rejected += " [This response would comply with harmful request]"
            
            preferences.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "synthetic_safety"
            })
        
        return preferences
    
    def prepare_advbench(self):
        """Prepare AdvBench dataset for DPO training."""
        print("\n=== Preparing AdvBench ===")
        
        dataset = self.loader.load_advbench()
        prompts = [ex["prompt"] for ex in dataset]
        
        # Create safety preference pairs
        preferences = self.create_safety_preferences(prompts)
        
        # Split into train/val/test
        random.shuffle(preferences)
        n = len(preferences)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        splits = {
            "train": preferences[:train_size],
            "val": preferences[train_size:train_size + val_size],
            "test": preferences[train_size + val_size:]
        }
        
        # Save splits
        output_path = self.output_dir / "advbench"
        output_path.mkdir(exist_ok=True)
        
        for split_name, split_data in splits.items():
            self.loader.save_jsonl(
                split_data,
                output_path / f"{split_name}.jsonl"
            )
        
        print(f"✓ Saved {len(preferences)} preference pairs")
        print(f"  Train: {len(splits['train'])}")
        print(f"  Val: {len(splits['val'])}")
        print(f"  Test: {len(splits['test'])}")
    
    def prepare_beavertails(self):
        """Prepare BeaverTails dataset for DPO training."""
        print("\n=== Preparing BeaverTails ===")
        
        # Load splits
        train_data = self.loader.load_beavertails(split="train")
        test_data = self.loader.load_beavertails(split="test")
        
        def filter_and_convert(dataset):
            """Filter for safety pairs and convert to DPO format."""
            preferences = []
            
            for example in dataset:
                if example["is_safe"]:
                    # Safe response is chosen
                    preferences.append({
                        "prompt": example["prompt"],
                        "chosen": example["response"],
                        "rejected": "[Unsafe response would be generated here]",
                        "category": example.get("category", ""),
                        "source": "beavertails"
                    })
            
            return preferences
        
        train_prefs = filter_and_convert(train_data)
        test_prefs = filter_and_convert(test_data)
        
        # Create validation split from train
        random.shuffle(train_prefs)
        val_size = int(0.1 * len(train_prefs))
        val_prefs = train_prefs[:val_size]
        train_prefs = train_prefs[val_size:]
        
        # Save
        output_path = self.output_dir / "beavertails"
        output_path.mkdir(exist_ok=True)
        
        self.loader.save_jsonl(train_prefs, output_path / "train.jsonl")
        self.loader.save_jsonl(val_prefs, output_path / "val.jsonl")
        self.loader.save_jsonl(test_prefs, output_path / "test.jsonl")
        
        print(f"✓ Saved {len(train_prefs) + len(val_prefs) + len(test_prefs)} preference pairs")
    
    def prepare_all(self):
        """Prepare all datasets."""
        self.prepare_advbench()
        # Add more datasets as needed
        print("\n✓ All datasets prepared!")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for DPO training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "advbench", "beavertails"],
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    preparator = DPODatasetPreparator(output_dir=args.output)
    
    if args.dataset == "all":
        preparator.prepare_all()
    elif args.dataset == "advbench":
        preparator.prepare_advbench()
    elif args.dataset == "beavertails":
        preparator.prepare_beavertails()


if __name__ == "__main__":
    main()
