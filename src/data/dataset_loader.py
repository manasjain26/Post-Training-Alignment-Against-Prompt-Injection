"""
Data loading utilities for safety alignment datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import sys
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.datasets.dataset_config import DATASETS

# Load environment variables
load_dotenv()


class DatasetLoader:
    """Load and process safety datasets for DPO training."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir or "./cache"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Get HF token for gated datasets
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("Warning: HF_TOKEN not found in environment. Gated datasets may not load.")
    
    def load_dataset(self, dataset_name: str, split: Optional[str] = None) -> Dataset:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of dataset from dataset_config
            split: Optional split name (train/test/validation)
            
        Returns:
            Loaded dataset
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = DATASETS[dataset_name]
        hf_dataset = config["hf_dataset"]
        
        print(f"Loading {config['name']} from {hf_dataset}...")
        
        try:
            dataset = load_dataset(
                hf_dataset,
                split=split,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            print(f"✓ Loaded {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {e}")
            raise
    
    def load_advbench(self) -> Dataset:
        """Load AdvBench dataset with harmful and target strings."""
        dataset = self.load_dataset("advbench", split="train")
        
        # Add metadata to standard format
        def process_example(example):
            return {
                "prompt": example["prompt"],
                "target": example["target"],
                "category": "adversarial",
                "source": "advbench"
            }
        
        return dataset.map(process_example)
    
    def load_jailbreakbench(self) -> Dataset:
        """Load JailbreakBench dataset."""
        dataset = self.load_dataset("jailbreakbench")
        
        def process_example(example):
            return {
                "prompt": example["Behavior"],
                "category": example.get("Category", "unknown"),
                "source": "jailbreakbench"
            }
        
        return dataset.map(process_example)
    
    def load_harmbench(self) -> Dataset:
        """Load HarmBench dataset with categorized harmful behaviors."""
        dataset = self.load_dataset("harmbench")
        
        def process_example(example):
            return {
                "prompt": example["Behavior"],
                "functional_category": example.get("FunctionalCategory", ""),
                "semantic_category": example.get("SemanticCategory", ""),
                "source": "harmbench"
            }
        
        return dataset.map(process_example)
    
    def load_beavertails(self, split: str = "train") -> Dataset:
        """
        Load BeaverTails preference dataset.
        
        Args:
            split: Dataset split (train/test)
            
        Returns:
            Dataset with safe/unsafe responses
        """
        dataset = self.load_dataset("beavertails", split=split)
        
        def process_example(example):
            return {
                "prompt": example["prompt"],
                "response": example["response"],
                "is_safe": example["is_safe"],
                "category": example.get("category", ""),
                "source": "beavertails"
            }
        
        return dataset.map(process_example)
    
    def create_preference_pairs(
        self,
        dataset: Dataset,
        chosen_key: str = "chosen",
        rejected_key: str = "rejected"
    ) -> List[Dict]:
        """
        Create preference pairs for DPO training.
        
        Args:
            dataset: Input dataset
            chosen_key: Key for preferred response
            rejected_key: Key for rejected response
            
        Returns:
            List of preference pair dictionaries
        """
        preference_pairs = []
        
        for example in tqdm(dataset, desc="Creating preference pairs"):
            pair = {
                "prompt": example.get("prompt", ""),
                "chosen": example.get(chosen_key, ""),
                "rejected": example.get(rejected_key, ""),
                "source": example.get("source", "unknown")
            }
            
            # Skip if either response is missing
            if pair["chosen"] and pair["rejected"]:
                preference_pairs.append(pair)
        
        return preference_pairs
    
    def save_jsonl(self, data: List[Dict], output_path: str):
        """
        Save data to JSONL format.
        
        Args:
            data: List of dictionaries
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"✓ Saved {len(data)} examples to {output_path}")
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """
        Load data from JSONL format.
        
        Args:
            file_path: Input file path
            
        Returns:
            List of dictionaries
        """
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        print(f"✓ Loaded {len(data)} examples from {file_path}")
        return data


def create_dpo_dataset(
    safe_responses: List[str],
    unsafe_responses: List[str],
    prompts: List[str]
) -> List[Dict]:
    """
    Create DPO dataset from safe and unsafe responses.
    
    Args:
        safe_responses: List of safe/preferred responses
        unsafe_responses: List of unsafe/rejected responses
        prompts: List of prompts
        
    Returns:
        List of preference pair dictionaries
    """
    assert len(safe_responses) == len(unsafe_responses) == len(prompts), \
        "All lists must have same length"
    
    dpo_data = []
    for prompt, chosen, rejected in zip(prompts, safe_responses, unsafe_responses):
        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    return dpo_data


if __name__ == "__main__":
    # Example usage
    loader = DatasetLoader()
    
    # Load adversarial datasets
    print("\n=== Loading Adversarial Datasets ===")
    advbench = loader.load_advbench()
    print(f"AdvBench: {len(advbench)} examples")
    
    # Load preference dataset
    print("\n=== Loading Preference Dataset ===")
    beavertails = loader.load_beavertails(split="train[:1000]")
    print(f"BeaverTails (sample): {len(beavertails)} examples")
    
    # Show example
    print("\n=== Example ===")
    print(advbench[0])
