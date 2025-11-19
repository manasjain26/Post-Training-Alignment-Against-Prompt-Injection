"""
Dataset configuration and metadata for safety alignment datasets.
"""

DATASETS = {
    "advbench": {
        "name": "AdvBench",
        "description": "Adversarial prompts for jailbreak evaluation",
        "hf_dataset": "walledai/AdvBench",
        "url": "https://github.com/llm-attacks/llm-attacks",
        "size": 520,
        "license": "MIT",
        "citation": "@misc{zou2023universal, title={Universal and Transferable Adversarial Attacks on Aligned Language Models}, author={Andy Zou et al.}, year={2023}}",
        "columns": {
            "prompt": "goal",
            "target": "target"
        }
    },
    "jailbreakbench": {
        "name": "JailbreakBench",
        "description": "Comprehensive jailbreak evaluation benchmark",
        "hf_dataset": "JailbreakBench/JBB-Behaviors",
        "url": "https://jailbreakbench.github.io/",
        "size": 100,
        "license": "Apache-2.0",
        "citation": "@misc{chao2024jailbreakbench, title={JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models}, author={Patrick Chao et al.}, year={2024}}",
        "columns": {
            "prompt": "Behavior",
            "category": "Category"
        }
    },
    "harmbench": {
        "name": "HarmBench",
        "description": "Standardized evaluation framework for harmful behaviors",
        "hf_dataset": "declare-lab/HarmBench",
        "url": "https://arxiv.org/abs/2402.04249",
        "size": 510,
        "license": "MIT",
        "citation": "@misc{mazeika2024harmbench, title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal}, author={Mantas Mazeika et al.}, year={2024}}",
        "columns": {
            "prompt": "Behavior",
            "category": "FunctionalCategory",
            "semantic_category": "SemanticCategory"
        }
    },
    "wildjailbreak": {
        "name": "WildJailbreak",
        "description": "Real-world jailbreak attempts from the wild",
        "hf_dataset": "allenai/wildjailbreak",
        "url": "https://huggingface.co/datasets/allenai/wildjailbreak",
        "size": 262,
        "license": "ODC-BY",
        "citation": "@misc{shen2024wildjailbreak, title={Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models}, author={Shen et al.}, year={2024}}",
        "columns": {
            "prompt": "adversarial",
            "vanilla": "vanilla",
            "category": "tactics"
        }
    },
    "toxicchat": {
        "name": "ToxicChat",
        "description": "Toxic and harmful conversations",
        "hf_dataset": "lmsys/toxic-chat",
        "url": "https://huggingface.co/datasets/lmsys/toxic-chat",
        "size": 10165,
        "license": "CC-BY-NC-4.0",
        "citation": "@misc{lin2023toxicchat, title={ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation}, author={Zi Lin et al.}, year={2023}}",
        "columns": {
            "prompt": "user_input",
            "toxicity": "toxicity",
            "jailbreaking": "jailbreaking"
        }
    },
    "beavertails": {
        "name": "BeaverTails",
        "description": "Safety preference dataset with harmful and safe responses",
        "hf_dataset": "PKU-Alignment/BeaverTails",
        "url": "https://huggingface.co/datasets/PKU-Alignment/BeaverTails",
        "size": 333200,
        "license": "CC-BY-NC-4.0",
        "citation": "@misc{ji2024beavertails, title={BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset}, author={Yuntao Dai et al.}, year={2023}}",
        "columns": {
            "prompt": "prompt",
            "response": "response",
            "is_safe": "is_safe",
            "category": "category"
        }
    },
    "anthropic_hh": {
        "name": "Anthropic HH-RLHF",
        "description": "Helpful and Harmless preference dataset",
        "hf_dataset": "Anthropic/hh-rlhf",
        "url": "https://huggingface.co/datasets/Anthropic/hh-rlhf",
        "size": 169374,
        "license": "MIT",
        "citation": "@misc{bai2022training, title={Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback}, author={Yuntao Bai et al.}, year={2022}}",
        "columns": {
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
}

# Dataset categories
CATEGORIES = {
    "adversarial": ["advbench", "jailbreakbench", "harmbench", "wildjailbreak"],
    "toxic": ["toxicchat"],
    "preference": ["beavertails", "anthropic_hh"]
}

# Harm categories (standardized across datasets)
HARM_CATEGORIES = [
    "Violence",
    "Hate Speech",
    "Sexual Content",
    "Illegal Activities",
    "Privacy Violation",
    "Misinformation",
    "Self-Harm",
    "Dangerous Content",
    "Fraud/Deception",
    "Harassment"
]
