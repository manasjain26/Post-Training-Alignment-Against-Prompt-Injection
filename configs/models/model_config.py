"""
Model configuration for small language models suitable for DPO training.
"""

MODELS = {
    "llama-3.2-1b": {
        "name": "Llama 3.2 1B",
        "hf_model": "meta-llama/Llama-3.2-1B-Instruct",
        "parameters": "1.2B",
        "context_length": 8192,
        "architecture": "Llama",
        "provider": "Meta",
        "license": "Llama 3.2 Community License",
        "features": ["Instruction-tuned", "Fast inference", "Good multilingual support"],
        "recommended_batch_size": 8,
        "supports_flash_attention": True,
        "quantization": {
            "4bit": True,
            "8bit": True
        }
    },
    "phi-3-mini": {
        "name": "Phi-3 Mini",
        "hf_model": "microsoft/Phi-3-mini-4k-instruct",
        "parameters": "3.8B",
        "context_length": 4096,
        "architecture": "Phi",
        "provider": "Microsoft",
        "license": "MIT",
        "features": ["Excellent reasoning", "Small footprint", "Safety-aligned"],
        "recommended_batch_size": 4,
        "supports_flash_attention": True,
        "quantization": {
            "4bit": True,
            "8bit": True
        }
    },
    "gemma-2b": {
        "name": "Gemma 2B",
        "hf_model": "google/gemma-2b-it",
        "parameters": "2B",
        "context_length": 8192,
        "architecture": "Gemma",
        "provider": "Google",
        "license": "Gemma License",
        "features": ["Lightweight", "Good performance", "Pre-aligned"],
        "recommended_batch_size": 8,
        "supports_flash_attention": True,
        "quantization": {
            "4bit": True,
            "8bit": True
        }
    },
    "qwen-1.5b": {
        "name": "Qwen 2 1.5B",
        "hf_model": "Qwen/Qwen2-1.5B-Instruct",
        "parameters": "1.5B",
        "context_length": 32768,
        "architecture": "Qwen",
        "provider": "Alibaba",
        "license": "Apache-2.0",
        "features": ["Long context", "Multilingual", "Strong coding"],
        "recommended_batch_size": 8,
        "supports_flash_attention": True,
        "quantization": {
            "4bit": True,
            "8bit": True
        }
    },
    "stablelm-2-1.6b": {
        "name": "StableLM 2 1.6B",
        "hf_model": "stabilityai/stablelm-2-1_6b-chat",
        "parameters": "1.6B",
        "context_length": 4096,
        "architecture": "StableLM",
        "provider": "Stability AI",
        "license": "Apache-2.0",
        "features": ["Open source", "Fast training", "Good instruction following"],
        "recommended_batch_size": 8,
        "supports_flash_attention": False,
        "quantization": {
            "4bit": True,
            "8bit": True
        }
    },
    "mistral-7b": {
        "name": "Mistral 7B",
        "hf_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "parameters": "7B",
        "context_length": 8192,
        "architecture": "Mistral",
        "provider": "Mistral AI",
        "license": "Apache-2.0",
        "features": ["State-of-art performance", "Sliding window attention", "Efficient"],
        "recommended_batch_size": 2,
        "supports_flash_attention": True,
        "quantization": {
            "4bit": True,
            "8bit": True
        }
    }
}

# Recommended models for the project (1B-3B range)
RECOMMENDED_MODELS = [
    "llama-3.2-1b",
    "phi-3-mini",
    "gemma-2b",
    "qwen-1.5b",
    "stablelm-2-1.6b"
]

# Model selection criteria
SELECTION_CRITERIA = {
    "parameter_range": "1B-3B",
    "must_have": [
        "Instruction-tuned baseline",
        "Publicly available",
        "Supports modern training frameworks"
    ],
    "nice_to_have": [
        "Flash attention support",
        "4-bit quantization support",
        "Pre-existing safety alignment"
    ]
}
