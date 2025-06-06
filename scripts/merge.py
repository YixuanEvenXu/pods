"""
To merge a LoRA adapter with a base model.
Usage: python3 scripts/merge.py <base_model_name> <lora_name> <merged_directory>

Developed by: Yixuan Even Xu in 2025
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("base_model_name", type=str)
parser.add_argument("lora_name", type=str)
parser.add_argument("merged_directory", type=str)
args = parser.parse_args()
print(f"Using lora: {args.lora_name}")

# Define base model and LoRA adapter paths
base_model_name = args.base_model_name
lora_adapter_path = args.lora_name

# Load the base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the trained LoRA adapter onto the base model and merge it
model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged_model = model_with_lora.merge_and_unload()

# Save the merged model and tokenizer
merged_model.save_pretrained(args.merged_directory)
tokenizer.save_pretrained(args.merged_directory)
