"""
To compute the prompt length distribution of the GSM8K dataset.

Developed by: Yixuan Even Xu in 2025
"""

from utils.data import get_gsm8k_questions

if __name__ == "__main__":
    dataset = get_gsm8k_questions(split = "train")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    prompt_lengths = []
    for example in dataset:
        prompt = example['prompt']
        tokenized_prompt = tokenizer.apply_chat_template(prompt, tokenize = True, add_generation_prompt = True)
        prompt_lengths.append(len(tokenized_prompt))
    
    import matplotlib.pyplot as plt
    plt.hist(prompt_lengths, bins=20)
    plt.xlabel("Prompt length")
    plt.ylabel("Frequency")
    plt.title("Prompt length distribution")
    plt.savefig("prompt_length_distribution.png")