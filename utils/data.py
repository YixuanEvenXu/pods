"""
Implementation of data loading and processing functions for math datasets.

Developed by: Yixuan Even Xu in 2025
"""

import re
# from math_verify import parse, verify
from .grader import grade_answer
from datasets import load_dataset, Dataset, Features, Value, concatenate_datasets
from transformers import AutoTokenizer

tokenizer_name = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT_BASE = \
"""
An AI assistant is given a math problem and solves it step by step. The assistant first thinks about the reasoning process in the mind and then concludes the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., 
<think>
Reasoning
</think>
<answer>
Answer
</answer>
"""

SYSTEM_PROMPT_INSTRUCT = \
"""
You will be given a math question.
Reason about the question and correct your own mistakes if you make any.

Respond in the following format:
<think>
Reason here. Decide to answer when you are confident as your response length is limited.
</think>
<answer>
Answer here.
</answer>
"""

def extract_xml_answer(text: str) -> str:
    # Extracts the answer block from the XML format
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_last_integer(text: str) -> str:
    # Extracts the last integer from the text
    numbers = re.findall(r"\d+", text)
    if numbers:
        return numbers[-1]
    return "-1"

def answer_correct(text : str, answer: str) -> bool:
    # Uses math_verify to check if the answer is correct
    res = any(grade_answer(ans, answer) for ans in [text, extract_xml_answer(text), extract_last_integer(text)])
    return res

def format_score(text: str) -> float:
    pattern = r"<think>.*</think>[ \n]?<answer>.*</answer>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return max(0.0, 0.1 - 0.001 * (len(text)- len(match.group(0))))
    return 0.0

def format_correct(text: str) -> bool:
    # Checks if the text is in the correct format
    pattern = r"^[ \n]?<think>.*</think>[ \n]?<answer>.*</answer>[ \n]?$"
    match = re.match(pattern, text, flags=re.DOTALL)
    return match is not None

def filter_function(example, tokenizer: AutoTokenizer) -> bool:
    # Filter out examples that are too long
    prompt = example['prompt']
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')
    return tokenized_prompt['input_ids'].shape[1] <= 512

def get_math8k_questions(split = "train", style = "base") -> Dataset:
    # Loads the Math8K dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("parquet", data_files=f'datasets/math8k/{split}.parquet')['train']
    if style == "base":
        data = data.map(lambda x: {
            'prompt': SYSTEM_PROMPT_BASE + "Problem: " + x['question'] + "\nSolution: ",
            'answer': x['gt_answer']
        })
    elif style == "instruct":
        data = data.map(lambda x: {
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT_INSTRUCT},
                    {'role': 'user', 'content': x['question']}
                ],
                tokenize = False, 
                add_generation_prompt = True
            ),
            'answer': x['gt_answer']
        })
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.filter(filter_function, fn_kwargs={'tokenizer': tokenizer})
    return data

def extract_gsm8k_answer(text: str) -> str | None:
    # Extracts the answer from the GSM8K dataset
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "")

def get_gsm8k_questions(split = "train", style = "base") -> Dataset:
    # Loads the GSM8K dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset('openai/gsm8k', 'main')[split]
    if style == "base":
        data = data.map(lambda x: {
            'prompt': SYSTEM_PROMPT_BASE + "Problem: " + x['question'] + "\nSolution: ",
            'answer': extract_gsm8k_answer(x['answer'])
        })
    elif style == "instruct":
        data = data.map(lambda x: {
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT_INSTRUCT},
                    {'role': 'user', 'content': x['question']}
                ],
                tokenize = False, 
                add_generation_prompt = True
            ),
            'answer': extract_gsm8k_answer(x['answer'])
        })
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.filter(filter_function, fn_kwargs={'tokenizer': tokenizer})
    return data

def extract_math_answer(text: str) -> str:
    # Extracts the answer from the Math dataset
    if "\\boxed{" not in text:
        return None
    answer = text.split("\\boxed{")[-1]
    answer = answer.split("}")[0]
    return answer.strip()

def get_math_questions(split = "train", style = "base") -> Dataset:
    # Loads the Math dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    subsets = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 
               'number_theory', 'prealgebra', 'precalculus']
    datasets = [load_dataset('EleutherAI/hendrycks_math', s, split=split) for s in subsets]
    data = concatenate_datasets(datasets)
    if style == "base":
        data = data.map(lambda x: {
            'prompt': SYSTEM_PROMPT_BASE + "Problem: " + x['problem'] + "\nSolution: ",
            'answer': extract_math_answer(x['solution'])
        })
    elif style == "instruct":
        data = data.map(lambda x: {
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT_INSTRUCT},
                    {'role': 'user', 'content': x['problem']}
                ],
                tokenize = False, 
                add_generation_prompt = True
            ),
            'answer': extract_math_answer(x['solution'])
        })
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.filter(filter_function, fn_kwargs={'tokenizer': tokenizer}).shuffle(seed=42)
    return data

def get_math500_questions(split = "test", style = "base") -> Dataset:
    # Loads the Math500 dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset('HuggingFaceH4/MATH-500')[split]
    if style == "base":
        data = data.map(lambda x: {
            'prompt': SYSTEM_PROMPT_BASE + "Problem: " + x['problem'] + "\nSolution: ",
            'answer': x['answer']
        })
    elif style == "instruct":
        data = data.map(lambda x: {
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT_INSTRUCT},
                    {'role': 'user', 'content': x['problem']}
                ],
                tokenize = False, 
                add_generation_prompt = True
            ),
            'answer': x['answer']
        })
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.filter(filter_function, fn_kwargs={'tokenizer': tokenizer})
    return data

def get_amc23_questions(split = "test", style = "base") -> Dataset:
    # Loads the AMC23 dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset('zwhe99/amc23')[split]
    if style == "base":
        data = data.map(lambda x: {
            'prompt': SYSTEM_PROMPT_BASE + "Problem: " + x['question'] + "\nSolution: ",
            'answer': str(int(x['answer']))
        })
    elif style == "instruct":
        data = data.map(lambda x: {
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT_INSTRUCT},
                    {'role': 'user', 'content': x['question']}
                ],
                tokenize = False, 
                add_generation_prompt = True
            ),
            'answer': str(int(x['answer']))
        })
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.cast_column('answer', Value('string')).filter(filter_function, fn_kwargs={'tokenizer': tokenizer})
    return data

def get_questions(name: str, split = "train", style = "base") -> Dataset:
    # Loads the dataset based on the name provided
    if name == "math8k":
        return get_math8k_questions(split, style)
    elif name == "gsm8k":
        return get_gsm8k_questions(split, style)
    elif name == "math500":
        return get_math500_questions(split, style)
    elif name == "amc23":
        return get_amc23_questions(split, style)
    elif name == "math":
        return get_math_questions(split, style)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

# Reward functions
def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    # Correctness reward, 1.0 for correct, 0.0 for incorrect
    responses = [completion for completion in completions]
    return [1.0 if answer_correct(r, a) else 0.0 for r, a in zip(responses, answer)]

def format_reward_func(completions, answer, **kwargs) -> list[float]:
    # Format reward, 0.1 for correct format
    responses = [completion for completion in completions]
    return [0 if answer_correct(r, a) else format_score(r) for r, a in zip(responses, answer)]

def length_penalty_func(completion_mask, max_completion_length, **kwargs) -> list[float]:
    # Length penalty, 0.5 for completion length >= max_completion_length
    completion_lengths = completion_mask.sum(dim=1).tolist()
    return [-0.5 if l >= max_completion_length else 0.0 for l in completion_lengths]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.025
    if text.count("</think>") == 1:
        count += 0.025
    if text.count("<answer>") == 1:
        count += 0.025
    if text.count("</answer>") == 1:
        count += 0.025
    return count

def xmlcount_reward_func(completions, answer, **kwargs) -> list[float]:
    # XML count reward, 0.025 for each XML tag
    contents = [completion for completion in completions]
    return [0 if answer_correct(r, a) else count_xml(r) for r, a in zip(contents, answer)]