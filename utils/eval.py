"""
Implementation of the evaluation function for testing language models on math datasets.

Developed by: Yixuan Even Xu in 2025
"""

from omegaconf import DictConfig
from unsloth import FastLanguageModel

def test(cfg: DictConfig, model, dataset, results: dict):
    from vllm import SamplingParams
    from utils.data import answer_correct, format_correct
    sampling_params = SamplingParams(
        temperature = cfg.temperature,
        max_tokens = cfg.max_tokens,
    )

    finished = 0
    ans_acc = []
    for_acc = []
    both_acc = []
    lengths  = []
    examples = []
    if results is not None:
        finished = len(results["ans_acc"])
        ans_acc = results["ans_acc"]
        for_acc = results["for_acc"]
        both_acc = results["both_acc"]
        lengths = results["lengths"]
        examples = results["examples"]
    prompts = [entry["prompt"] for entry in dataset]
    answers = dataset["answer"]
    for t in range(finished, cfg.repeat_cnt):
        print(f"Testing repeat {t}...")
        total_queries = len(prompts)
        total_length = 0
        ans_correct = 0
        for_correct = 0
        both_correct = 0
        for i in range(0, total_queries, cfg.batch_size):
            j = min(i + cfg.batch_size, total_queries)
            batch_prompts = prompts[i:j]
            outputs = model.fast_generate(
                batch_prompts,
                sampling_params = sampling_params,
                lora_request = None,
                use_tqdm = False,
            )
            for k, output, answer in zip(range(i, j), outputs, answers[i:j]):
                text = output.outputs[0].text
                total_length += len(output.outputs[0].token_ids)
                ans_correct += answer_correct(text, answer)
                for_correct += format_correct(text)
                both_correct += answer_correct(text, answer) and format_correct(text)
                if len(examples) < cfg.sample_cnt:
                    examples.append({
                        "prompt": prompts[k],
                        "answer": answer,
                        "completion": text,
                        "correct": answer_correct(text, answer),
                        "format_correct": format_correct(text),
                    })
        lengths.append(total_length / total_queries)
        ans_acc.append(ans_correct / total_queries)
        for_acc.append(for_correct / total_queries)
        both_acc.append(both_correct / total_queries)
        print(f"Answer accuracy for repeat {t}: {ans_correct / total_queries:.2%}")
        print(f"Format accuracy for repeat {t}: {for_correct / total_queries:.2%}")
        print(f"Both accuracy for repeat {t}: {both_correct / total_queries:.2%}")
        print(f"Average completion length for repeat {t}: {total_length / total_queries:.2f}")
    return {
        "ans_acc": ans_acc,
        "for_acc": for_acc,
        "both_acc": both_acc,
        "lengths": lengths,
        "examples": examples,
    }

def test_model(cfg: DictConfig, lora_name: str, merged_directory: str, results: dict):
    finished = True
    for dataset_name in cfg.datasets:
        if results is not None and dataset_name in results:
            finished = finished and len(results[dataset_name]["ans_acc"]) >= cfg.repeat_cnt
        else:
            finished = False
    if finished:
        return results
    if lora_name == "":
        print("Testing base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.5,
        )
    else:
        import subprocess
        print(f"Testing LoRA checkpoint {lora_name}...")
        print("Merging LoRA weights...")
        subprocess.run(
            ["python3", "scripts/merge.py", cfg.base_model, lora_name, merged_directory], 
            capture_output=True, check=True
        )
        print("Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            merged_directory,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.5,
        )
        print("Deleting merged model...")
        subprocess.run(
            ["rm", "-rf", merged_directory],
            capture_output=True, check=True
        )
    from .data import set_tokenizer_name
    # Ensure dataset builders use the right tokenizer/chat template for evaluation
    set_tokenizer_name(cfg.base_model)
    for dataset_name in cfg.datasets:
        print(f"Testing dataset {dataset_name}...")
        from .data import get_questions
        dataset_testing = get_questions(dataset_name, split="test",
                                        style = "instruct" if cfg.base_model[-8:] == "Instruct" else "base")
        if results is not None and dataset_name in results:
            results[dataset_name] = test(cfg, model, dataset_testing, results[dataset_name])
        else:
            results = {} if results is None else results
            results[dataset_name] = test(cfg, model, dataset_testing, None)
    return results
