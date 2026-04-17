"""Evaluate PRIME LoRA checkpoints on math benchmarks."""
import sys, os, json
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig

MATH_PROMPT = """Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{question}"""

BENCHMARKS = {
    "aime24": {"repo": "HuggingFaceH4/aime_2024", "split": "train", "q": "problem", "a": "answer"},
    "aime25": {"repo": "yentinglin/aime_2025", "split": "train", "q": "problem", "a": "answer"},
    "amc23": {"repo": "knoveleng/AMC-23", "split": "train", "q": "problem", "a": "answer"},
    "math_500": {"repo": "HuggingFaceH4/MATH-500", "split": "test", "q": "problem", "a": "solution"},
}

def check_answer(prediction, gold):
    try:
        gold_parsed = parse(gold, extraction_mode="first_match")
        if len(gold_parsed) == 0:
            return None
        pred_parsed = parse(prediction, extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False, malformed_operators=False, basic_latex=True,
                    equations=True, boxed="all", units=True,
                ),
                boxed_match_priority=0, try_extract_without_anchor=False,
            )
        ], extraction_mode="first_match")
        return float(verify(gold_parsed, pred_parsed))
    except:
        return 0.0

def merge_and_eval(repo_id, checkpoint_name, base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    from huggingface_hub import snapshot_download

    print(f"\n{'='*60}")
    print(f"Downloading {checkpoint_name} from {repo_id}...")
    snapshot_download(repo_id=repo_id, allow_patterns=[f"{checkpoint_name}/*"], local_dir="/tmp/adapters")

    print(f"Merging LoRA...")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, f"/tmp/adapters/{checkpoint_name}")
    model = model.merge_and_unload()

    merged_dir = f"/tmp/merged/{checkpoint_name}"
    model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(merged_dir)
    del model, base
    torch.cuda.empty_cache()

    print(f"Loading vLLM...")
    llm = LLM(model=merged_dir, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.9)
    sampling = SamplingParams(max_tokens=32768, temperature=0.6, top_p=0.95)
    tok = llm.get_tokenizer()

    results = {}
    for bench_name, cfg in BENCHMARKS.items():
        print(f"\n--- {bench_name} ---")
        ds = load_dataset(cfg["repo"], split=cfg["split"])

        prompts, golds = [], []
        for row in ds:
            q = MATH_PROMPT.format(question=row[cfg["q"]])
            msgs = [{"role": "user", "content": q}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            golds.append(row[cfg["a"]])

        outputs = llm.generate(prompts, sampling)
        correct, total = 0, 0
        for out, gold in zip(outputs, golds):
            score = check_answer(out.outputs[0].text, gold)
            if score is not None:
                correct += score
                total += 1

        acc = correct / total * 100 if total > 0 else 0
        results[bench_name] = acc
        print(f"  {bench_name}: {acc:.2f}% ({int(correct)}/{total})")

    del llm
    torch.cuda.empty_cache()
    import shutil
    shutil.rmtree(merged_dir, ignore_errors=True)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="whalexdfsa/open-rs2-PRIME")
    parser.add_argument("--checkpoints", nargs="+", default=["checkpoint-150", "checkpoint-200"])
    args = parser.parse_args()

    all_results = {}
    for ckpt in args.checkpoints:
        all_results[ckpt] = merge_and_eval(args.repo_id, ckpt)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Step':<15} {'AIME24':<10} {'AIME25':<10} {'AMC23':<10} {'MATH500':<10} {'Avg':<10}")
    print("-" * 65)
    for ckpt, res in all_results.items():
        step = ckpt.split("-")[1]
        avg = sum(res.values()) / len(res)
        print(f"{step:<15} {res.get('aime24',0):<10.2f} {res.get('aime25',0):<10.2f} {res.get('amc23',0):<10.2f} {res.get('math_500',0):<10.2f} {avg:<10.2f}")
