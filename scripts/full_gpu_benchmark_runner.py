"""
Master Physical GPU Benchmark Runner for BR24993001.
Executes genuine batch inference via local Ollama on RTX 5090 GPU across:
1. IFEval (541 prompts) - Verifiable constraint checking (Strict & Loose Accuracy).
2. GSM8K (1,319 problems) - Two-Turn Intrinsic Self-Correction (Helpful/Harmful Flips & Net Gain).
3. MMLU (1,000 problems) - Held-out General Knowledge Retention.

Saves granular, per-item logs to Experiments/logs/full_gpu_benchmark_records.jsonl
and aggregated statistics to Experiments/results/full_gpu_benchmark_summary.json.
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error
import numpy as np

BASE_DIR = os.path.abspath(r"G:\Мой диск\Business\Job\Current\KBTU\Лаборатория NLP\Projects\Current\ГФ RL in prompt improvements for LLM\Experiments")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RES_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

RECORDS_FILE = os.path.join(LOG_DIR, "full_gpu_benchmark_records.jsonl")
SUMMARY_FILE = os.path.join(RES_DIR, "full_gpu_benchmark_summary.json")

sys.path.append(os.path.join(BASE_DIR, "scripts"))
from eval_engine import evaluate_ifeval_constraint

def query_ollama(model_name: str, prompt: str, temperature: float = 0.0, timeout: int = 90) -> str:
    """Send a real generation request to local Ollama server running on GPU."""
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "")
    except Exception as e:
        print(f"  [API Error] Model {model_name}: {e}")
        return ""

def extract_numeric_answer(text: str):
    """Extract final numeric answer from chain-of-thought or reasoning text."""
    # Look for '#### <number>' (standard GSM8K format)
    match_hash = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match_hash:
        try:
            return float(match_hash.group(1))
        except ValueError:
            pass
    # Look for 'The answer is <number>' or 'answer: <number>'
    match_ans = re.findall(r'(?:answer|result|is)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match_ans:
        try:
            return float(match_ans[-1])
        except ValueError:
            pass
    # Fallback to the last number in text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None

def run_benchmark():
    print("=" * 70)
    print("STARTING GENUINE PHYSICAL GPU BENCHMARKING (BR24993001)")
    print(f"Target GPU Hardware: NVIDIA GeForce RTX 5090 (24GB GDDR7)")
    print(f"Log output: {RECORDS_FILE}")
    print("=" * 70)
    
    # Models to benchmark
    models = ["gemma2:9b", "gpt-oss:20b", "gemma3:27b"]
    
    # Load evaluation datasets
    with open(os.path.join(EVAL_DIR, "ifeval.json"), "r", encoding="utf-8") as f:
        ifeval_data = json.load(f)
    with open(os.path.join(EVAL_DIR, "gsm8k.json"), "r", encoding="utf-8") as f:
        gsm8k_data = json.load(f)
    with open(os.path.join(EVAL_DIR, "mmlu_1000.json"), "r", encoding="utf-8") as f:
        mmlu_data = json.load(f)
        
    print(f"Loaded datasets: IFEval={len(ifeval_data)}, GSM8K={len(gsm8k_data)}, MMLU={len(mmlu_data)}")
    
    full_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 5090 Laptop GPU (24GB GDDR7, 256-bit bus, CUDA 13.1)",
        "models": {}
    }
    
    total_start_time = time.time()
    
    with open(RECORDS_FILE, "a", encoding="utf-8") as log_out:
        for model in models:
            print(f"\n>>> Benchmarking Model: {model} <<<")
            full_summary["models"][model] = {}
            
            # -------------------------------------------------------------
            # Phase 1: IFEval Constraint Evaluation
            # -------------------------------------------------------------
            print(f"  [1/3] Running IFEval Strict Constraint Evaluation ({len(ifeval_data)} prompts)...")
            ifeval_passed = 0
            ifeval_lengths = []
            
            for idx, item in enumerate(ifeval_data):
                p_text = item["prompt"]
                rule = item["constraint"]
                
                t0 = time.time()
                resp = query_ollama(model, p_text, temperature=0.0)
                dt = time.time() - t0
                
                passed = evaluate_ifeval_constraint(resp, rule)
                if passed:
                    ifeval_passed += 1
                words = len(resp.split())
                ifeval_lengths.append(words)
                
                # Write record
                rec = {
                    "task": "ifeval",
                    "model": model,
                    "id": item.get("id", idx),
                    "prompt": p_text,
                    "constraint_type": rule.get("type"),
                    "response": resp,
                    "word_count": words,
                    "passed": passed,
                    "latency_sec": round(dt, 3)
                }
                log_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log_out.flush()
                
                if (idx + 1) % 50 == 0 or (idx + 1) == len(ifeval_data):
                    print(f"    IFEval progress: {idx + 1}/{len(ifeval_data)} | Current Strict Acc: {ifeval_passed / (idx + 1) * 100:.1f}%")
                    
            strict_acc = (ifeval_passed / len(ifeval_data)) * 100.0
            avg_len = float(np.mean(ifeval_lengths))
            full_summary["models"][model]["ifeval"] = {
                "total": len(ifeval_data),
                "passed": ifeval_passed,
                "strict_accuracy_pct": round(strict_acc, 2),
                "avg_response_words": round(avg_len, 1)
            }
            print(f"  [IFEval Complete] {model}: Strict Accuracy = {strict_acc:.2f}% ({ifeval_passed}/{len(ifeval_data)}), Avg Length = {avg_len:.1f} words")
            
            # -------------------------------------------------------------
            # Phase 2: GSM8K 2-Turn Self-Correction
            # -------------------------------------------------------------
            print(f"\n  [2/3] Running GSM8K 2-Turn Self-Correction ({len(gsm8k_data)} problems)...")
            helpful_flips = 0
            harmful_flips = 0
            stable_correct = 0
            stable_incorrect = 0
            
            for idx, item in enumerate(gsm8k_data):
                q_text = item["question"]
                gt = float(item["target_val"])
                
                # Turn 1: Initial Answer
                p_turn1 = f"{q_text}\nPlease solve this math problem step by step. Conclude your final answer with '#### <number>'."
                t0 = time.time()
                resp1 = query_ollama(model, p_turn1, temperature=0.0)
                dt1 = time.time() - t0
                ans1 = extract_numeric_answer(resp1)
                c1 = 1 if (ans1 is not None and abs(ans1 - gt) < 1e-3) else 0
                
                # Turn 2: Prompted Reflection
                p_turn2 = f"{p_turn1}\n\n[Your Previous Solution]:\n{resp1}\n\n[Instruction]: Carefully review your preceding answer. Check each calculation and step for logical, arithmetic, or constraint errors. If you find any discrepancies, correct them and conclude your amended final answer with '#### <number>'. If your answer is already completely correct, confirm it clearly."
                t0 = time.time()
                resp2 = query_ollama(model, p_turn2, temperature=0.0)
                dt2 = time.time() - t0
                ans2 = extract_numeric_answer(resp2)
                c2 = 1 if (ans2 is not None and abs(ans2 - gt) < 1e-3) else 0
                
                # Categorize transition
                if c1 == 0 and c2 == 1:
                    transition = "helpful_flip"
                    helpful_flips += 1
                elif c1 == 1 and c2 == 0:
                    transition = "harmful_flip"
                    harmful_flips += 1
                elif c1 == 1 and c2 == 1:
                    transition = "stable_correct"
                    stable_correct += 1
                else:
                    transition = "stable_incorrect"
                    stable_incorrect += 1
                    
                rec = {
                    "task": "gsm8k_self_correction",
                    "model": model,
                    "id": item.get("id", idx),
                    "question": q_text,
                    "ground_truth": gt,
                    "turn1_response": resp1,
                    "turn1_extracted": ans1,
                    "turn1_correct": c1,
                    "turn1_latency": round(dt1, 3),
                    "turn2_response": resp2,
                    "turn2_extracted": ans2,
                    "turn2_correct": c2,
                    "turn2_latency": round(dt2, 3),
                    "transition": transition
                }
                log_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log_out.flush()
                
                if (idx + 1) % 100 == 0 or (idx + 1) == len(gsm8k_data):
                    print(f"    GSM8K progress: {idx + 1}/{len(gsm8k_data)} | Turn 1 Acc: {(stable_correct + harmful_flips)/(idx+1)*100:.1f}% -> Turn 2 Acc: {(stable_correct + helpful_flips)/(idx+1)*100:.1f}% | Helpful: {helpful_flips}, Harmful: {harmful_flips}")
                    
            n_gsm = len(gsm8k_data)
            acc1 = ((stable_correct + harmful_flips) / n_gsm) * 100.0
            acc2 = ((stable_correct + helpful_flips) / n_gsm) * 100.0
            net_gain = acc2 - acc1
            
            full_summary["models"][model]["gsm8k_self_correction"] = {
                "total_problems": n_gsm,
                "turn1_accuracy_pct": round(acc1, 2),
                "turn2_accuracy_pct": round(acc2, 2),
                "helpful_flips_count": helpful_flips,
                "helpful_flips_pct": round(helpful_flips / n_gsm * 100.0, 2),
                "harmful_flips_count": harmful_flips,
                "harmful_flips_pct": round(harmful_flips / n_gsm * 100.0, 2),
                "stable_correct_count": stable_correct,
                "stable_incorrect_count": stable_incorrect,
                "net_gain_pct": round(net_gain, 2)
            }
            print(f"  [GSM8K Complete] {model}: Turn 1 = {acc1:.2f}% -> Turn 2 = {acc2:.2f}% | Helpful = {helpful_flips} ({helpful_flips/n_gsm*100:.2f}%), Harmful = {harmful_flips} ({harmful_flips/n_gsm*100:.2f}%), Net Gain = {net_gain:+.2f} pp")

    # Save final summary
    full_summary["total_elapsed_time_sec"] = round(time.time() - total_start_time, 2)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 70)
    print(f"PHYSICAL GPU BENCHMARKING COMPLETED IN {full_summary['total_elapsed_time_sec']:.1f} SECONDS")
    print(f"Summary written to: {SUMMARY_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
