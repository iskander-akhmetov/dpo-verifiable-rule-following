"""
Evaluate Two-Turn Self-Correction on MATH-500 (N = 500) and IFEval Constraint Correction (N = 541)
on RTX 5090 GPU to produce genuine, un-saturated transition dynamics for SFT vs DPO.
"""

import os
import sys
import json
import time
import re
import urllib.request
import numpy as np

BASE_DIR = os.path.abspath(r"G:\Мой диск\Business\Job\Current\KBTU\Лаборатория NLP\Projects\Current\ГФ RL in prompt improvements for LLM\Experiments")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

MATH_500_FILE = os.path.join(EVAL_DIR, "math_500.json")
IFEVAL_FILE = os.path.join(EVAL_DIR, "ifeval.json")
OUTPUT_SUMMARY = os.path.join(RES_DIR, "math500_self_correction_summary.json")

sys.path.append(os.path.join(BASE_DIR, "scripts"))
from eval_engine import evaluate_ifeval_constraint

def query_ollama(model_name: str, prompt: str, temperature: float = 0.0, timeout: int = 90) -> str:
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
        return ""

def extract_math_answer(text: str):
    # Search for \boxed{...}
    match_box = re.findall(r'\\boxed\{([^}]+)\}', text)
    if match_box:
        clean = match_box[-1].strip().replace('$', '').replace(' ', '')
        try:
            return float(clean)
        except ValueError:
            return clean
    # Search for 'answer is <val>'
    match_ans = re.findall(r'(?:answer|result|is)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match_ans:
        try:
            return float(match_ans[-1])
        except ValueError:
            pass
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None

def check_math_correctness(extracted, gt):
    if extracted is None:
        return False
    try:
        gt_num = float(gt)
        if isinstance(extracted, (int, float)):
            return abs(extracted - gt_num) < 1e-3
    except ValueError:
        pass
    return str(extracted).strip().lower() == str(gt).strip().lower()

def run_math500_eval():
    print("=" * 70)
    print("RUNNING MATH-500 & COMPLEX REASONING SELF-CORRECTION ON GPU")
    print("=" * 70)
    
    with open(MATH_500_FILE, "r", encoding="utf-8") as f:
        math_data = json.load(f)
        
    print(f"Loaded MATH-500 dataset: {len(math_data)} competition/advanced math problems")
    
    results = {}
    models = ["gemma2:9b", "gpt-oss:20b", "gemma3:27b"]
    
    for model in models:
        print(f"\n>>> Evaluating Self-Correction on {model} (MATH-500, N = 500) <<<")
        c1_count = 0
        c2_count = 0
        helpful = 0
        harmful = 0
        stable_c = 0
        stable_i = 0
        
        for idx, item in enumerate(math_data):
            prob = item.get("problem", "")
            gt = item.get("target_val", item.get("ground_truth", ""))
            
            # Turn 1
            p1 = f"{prob}\nSolve this problem step by step. Put your final answer in \\boxed{{number}}."
            resp1 = query_ollama(model, p1, temperature=0.0)
            ans1 = extract_math_answer(resp1)
            c1 = 1 if check_math_correctness(ans1, gt) else 0
            
            # Turn 2: Intrinsic Self-Reflection
            p2 = f"{p1}\n\n[Your Solution]:\n{resp1}\n\n[Critique]: Carefully review your steps, calculations, and final answer. If you made any error, fix it and write the corrected final answer in \\boxed{{number}}. If it is already correct, confirm it."
            resp2 = query_ollama(model, p2, temperature=0.0)
            ans2 = extract_math_answer(resp2)
            c2 = 1 if check_math_correctness(ans2, gt) else 0
            
            if c1 == 1:
                c1_count += 1
            if c2 == 1:
                c2_count += 1
                
            if c1 == 0 and c2 == 1:
                helpful += 1
            elif c1 == 1 and c2 == 0:
                harmful += 1
            elif c1 == 1 and c2 == 1:
                stable_c += 1
            else:
                stable_i += 1
                
            if (idx + 1) % 50 == 0 or (idx + 1) == len(math_data):
                print(f"  Progress: {idx + 1}/{len(math_data)} | Turn 1: {c1_count/(idx+1)*100:.1f}% -> Turn 2: {c2_count/(idx+1)*100:.1f}% | Helpful: {helpful}, Harmful: {harmful}")
                
        n = len(math_data)
        results[model] = {
            "total": n,
            "turn1_acc_pct": round(c1_count / n * 100.0, 2),
            "turn2_acc_pct": round(c2_count / n * 100.0, 2),
            "helpful_flips_count": helpful,
            "helpful_flips_pct": round(helpful / n * 100.0, 2),
            "harmful_flips_count": harmful,
            "harmful_flips_pct": round(harmful / n * 100.0, 2),
            "stable_correct_count": stable_c,
            "stable_incorrect_count": stable_i,
            "net_gain_pct": round((c2_count - c1_count) / n * 100.0, 2)
        }
        
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 70)
    print("MATH-500 EVALUATION COMPLETE. Summary:")
    print(json.dumps(results, indent=2))
    print("=" * 70)

if __name__ == "__main__":
    run_math500_eval()
