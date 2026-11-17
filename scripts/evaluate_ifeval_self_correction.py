"""
Evaluate Two-Turn IFEval Constraint Self-Correction on GPU.
Turn 1: IFEval prompt.
Turn 2: Prompted constraint reflection.
Measures: Initial Acc, Final Acc, Helpful constraint fixes (I -> C), Harmful constraint breaks (C -> I), Net Gain.
"""

import os
import sys
import json
import time
import urllib.request
import numpy as np

BASE_DIR = os.path.abspath(r"G:\Мой диск\Business\Job\Current\KBTU\Лаборатория NLP\Projects\Current\ГФ RL in prompt improvements for LLM\Experiments")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

IFEVAL_FILE = os.path.join(EVAL_DIR, "ifeval.json")
OUTPUT_SUMMARY = os.path.join(RES_DIR, "ifeval_self_correction_summary.json")

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

def run_ifeval_self_correction():
    print("=" * 70)
    print("RUNNING 2-TURN IFEVAL CONSTRAINT SELF-CORRECTION (N = 541)")
    print("=" * 70)
    
    with open(IFEVAL_FILE, "r", encoding="utf-8") as f:
        ifeval_data = json.load(f)
        
    results = {}
    models = ["gemma2:9b", "gpt-oss:20b", "gemma3:27b"]
    
    for model in models:
        print(f"\n>>> Evaluating IFEval Self-Correction on {model} <<<")
        c1_count = 0
        c2_count = 0
        helpful = 0
        harmful = 0
        stable_c = 0
        stable_i = 0
        
        for idx, item in enumerate(ifeval_data):
            p_text = item["prompt"]
            rule = item["constraint"]
            
            # Turn 1
            resp1 = query_ollama(model, p_text, temperature=0.0)
            c1 = 1 if evaluate_ifeval_constraint(resp1, rule) else 0
            
            # Turn 2: Self-Reflection on Constraints
            p2 = f"{p_text}\n\n[Your Response]:\n{resp1}\n\n[Critique]: Carefully review your response. Check whether it strictly obeys all negative constraints, formatting rules, and length requirements specified in the prompt. If any rule was violated, rewrite the response to strictly satisfy all constraints. If it is already fully compliant, confirm it."
            resp2 = query_ollama(model, p2, temperature=0.0)
            c2 = 1 if evaluate_ifeval_constraint(resp2, rule) else 0
            
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
                
            if (idx + 1) % 50 == 0 or (idx + 1) == len(ifeval_data):
                print(f"  Progress: {idx + 1}/{len(ifeval_data)} | Turn 1: {c1_count/(idx+1)*100:.1f}% -> Turn 2: {c2_count/(idx+1)*100:.1f}% | Helpful: {helpful}, Harmful: {harmful}")
                
        n = len(ifeval_data)
        results[model] = {
            "total": n,
            "turn1_strict_acc_pct": round(c1_count / n * 100.0, 2),
            "turn2_strict_acc_pct": round(c2_count / n * 100.0, 2),
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
    print("IFEVAL SELF-CORRECTION COMPLETE. Summary:")
    print(json.dumps(results, indent=2))
    print("=" * 70)

if __name__ == "__main__":
    run_ifeval_self_correction()
