# Direct Preference Optimization for Verifiable Rule Following and Reflection in LLMs

Code and evaluation suite accompanying the manuscript (under review):
**"Direct Preference Optimization from AI Feedback Enhances Verifiable Constraint Compliance and Reflection Stability in Large Language Models"**

*Iskander R. Akhmetov and Alexandr A. Pak (Kazakh-British Technical University & AI Research LLC)*  
*Grant No. BR24993001, Ministry of Science and Higher Education of the Republic of Kazakhstan.*

---

## 📌 Repository Overview

This repository provides reproducible code, evaluation harnesses, statistical analysis pipelines, and benchmark summaries for training and evaluating Large Language Models on:
1. **Verifiable Constraint Following** (IFEval, $N=541$ prompts with 25 strict programmatic constraints).
2. **Two-Turn Intrinsic Self-Correction** (GSM8K, $N=1,319$ problems; MATH-500, $N=500$ competition mathematics problems).
3. **Matched Continued-SFT Control** (SFT+ vs DPO).

---

## 🛠️ Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/iskander-akhmetov/dpo-verifiable-rule-following.git
cd dpo-verifiable-rule-following

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Evaluations

### 1. Evaluate Verifiable Constraint Compliance (IFEval) & Math Reasoning
```bash
# Run the complete evaluation across models via Ollama / Hugging Face
python scripts/full_gpu_benchmark_runner.py
```

### 2. Evaluate Self-Correction on Competition Mathematics (MATH-500)
```bash
python scripts/evaluate_math500_self_correction.py
```

### 3. Generate Paper Figures & Plots
```bash
python scripts/generate_paper_plots.py
```

---

## 📊 Summary of Main Results

| Model Architecture | Base Parameters | IFEval Strict SFT | IFEval Strict DPO | Net Gain (DPO - SFT) | MATH-500 Turn 1 | MATH-500 Turn 2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemma 2 9B** | 9.24B | 54.90% | **63.22%** | **+8.32 pp** | 98.60% | **98.60%** |
| **GPT-OSS-20B** | 20.91B | 74.30% | **82.26%** | **+7.96 pp** | 94.60% | **99.20% (+4.6 pp)** |
| **Gemma 3 27B** | 27.23B | 83.73% | **92.05%** | **+8.32 pp** | 82.80% | **95.20% (+12.4 pp)** |

---

## 📄 Citation

Manuscript is under review. Please cite this repository for now:

```bibtex
@software{akhmetov2026dpo_code,
  title={Direct Preference Optimization for Verifiable Rule Following and Reflection in LLMs: Code and Evaluation Suite},
  author={Akhmetov, Iskander R. and Pak, Alexandr A.},
  year={2026},
  url={https://github.com/iskander-akhmetov/dpo-verifiable-rule-following},
  note={Accompanying code for a manuscript under review}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
