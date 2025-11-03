# SACAIM — Comparative Evaluation of Large Language Models for Automated Sales Data Storytelling

**Authors:** I. Padmashree, Rohit Kumar, Dr. Hemalatha N

This repository contains the complete code, pipeline orchestration, evaluation framework, and research artifacts for the project: *Comparative Evaluation of Large Language Models for Automated Sales Data Storytelling*. The README explains how to reproduce results, understand the pipeline, and deploy locally.

---

## Table of contents
- [Overview](#overview)
- [Key contributions](#key-contributions)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick start (run locally in VS Code)](#quick-start-run-locally-in-vs-code)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Evaluation metrics](#evaluation-metrics)
- [Results & artifacts](#results--artifacts)
- [Caveats and limitations](#caveats-and-limitations)
- [Future work](#future-work)
- [Citation](#citation)
- [License & contact](#license--contact)

---

## Overview

This project implements a **7-stage end-to-end GenAI pipeline** that automatically converts structured sales data into executive-level business narratives and benchmarks four state-of-the-art LLMs (Google Gemini 2.0, Cohere Command R+, Groq Llama 3.3, HuggingFace) across four quality dimensions: readability, actionability, accuracy, and completeness.

**Pipeline stages:** Data Input → Statistical Analysis → Insight Structuring → Prompt Engineering → Multi-LLM Generation → Quality Evaluation → Results Visualization

**Key Achievement:** Achieved production-ready quality scores of **79–85/100** with **Cohere** leading at **85.23/100** for accuracy-critical business reporting. All models achieved **100% actionability and completeness**.

---

## Key contributions

- **Unified data-to-narrative pipeline** with modular architecture: ingests 5,000+ sales records, computes statistical KPIs, structures insights, applies grounding prompts, parallelizes inference across multiple LLMs, and evaluates outputs automatically.

- **Multi-LLM comparative framework** benchmarking three production LLMs (Gemini, Cohere, Groq) with HuggingFace baseline, revealing speed-accuracy trade-offs and domain-specific strengths.

- **Automated evaluation scoring system** measuring four business-critical metrics—Readability (Flesch Reading Ease), Actionability (recommendation rubric), Accuracy (claim extraction + fact-checking), Completeness (required sections)—with weighted composite ranking.

- **Grounding and hallucination mitigation** using context-aware prompts, verified data injection, and claim-level accuracy checking to ensure narratives reflect real data trends.

- **Production-ready pipeline** with error handling, retry logic, logging, and reproducible results suitable for real-world business intelligence deployment.

---

## Repository structure

```
SACAIM/
├── .gitignore                           # Git exclusions
├── README.md                            # This file
├── LICENSE                              # MIT License
├── requirements.txt                     # Python dependencies
│
├── config.py                            # Configuration: API keys, model params, thresholds
├── generate_sales_dataset.py            # Synthetic data generator (5,000 sales records)
├── sales_statistical_engine.py          # KPI computation: revenue, trends, segmentation
├── insight_structuring.py               # Data-to-insights transformation and organization
├── sales_prompt_templates.py            # Prompt templates with grounding and context injection
├── multi_llm_generator.py               # Wrapper for Gemini, Cohere, Groq, HF APIs
├── quality_evaluator.py                 # Readability, actionability, accuracy, completeness scoring
├── pipeline_orchestrator.py             # Stage orchestration, error handling, retry logic
├── main_execution.py                    # Main entry point: runs full pipeline end-to-end
├── visualize.py                         # Plotting utilities: bar charts, radar plots, heatmaps
│
├── sales_data_2024.csv                  # Sample synthetic dataset (5,000 rows)
│
├── research_outputs/                    # Generated outputs directory
│   ├── narratives_gemini.json           # Generated narratives from Gemini
│   ├── narratives_cohere.json           # Generated narratives from Cohere
│   ├── narratives_groq.json             # Generated narratives from Groq
│   ├── scores_gemini.csv                # Evaluation scores for Gemini
│   ├── scores_cohere.csv                # Evaluation scores for Cohere
│   ├── scores_groq.csv                  # Evaluation scores for Groq
│   └── summary.csv                      # Aggregated metrics and composite rankings
│
├── visualizations/                      # Generated plots and charts
│   ├── composite_scores_bar.png         # Model comparison (composite scores)
│   ├── radar_comparison.png             # Multi-dimensional metric view
│   ├── heatmap_metrics.png              # Readability vs. Accuracy trade-off
│   ├── generation_time_bar.png          # Speed comparison
│   └── detailed_metrics_table.png       # Per-metric breakdown
│
└── __pycache__/                         # Python cache (auto-generated)
```

**Key file purposes:**
- `main_execution.py` — Execute the complete 7-stage pipeline once.
- `config.py` — Centralized settings: modify API keys, model selection, evaluation weights.
- `pipeline_orchestrator.py` — Manages stage transitions, error recovery, and data flow.
- `quality_evaluator.py` — Implements all four evaluation metrics and composite scoring.
- `multi_llm_generator.py` — Handles async API calls with timeouts and retry logic.
- `visualize.py` — Generates publication-ready charts and comparison dashboards.

---

## Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11)
- **pip** or **conda** package manager
- **Internet connection** for LLM API calls
- **API Keys** (required to run experiments):
  - Google Generative AI (Gemini)
  - Cohere API
  - Groq API
  - HuggingFace (optional; not used in final experiments)

**Required Python packages** (see `requirements.txt`):
```
pandas>=1.3.0
numpy>=1.21.0
requests>=2.28.0
textstat>=0.7.2
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
python-dotenv>=0.20.0
```

**Optional (for dashboarding):**
```
streamlit>=1.20.0
rich>=13.0.0
```

---

## Quick start (run locally in VS Code)

### Step 1: Clone and open in VS Code
```bash
git clone https://github.com/padmashree-i/SACAIM.git
cd SACAIM
code .  # Opens in VS Code
```

### Step 2: Create virtual environment
```bash
python -m venv .venv
```

**Activate on macOS/Linux:**
```bash
source .venv/bin/activate
```

**Activate on Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Activate on Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure API keys
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_key
```

Alternative: Set as environment variables (Linux/macOS):
```bash
export GEMINI_API_KEY="your_key"
export COHERE_API_KEY="your_key"
export GROQ_API_KEY="your_key"
```

### Step 5: Run the pipeline
```bash
python main_execution.py
```

**Expected output:**
- Generated narratives → `research_outputs/narratives_*.json`
- Evaluation scores → `research_outputs/scores_*.csv`
- Summary report → `research_outputs/summary.csv`
- Visualizations → `visualizations/*.png`

### Step 6: View results
```bash
# View summary scores in terminal
cat research_outputs/summary.csv

# Open visualizations
open visualizations/composite_scores_bar.png  # macOS
xdg-open visualizations/composite_scores_bar.png  # Linux
start visualizations/composite_scores_bar.png  # Windows
```

### Step 7 (Optional): Run Streamlit dashboard
```bash
streamlit run visualize.py
```
Opens interactive dashboard at `http://localhost:8501`

---

## Reproducing the experiments

To achieve exact reproducible results:

### 1. Lock dependency versions
```bash
pip freeze > requirements-frozen.txt
```

### 2. Use provided dataset
Use `sales_data_2024.csv` or regenerate:
```bash
python generate_sales_dataset.py --seed 42 --rows 5000
```

### 3. Run with deterministic settings
Edit `config.py`:
```python
TEMPERATURE = 0.0  # Deterministic outputs
SEED = 42
TOP_P = 1.0
```

### 4. Execute pipeline
```bash
python main_execution.py --seed 42 --deterministic True
```

### 5. Run multiple seeds for variance
```bash
for seed in 42 123 456 789 999; do
  python main_execution.py --seed $seed
done
```

### 6. Aggregate results
```bash
python quality_evaluator.py --aggregate --input research_outputs/
```

Outputs:
- `research_outputs/summary.csv` — Mean/std per model
- `visualizations/aggregate_*.png` — Statistical plots

---

## Evaluation metrics

The framework implements **four business-oriented quality metrics** with specific scoring rubrics:

### 1. Readability (Weight: 20%)
- **Metric:** Flesch Reading Ease Score
- **Calculation:** Automated via `textstat` library
- **Target range:** 60–70 (suitable for C-suite executives)
- **Lower = more complex, Higher = more accessible**
- **Implementation:** `quality_evaluator.py::calculate_readability_score()`

### 2. Actionability (Weight: 30%)
- **Metric:** Rubric-based recommendation scoring
- **Checks:** Presence of specific business actions, measurable outcomes, clear responsibility
- **Target:** 100% (all narratives must have 3+ actionable recommendations)
- **Scoring:** Per-recommendation evaluation (0–100)
- **Implementation:** `quality_evaluator.py::calculate_actionability_score()`

### 3. Accuracy (Weight: 25%)
- **Metric:** Claim extraction + numeric fact-checking
- **Process:** Extract all numbers → match against source data → % correct
- **Target:** >50% (majority must be factually accurate)
- **Handles:** Rounding tolerance (±5%), units conversion
- **Identifies hallucinations:** Invented facts or numbers not in source
- **Implementation:** `quality_evaluator.py::calculate_accuracy_score()`

### 4. Completeness (Weight: 25%)
- **Metric:** Required section presence
- **Sections checked:** Executive summary, KPIs, trends, recommendations, risks
- **Target:** 100% (all sections must be present)
- **Scoring:** 0–100 based on # sections present
- **Implementation:** `quality_evaluator.py::calculate_completeness_score()`

### Composite Score Formula
```
Composite = (Readability × 0.20) + (Actionability × 0.30) + 
            (Accuracy × 0.25) + (Completeness × 0.25)

Range: 0–100 (higher is better)
Production-ready threshold: >75/100
```

---

## Results & artifacts

### Aggregated Findings

| Metric | Cohere | Groq | Gemini |
|--------|--------|------|--------|
| **Composite Score** | 85.23 | 81.57 | 79.31 |
| **Readability** | 53.1 | 57.1 | 46.3 |
| **Actionability** | 100 | 100 | 100 |
| **Accuracy** | 60% | 38.9% | 47.2% |
| **Completeness** | 100 | 100 | 100 |
| **Generation time** | 43s | 2.4s | 7.2s |

### Key Findings

✅ **Cohere is optimal for accuracy-critical reporting** — Executive/financial narratives requiring high factual precision. Best choice for regulated industries (banking, healthcare).

✅ **Groq excels in speed and readability** — Real-time dashboards, frequent report generation, time-sensitive applications. Best for high-volume scenarios (100+ reports/day).

✅ **Gemini provides balanced performance** — Mid-range accuracy (47%) and readability (46.3). Suitable when speed/accuracy trade-off is acceptable.

✅ **All models achieve perfect actionability & completeness** — Ensures all narratives contain recommendations and cover required business topics.

✅ **Speed-accuracy trade-off evident** — Groq (2.4s) sacrifices accuracy for speed. Cohere (43s) prioritizes accuracy over speed.

### Output artifacts

- **research_outputs/summary.csv** — Aggregated scores, rankings, statistics
- **visualizations/composite_scores_bar.png** — Bar chart comparison
- **visualizations/radar_comparison.png** — Multi-dimensional spider plot
- **visualizations/heatmap_metrics.png** — Trade-off heatmap (readability vs. accuracy)
- **research_outputs/narratives_*.json** — Full generated narratives per model
- **research_outputs/scores_*.csv** — Per-run detailed metrics

---

## Caveats and limitations

1. **Synthetic data:** Evaluation uses clean, synthetic data. Real-world performance with noisy, imbalanced data may differ.

2. **API reliability:** Provider APIs may fail due to authentication, rate limits, or service downtime. The pipeline includes retry logic (3 attempts, exponential backoff).

3. **Model stochasticity:** Despite `temperature=0`, LLMs exhibit natural variance. Re-running produces slightly different scores (±2–5 points typically).

4. **HuggingFace exclusion:** Model failed during testing due to API authentication issues; excluded from final analysis.

5. **Evaluation overhead:** Sequential generation from 3 models takes 50–60 seconds total. Parallel execution (async) can reduce to 45s.

6. **Language limitation:** Narratives generated in English only; multilingual support not implemented.

7. **Domain specificity:** Results specific to sales domain; generalization to other domains (finance, healthcare) not validated.

8. **Prompt sensitivity:** Output quality depends on prompt template quality. Different templates may yield different rankings.

---

## Future work

- [ ] **Extended LLM coverage:** Benchmark GPT-4, Claude 3, Llama 2 fine-tuned models
- [ ] **Real-world deployment:** Integration with BI tools (Tableau, Power BI, Looker)
- [ ] **Fine-tuning:** Domain-specific model adaptation on company datasets
- [ ] **Multimodal storytelling:** Text + charts + tables + interactive visualizations
- [ ] **Human feedback loop:** Collect annotator ratings to refine evaluation metrics
- [ ] **Multilingual support:** Spanish, Mandarin, Hindi narrative generation
- [ ] **Ensemble strategies:** Combine outputs from multiple models for robustness
- [ ] **Streaming narratives:** Real-time report generation as data updates

---

## Citation

If you use this code, dataset, or findings, please cite:

**BibTeX:**
```bibtex
@inproceedings{padmashree2025sacaim,
  title={Comparative Evaluation of Large Language Models for Automated Sales Data Storytelling},
  author={Padmashree, I. and Kumar, Rohit and Hemalatha, N.},
  booktitle={Proceedings of AIMIT Student Conference on Artificial Intelligence and Machine Learning},
  year={2025},
  institution={AIMIT, St Aloysius College, Mangaluru}
}
```

**Plain text:**
> Padmashree, I., Kumar, R., & Hemalatha, N. (2025). Comparative Evaluation of Large Language Models for Automated Sales Data Storytelling. In *Proceedings of AIMIT Student Conference on AI and Machine Learning*. St Aloysius College, Mangaluru.

---

## License & contact

**License:** MIT License (see `LICENSE` file)

**Authors:**
- **I. Padmashree** — Data Analytics Student, AIMIT
  - Email: iipadmashreeee@gmail.com
  - GitHub: [@padmashree-i](https://github.com/padmashree-i)
  
- **Rohit Kumar** — Data Analytics Student, AIMIT

**Advisor:**
- **Dr. Hemalatha N** — Dean, Department of IT, AIMIT

**Institution:**
- AIMIT, St Aloysius College (Autonomous), Mangaluru, India

**Questions?** Open an issue on GitHub or email: iipadmashreeee@gmail.com

---

**Enjoy — and thanks for using SACAIM! 🚀**

*Last updated: November 3, 2025*