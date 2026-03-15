# NLDD Pipeline — ICLR LIT Workshop 2026

> Mechanistic interpretability pipeline for evaluating reasoning trajectories in LLMs.
> Submitted to the ICLR LIT Workshop 2026.

This repository provides tools to measure **causal faithfulness**, **representational similarity**, and **reasoning efficiency** in Chain-of-Thought (CoT) prompts using the **Normalized Logit Difference Decay (NLDD)** metric and related mechanistic analyses.

---

## Mechanistic Components

| ID | Component | Description |
|----|-----------|-------------|
| A | Counterfactual CoT | Systematic intervention & semantic corruption on reasoning steps |
| B | NLDD | Faithfulness measurement via Normalized Logit Difference Decay |
| C | Patching | Causal localization of reasoning features |
| D | Probing | Linear probing to contrast representation vs. control utilization |
| E | RSA | Representational Similarity Analysis across network layers |
| F/G | Geometry | PCA & TAS for clean-trajectory reasoning efficiency |

---

## Core Metrics

### 1. Normalized Logit Difference Decay (NLDD)
Measures causal faithfulness as the standardized margin between clean and corrupted reasoning trajectories.

- **Categorical tasks**: uses final-answer token margin
- **GSM8K**: uses answer-level sequence margin over teacher-forced numeric strings
- Low-margin rows with |LD_clean| ≤ ε are assigned NLDD = 0 and flagged in audit output

### 2. Clean-Trajectory Efficiency (TAS)

$$\text{TAS} = \frac{\text{Displacement}}{\text{Path Length}}$$

### 3. Representational Similarity (RSA)
Measures computational divergence between clean and corrupted states via the Spearman Correlation of the Representational Dissimilarity Matrix (RDM).

---

## Supported Tasks

| Dataset | Domain |
|---------|--------|
| Dyck-n | Synthetic bracket-matching logic |
| ProntoQA | Deductive logical reasoning (forced deep/long chains) |
| GSM8K | Multi-step arithmetic reasoning |
| StrategyQA | Commonsense multi-hop question answering |

---

## Supported Models

```python
# Default active model
"deepseek-ai/deepseek-coder-6.7b-instruct"

# Also configured
"meta-llama/Llama-3.1-8B-Instruct"
"google/gemma-2-9b-it"
"meta-llama/Llama-3-70B"
```

> A configured Hugging Face token is required for restricted model weights.

---

## Installation

```bash
pip install torch transformers datasets scikit-learn scipy statsmodels \
            numpy pandas matplotlib seaborn tqdm
```

### Hugging Face Authentication

The pipeline resolves your HF token automatically via:
1. Google Colab Secrets (`HF_TOKEN`)
2. Environment variables
3. Local `.env` files

If none are found, you will be prompted to authenticate interactively.

---

## Usage

Experiments are managed through the `ExperimentConfig` dataclass. Run the pipeline by executing the Jupyter Notebook sequentially:

```
1. Initialize environment and log into Hugging Face
2. Load ExperimentConfig and target model
3. Instantiate MechanisticToolkit (Probing, RSA, CKA, TAS)
4. Build task-specific datasets via Data Generators
5. Trigger NLDD Analyzer and Counterfactual Accuracy pipelines
```

Outputs (metrics, CSV audits, charts) are saved to:

```
./results_iclr_lit_2026/
```

---

## Implementation Notes

- Low-margin rows (|LD_clean| ≤ ε) → NLDD = 0, flagged in audit CSV
- GSM8K NLDD uses answer-level sequence-margin scoring with task-specific numeric distractors
- Horizon confidence intervals use raw per-step BCa intervals
- Supports prediction caching and dynamic metric scaling via S_model to prevent logit-flattening artifacts

---

## Citation

If you use this pipeline or the NLDD metric in your research, please cite the corresponding ICLR LIT Workshop 2026 paper.

```bibtex
@inproceedings{nldd2026,
  title     = {[Your Paper Title]},
  author    = {[Authors]},
  booktitle = {ICLR LIT Workshop 2026},
  year      = {2026}
}
```

