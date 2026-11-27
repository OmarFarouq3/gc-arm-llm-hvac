# GC-ARM + Knowledge-Enhanced LLM for HVAC Energy-Efficiency Diagnosis

Course project (Data Mining, Part III): implementation, evaluation and extension of

> Q. Deng, Z. Chen, Y. Liu, Z. Zhu, J. Xu, Z. Jiang, K.J. Chua,
> *Graph-constrained association rule mining-based energy efficiency diagnosis method for HVAC systems with knowledge-enhanced large language model*,
> Building and Environment 289 (2026) 114056.

## Pipeline

```
sensor data ──► preprocessing ──► discretisation ──► GC-ARM ──► candidate rules ──► LLM + RAG ──► label + reason
                                                       ▲                                 ▲
                                         association graph (k-hop)               knowledge base
```

1. **Preprocessing** (`src/preprocess.py`): 3-sigma outlier removal, frozen-sensor removal, derived metrics (COP, ...).
2. **Association graph** (`src/graph.py`): nodes = sensors / derived metrics / environment-control variables; edges = physical connection, thermodynamic coupling, control dependency.
3. **Discretisation** (`src/discretise.py`): design breakpoints + entropy-based splits.
4. **GC-ARM** (`src/gcarm.py`): Apriori with k-hop pruning during candidate generation (paper Algorithm 1, Eq. 13-14).
5. **Knowledge-enhanced LLM** (`src/rag.py`, `src/llm_diagnose.py`): retrieve domain knowledge, classify each rule as
   0 = invalid/normal, 1 = design anomaly, 2 = operational anomaly, 3 = performance degradation.
6. **Evaluation** (`src/evaluate.py`): accuracy, FAR, MAR (paper definitions) + macro-F1 and per-class results.
7. **Our improvement** (`src/improvement.py`): confounder-aware rule screening + self-consistency voting.

## Dataset

LBNL Fault Detection and Diagnostics Data Sets, **chiller plant** (https://faultdetection.lbl.gov/data).
Granderson et al., *A labeled dataset for building HVAC systems operating in faulted and fault-free states*, Scientific Data 10 (2023) 342.

Raw data is **not committed** (too large). Download it into `data/raw/` (see `data/README.md`).

## Repository layout

```
data/raw/            downloaded dataset (git-ignored)
data/processed/      cleaned + discretised data (git-ignored)
knowledge_base/      documents used for RAG
notebooks/           exploration and experiments
src/                 pipeline code
results/figures/     plots for the report
results/tables/      CSV/LaTeX tables for the report
report/              conference-paper report
config.yaml          all parameters in one place
TASKS.md             daily checklist
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python src/gcarm.py          # Apriori demo on toy data
python tests/test_gcarm.py   # tests for the mining code
python tests/test_baseline_threshold.py   # threshold baseline (paper Table 3)
python tests/test_improvement.py          # confounder check + majority vote
python -m src.baseline_threshold          # baseline on data/processed/*.parquet -> results/tables/baseline_threshold_rows.csv
```

## LLM / RAG

The rules are judged by a local LLM through Ollama's OpenAI-compatible API, with context
retrieved from `knowledge_base/` (general HVAC knowledge + plant description).

1. Install Ollama and pull the model: `ollama pull qwen3.5:4b` (Ollama must be running).
2. In `.env` (copy from `.env.example`): `LLM_BASE_URL=http://localhost:11434/v1`,
   `LLM_MODEL=qwen3.5:4b`, `OPENAI_API_KEY=ollama` (any value works for Ollama).
   `LLM_MODEL` overrides `llm.model` in `config.yaml`; temperature, top_p and runs come from `config.yaml`.
3. Test: open `notebooks/02_llm_test.ipynb` (paper Fig. 10 cases, test rules, time estimate),
   or `python -m src.rag` to check retrieval.
4. In code:

```python
from src.llm_diagnose import classify_all
df = classify_all(rules, use_rag=True)    # rules in the format below; one row per (rule, run)
```

Results are saved to `results/tables/llm_rag.csv` (or `llm_norag.csv`). Columns include
`validity`, `label` (0-3), `reason`, `failed`, `raw`, `latency_s` and token counts.
`extra_lines_fn=lambda rule: [...]` adds extra lines to the prompt's `[Context]` (for Member D).

Notes: the FAISS index is cached in `.cache/rag/` and rebuilt automatically when the knowledge base
changes. Reasoning ("thinking") is switched off with `reasoning_effort="none"`, which makes a call about
20x faster; `think=false` and `/no_think` are ignored by Ollama's OpenAI-compatible endpoint.
The knowledge base must stay general: never write dataset-specific answers into it.

## Rule format (shared by B, C and D)

`generate_rules()` returns a list of dicts. Every later step (LLM, labelling, evaluation) uses this format:

| Key | Type | Example |
|---|---|---|
| `antecedent` | tuple of items, sorted | `("dT=low",)` |
| `consequent` | tuple of items, sorted | `("P=low",)` |
| `support` | float | `0.5` |
| `confidence` | float | `0.75` |
| `lift` | float | `1.5` |

Items are strings `"VARIABLE=value"`; the variable name is the part before `=`.
The list is sorted by confidence (descending), then lift (descending).
Do not change this format without agreeing with the team.

## Team

| Member | Role | Owns |
|---|---|---|
| A | Data | `preprocess.py`, `graph.py`, `discretise.py`, Dataset section |
| B | Mining | `gcarm.py`, unconstrained baseline, efficiency experiments |
| C | LLM | `rag.py`, `llm_diagnose.py`, knowledge base, prompts |
| D | Improvement & evaluation | `labels.py`, `improvement.py`, `evaluate.py`, comparisons |

## Workflow

- One branch per member (`a-data`, `b-mining`, `c-llm`, `d-improvement`); merge into `main` via pull requests.
- Never commit data, API keys or `.env` files.
- Parameters go in `config.yaml`, not hard-coded.
