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
