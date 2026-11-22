# Daily tasks

Deadline: presentation and report on **7 December**.

| Dates | Goal |
|---|---|
| Nov 22-24 | Dataset downloaded and explored; fault-to-category mapping; improvement proposal sent |
| Nov 25-29 | Full baseline pipeline working end to end (rough is fine) |
| Nov 30-Dec 2 | Baseline experiments + ablations; improvement implemented |
| Dec 2-4 | Improvement experiments; tables and figures |
| Dec 3-6 | Report (conference template) + slides |
| Dec 7 | Present |

## Day 1: Sun 22 Nov (setup and first look at the data)

**Everyone**
- [✅] Clone the repo, create a virtual environment, `pip install -r requirements.txt`
- [✅ ] Skim the LBNL dataset paper (Granderson et al., Scientific Data, 2023), chiller plant section

**Member A: data**
- [✅ ] Download the chiller plant dataset into `data/raw/chiller_plant/` (see `data/README.md`)
- [✅ ] Explore in `notebooks/01_explore.ipynb`: rows, columns, sampling interval, date range, sensors + units
- [✅ ] Find how fault labels are stored; list the 7 faults and severity levels
- [✅ ] Deliverable: `data/dataset_notes.md`

**Member B: mining**
- [✅ ] Re-read Algorithm 1 and Eq. 13-14
- [✅ ] First plain-Python Apriori in `src/gcarm.py` on a toy dataset (no graph constraint yet)
- [✅ ] Deliverable: toy Apriori printing itemsets with support and rules with confidence

**Member C: LLM**
- [ ] Decide LLM access (API cost estimate for ~300 rules x 3 runs, or local model via Ollama)
- [ ] Test one prompt with the paper's Fig. 10 Case 2 rule; get JSON {validity, label, reason}
- [ ] Deliverable: working prompt + note on cost/speed in `notebooks/02_llm_test.ipynb`

**Member D: improvement and coordination**
- [ ] Send the improvement proposal to the instructor
- [ ] Draft the ground-truth labelling scheme (rules vs. fault periods, fault -> 4 categories)
- [ ] Deliverable: proposal sent; scheme in `report/labelling_scheme.md`

**End of day:** post deliverable + blockers in the group chat.
