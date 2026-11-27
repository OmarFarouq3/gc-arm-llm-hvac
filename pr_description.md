**Title:** Threshold baseline for Table 3 + improvement step (Member C)

<!-- claude-projects-attribution -->
_Requested by **Abdalla Alnaqbi** · [project thread](https://claude.ai/code/project/chan_01GiJDTRqnfjVvZ6HN4N9H3g?thread=cmsg_01GiJDTRqnfjVvZ6HN4N9H3gPgr2NubQFdUBKs1JnjyWNW)_

## What this adds
- `src/baseline_threshold.py`: the "conventional engineering method" of the paper's Table 3. Fixed limits on chiller COP, tower approach, chilled-water ΔT and supply temperature vs set-point. Works on rows (flag rate per fault file) and on mined rules (a rule is anomalous if an item lies fully past a limit), scored with the same accuracy / FAR / MAR as the LLM (`detection_metrics`). `check_limits()` prints how many healthy rows break each limit.
- `src/improvement.py`: `stratified_lift`, `flag_confounded`, `majority_vote` (toy data for now).
- `config.yaml`: new `baseline:` section only; the `labelling:` section from main is untouched (merges cleanly).
- Tests: `tests/test_baseline_threshold.py`, `tests/test_improvement.py`. README lists the commands.

## Limits (this plant, not textbook)
| Limit | Value | Why |
|---|---|---|
| Low COP | < 2.0 | plant runs at about 2.5; COP < 4 would flag every healthy row |
| High approach | > 9.8 °F | matches the fitted `CT_APPROACH_1` breakpoint; 7% of healthy windows vs 31% with 65% tower fouling. 11 °F never matched a mined interval, so it could not fire on rules |
| Low ΔT | < 3.6 °F | paper Case 2 rule, same design breakpoint as `discretise.py` |
| Supply vs set-point | abs > 2 °F | row level only (`CHL_SPT_DEV_i` is not mined) |

## Known limitation
From `results/breakpoints.json`, at least 10% of healthy chiller-1 windows have ΔT < 3.6 °F (winter low load) and about 7% have COP < 1.9, so the baseline will give false alarms. That matches the paper's point (its threshold baseline had a 31% FAR). Exact rates come from `python -m src.baseline_threshold` on the processed files.

## How to test
```
python tests/test_baseline_threshold.py
python tests/test_improvement.py
python -m src.baseline_threshold    # needs data/processed/*.parquet
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015roHvMyhpDXcHTbmBXybJv
