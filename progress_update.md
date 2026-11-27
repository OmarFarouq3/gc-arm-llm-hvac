Subject: Progress update – threshold baseline and improvement step (Abdalla, Member C)

Hi Omar,

Quick update on my part:

1. Threshold baseline for Table 3 is finished, tested and in a pull request from `threshold-baseline` into main for you to review. It flags 15-min windows (and mined rules) where a running chiller or tower breaks a fixed limit: COP < 2.0, tower approach > 9.8 °F, chilled-water ΔT < 3.6 °F, or supply more than 2 °F from set-point. The limits follow our plant's real ranges and line up with the breakpoints in results/breakpoints.json, so the same limits work on mined rules and are scored with the same accuracy / FAR / MAR as the LLM.

2. Checked against our fault-free breakpoints: tower approach ≥ 9.8 °F happens in 7% of healthy windows but 31% with 65% tower fouling, so that limit separates well. ΔT < 3.6 °F occurs in at least 10% of healthy chiller-1 windows (winter low load), so the baseline will give some false alarms, which is in line with the paper's 31% FAR for this method.

3. Improvement step (src/improvement.py) is in the same PR, tested on toy data: stratified_lift, flag_confounded and majority_vote.

Next: run the baseline on the processed LBNL files to fill Table 3, and plug the improvement step into the rule pipeline once the mined rules are ready.

I haven't received your part yet. Could you send it, or let me know when to expect it?

Thanks,
Abdalla
