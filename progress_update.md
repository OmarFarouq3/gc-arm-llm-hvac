Subject: Progress update – threshold baseline and improvement step

Hi,

Quick update on my part:

1. Threshold baseline (Table 3) is finished and tested. It flags 15-min windows where a running chiller or tower breaks a fixed limit on COP, tower approach, chilled-water ΔT, or supply temperature vs set-point. The limits now follow our plant's real ranges rather than textbook values: COP < 2.0 (plant normally ~2.5) and approach > 11 °F (normally ~8 °F), plus ΔT < 3.6 °F and |supply − set-point| > 2 °F. With textbook limits (COP < 4) every healthy row would have been flagged. The same limits also classify mined rules as normal/anomalous, scored with the same accuracy / FAR / MAR as the LLM. A check_limits() report compares each limit with the fault-free data, so we can confirm them once the processed data is in.

2. Improvement step (src/improvement.py) is started on toy data, with tests: stratified_lift (a rule's lift inside outdoor-temperature bins), flag_confounded (flags a rule whose lift is about 1 inside the bins), and majority_vote (combines labels from several LLM runs into one).

Next: run the baseline on the processed LBNL files to fill Table 3, and plug the improvement step into the rule pipeline.

I haven't received your part yet. Could you send it, or let me know when to expect it?

Thanks,
Abdalla
