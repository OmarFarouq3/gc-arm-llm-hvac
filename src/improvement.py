"""Our improvement (not in the paper).

1. Confounder-aware rule screening: recompute lift within bins of a confounder
   (e.g. outdoor temperature); a rule whose association disappears is flagged
   as likely confounded, and this evidence is added to the LLM prompt.
2. Self-consistency: temperature 0 / majority vote over several runs.

Owner: Member D
"""


def stratified_lift(rule, df_discrete, confounder, n_bins):
    raise NotImplementedError


def flag_confounded(rules, df_discrete, confounder):
    raise NotImplementedError


def majority_vote(predictions_by_run):
    raise NotImplementedError
