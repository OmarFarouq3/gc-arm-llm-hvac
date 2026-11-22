"""Ground-truth labels for rules, derived from the dataset fault timeline.

A rule is an anomaly rule if its antecedent occurs mainly in faulty periods;
fault types are mapped to the 4 paper categories.

Owner: Member D
"""


def fault_mask(df):
    raise NotImplementedError


def rule_fault_ratio(rule, df_discrete, mask):
    raise NotImplementedError


def label_rules(rules, df_discrete, fault_info):
    raise NotImplementedError
