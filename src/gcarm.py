"""Graph-constrained association rule mining.

Paper: Section 3.3, step 3, Algorithm 1, Eq. 13-14. A candidate itemset is kept only
if every pair of its variables is within k hops. Pruning happens DURING candidate
generation, not as a filter afterwards.

Owner: Member B
"""


def apriori(transactions, min_support, max_size, allowed_pairs=None):
    raise NotImplementedError


def generate_rules(itemsets, min_confidence):
    raise NotImplementedError


def unconstrained_baseline(transactions, min_support, min_confidence):
    raise NotImplementedError
