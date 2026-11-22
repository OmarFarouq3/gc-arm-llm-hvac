"""Graph-constrained association rule mining.

Paper: Section 3.3, step 3, Algorithm 1, Eq. 13-14. A candidate itemset is kept only
if every pair of its variables is within k hops. Pruning happens DURING candidate
generation, not as a filter afterwards.

Owner: Member B
"""

from itertools import combinations

# Floats like 2/6 can come out a hair below their "true" value. A tiny tolerance
# makes sure an itemset whose support is exactly min_support is not dropped by
# rounding noise.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _variable(item):
    """Return the variable name of an item: "dT=low" -> "dT"."""
    return item.split("=", 1)[0]


def _has_repeated_variable(itemset):
    """True if the itemset has two items of the same variable,
    e.g. {"dT=low", "dT=normal"}. Such an itemset can never occur in a real
    transaction (a sensor has one value at a time), so it is pointless to count it."""
    variables = [_variable(item) for item in itemset]
    return len(variables) != len(set(variables))


def _count_support(candidates, transactions):
    """Return {candidate: support} where support = fraction of transactions
    that contain every item of the candidate."""
    n = len(transactions)
    counts = {c: 0 for c in candidates}
    for t in transactions:
        for c in candidates:
            if c <= t:          # c is a subset of transaction t
                counts[c] += 1
    return {c: counts[c] / n for c in candidates}


def _generate_candidates(frequent_prev, k, allowed_pairs=None):
    """Build size-k candidate itemsets from the frequent (k-1)-itemsets.

    Steps (classic Apriori):
      1. JOIN:  merge two frequent (k-1)-itemsets that share k-2 items; the union
                then has exactly k items.
      2. Skip candidates with two values of the same variable.
      3. (Next task) graph constraint: skip candidates whose variables are not
                all within k hops of each other.
      4. PRUNE: skip candidates that have an infrequent (k-1)-subset. If any
                subset is infrequent, the candidate cannot be frequent either
                (the "Apriori property"), so we never waste time counting it.
    """
    prev = list(frequent_prev)
    prev_set = set(prev)
    candidates = set()

    for i in range(len(prev)):
        for j in range(i + 1, len(prev)):
            union = prev[i] | prev[j]

            # Step 1: only keep unions of exactly size k (they shared k-2 items).
            if len(union) != k or union in candidates:
                continue

            # Step 2: same variable twice -> impossible itemset, skip.
            if _has_repeated_variable(union):
                continue

            # Step 3: graph constraint (k-hop pruning), paper Algorithm 1, Eq. 13-14.
            # TODO(next task): when allowed_pairs is not None, keep the candidate
            # ONLY if every pair of its variables is in allowed_pairs, e.g.
            #     variables = sorted(_variable(item) for item in union)
            #     if any(frozenset(p) not in allowed_pairs
            #            for p in combinations(variables, 2)):
            #         continue
            # This check must stay HERE, inside candidate generation, so pruned
            # candidates are never counted (that is the efficiency gain of GC-ARM).
            # For now allowed_pairs is ignored.

            # Step 4: Apriori pruning - every (k-1)-subset must be frequent.
            if all(frozenset(sub) in prev_set for sub in combinations(union, k - 1)):
                candidates.add(union)

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apriori(transactions, min_support, max_size=4, allowed_pairs=None):
    """Level-wise Apriori.

    Args:
        transactions: list of sets of "VARIABLE=value" strings.
        min_support: minimum fraction of transactions an itemset must appear in.
        max_size: largest itemset size to search for.
        allowed_pairs: reserved for the graph constraint (next task); ignored if None.

    Returns:
        {frozenset(itemset): support} for every frequent itemset (all sizes).
    """
    transactions = [frozenset(t) for t in transactions]
    if not transactions:
        return {}

    result = {}

    # Level 1: every single item that appears in the data is a candidate.
    items = {frozenset([item]) for t in transactions for item in t}
    support = _count_support(items, transactions)
    frequent = {c: s for c, s in support.items() if s >= min_support - _EPS}
    result.update(frequent)

    # Levels 2, 3, ...: grow itemsets one item at a time.
    k = 2
    while frequent and k <= max_size:
        candidates = _generate_candidates(frequent.keys(), k, allowed_pairs)
        support = _count_support(candidates, transactions)
        frequent = {c: s for c, s in support.items() if s >= min_support - _EPS}
        result.update(frequent)   # if nothing is frequent, the loop stops next round
        k += 1

    return result


def generate_rules(itemsets, min_confidence):
    """Turn frequent itemsets into association rules A -> B.

    For each frequent itemset X with 2+ items, every non-empty proper subset A
    is tried as antecedent, with B = X - A as consequent.
        support    = support(X)                 how often A and B occur together
        confidence = support(X) / support(A)    P(B | A)
        lift       = confidence / support(B)    >1 means A and B co-occur more
                                                than chance would predict
    Because Apriori returns every frequent itemset, and subsets of frequent
    itemsets are also frequent, support(A) and support(B) are always available.

    Returns a list of dicts with keys "antecedent", "consequent" (sorted tuples),
    "support", "confidence", "lift", sorted by confidence then lift (both descending).
    """
    rules = []
    for itemset, sup in itemsets.items():
        if len(itemset) < 2:
            continue
        # Antecedent sizes 1 .. len-1 give all non-empty proper subsets.
        for size in range(1, len(itemset)):
            for a in combinations(sorted(itemset), size):
                antecedent = frozenset(a)
                consequent = itemset - antecedent
                confidence = sup / itemsets[antecedent]
                if confidence < min_confidence - _EPS:
                    continue
                lift = confidence / itemsets[consequent]
                rules.append({
                    "antecedent": tuple(sorted(antecedent)),
                    "consequent": tuple(sorted(consequent)),
                    "support": float(sup),
                    "confidence": float(confidence),
                    "lift": float(lift),
                })

    # Highest confidence first; ties broken by higher lift. Remaining ties are
    # ordered alphabetically by antecedent/consequent so the output is identical
    # on every run (Python's set order is not).
    rules.sort(key=lambda r: (-r["confidence"], -r["lift"], r["antecedent"], r["consequent"]))
    return rules


def unconstrained_baseline(transactions, min_support, min_confidence):
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Demo on toy data:  python src/gcarm.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transactions = [
        {"dT=low",    "P=low",  "OA=cool"},
        {"dT=low",    "P=low",  "OA=cool"},
        {"dT=normal", "P=high", "OA=hot"},
        {"dT=low",    "P=low",  "OA=cool"},
        {"dT=normal", "P=high", "OA=hot"},
        {"dT=low",    "P=high", "OA=hot"},
    ]
    min_support, min_confidence = 0.3, 0.6

    itemsets = apriori(transactions, min_support)
    print(f"Frequent itemsets (min_support={min_support}): {len(itemsets)}")
    print(f"{'size':>4}  {'support':>7}  itemset")
    for itemset, sup in sorted(itemsets.items(), key=lambda x: (len(x[0]), -x[1], sorted(x[0]))):
        print(f"{len(itemset):>4}  {sup:>7.3f}  {{{', '.join(sorted(itemset))}}}")

    rules = generate_rules(itemsets, min_confidence)
    print(f"\nRules (min_confidence={min_confidence}): {len(rules)}")
    print(f"{'antecedent':<20} -> {'consequent':<20} {'supp':>6} {'conf':>6} {'lift':>6}")
    for r in rules:
        print(f"{', '.join(r['antecedent']):<20} -> {', '.join(r['consequent']):<20} "
              f"{r['support']:>6.3f} {r['confidence']:>6.3f} {r['lift']:>6.3f}")
