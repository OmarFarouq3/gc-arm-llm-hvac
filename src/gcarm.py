
"""Graph-constrained association rule mining.

Paper: Section 3.3, Algorithm 1, Equations 13-14.

A candidate itemset is kept only if every pair of its variables
is within the permitted graph distance.

Graph pruning happens DURING candidate generation, not afterwards.

Owner: Member B
"""

from itertools import combinations


# Small tolerance for floating-point support comparisons.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _variable(item):
    """Extract the variable name from an item.

    Example:
        "CHL_DT_1=low" -> "CHL_DT_1"
    """
    return item.split("=", 1)[0]


def _has_repeated_variable(itemset):
    """Check whether an itemset contains multiple values of one variable."""
    variables = [_variable(item) for item in itemset]
    return len(variables) != len(set(variables))


def _count_support(candidates, transactions):
    """Calculate the support of each candidate itemset."""
    n = len(transactions)

    if n == 0:
        return {}

    counts = {candidate: 0 for candidate in candidates}

    for transaction in transactions:
        for candidate in candidates:
            if candidate <= transaction:
                counts[candidate] += 1

    return {
        candidate: count / n
        for candidate, count in counts.items()
    }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_candidates(frequent_prev, k, allowed_pairs=None):
    """Generate size-k candidates using Apriori and graph constraints.

    Steps:
        1. JOIN frequent (k-1)-itemsets.
        2. Reject itemsets containing repeated variables.
        3. Apply graph-based k-hop pruning when allowed_pairs is provided.
        4. Apply the standard Apriori subset pruning.

    Parameters
    ----------
    frequent_prev : iterable
        Frequent itemsets of size k-1.
    k : int
        Required candidate itemset size.
    allowed_pairs : set of frozensets or None
        Allowed unordered pairs of variable names.
        If None, graph pruning is disabled.

    Returns
    -------
    set
        Generated candidate itemsets.
    """

    prev = list(frequent_prev)
    prev_set = set(prev)
    candidates = set()

    for i in range(len(prev)):
        for j in range(i + 1, len(prev)):

            union = prev[i] | prev[j]

            # Step 1: Join only itemsets of exactly size k.
            if len(union) != k or union in candidates:
                continue

            # Step 2: Reject multiple values of the same variable.
            if _has_repeated_variable(union):
                continue

            # Step 3: Apply graph-based pruning.
            if allowed_pairs is not None:

                variables = sorted(
                    _variable(item) for item in union
                )

                # Every pair of variables must be within the
                # permitted graph distance.
                if any(
                    frozenset((var1, var2)) not in allowed_pairs
                    for var1, var2 in combinations(variables, 2)
                ):
                    continue

            # Step 4: Standard Apriori pruning.
            # Every (k-1)-subset must already be frequent.
            if all(
                frozenset(subset) in prev_set
                for subset in combinations(union, k - 1)
            ):
                candidates.add(union)

    return candidates


# ---------------------------------------------------------------------------
# Apriori algorithm
# ---------------------------------------------------------------------------

def apriori(transactions, min_support, max_size=4, allowed_pairs=None):
    """Run the Apriori algorithm with optional graph-based pruning.

    Parameters
    ----------
    transactions : list
        Transactions containing items such as "CHL_POW_1=high".
    min_support : float
        Minimum support threshold.
    max_size : int
        Maximum candidate itemset size.
    allowed_pairs : set of frozensets or None
        Allowed variable pairs obtained from the HVAC association graph.
        If None, ordinary Apriori is performed.

    Returns
    -------
    dict
        Frequent itemsets and their support values.
    """

    transactions = [frozenset(t) for t in transactions]

    if not transactions:
        return {}

    result = {}

    # Level 1: Generate all individual items.
    items = {
        frozenset([item])
        for transaction in transactions
        for item in transaction
    }

    support = _count_support(items, transactions)

    frequent = {
        candidate: value
        for candidate, value in support.items()
        if value >= min_support - _EPS
    }

    result.update(frequent)

    # Levels 2, 3, ...: Generate larger itemsets.
    k = 2

    while frequent and k <= max_size:

        candidates = _generate_candidates(
            frequent.keys(),
            k,
            allowed_pairs
        )

        support = _count_support(candidates, transactions)

        frequent = {
            candidate: value
            for candidate, value in support.items()
            if value >= min_support - _EPS
        }

        result.update(frequent)

        k += 1

    return result


# ---------------------------------------------------------------------------
# Association rule generation
# ---------------------------------------------------------------------------

def generate_rules(itemsets, min_confidence):
    """Generate association rules from frequent itemsets.

    For each frequent itemset X:

        support    = support(X)
        confidence = support(X) / support(antecedent)
        lift       = confidence / support(consequent)

    Returns
    -------
    list
        Association rules sorted by descending confidence and lift.
    """

    rules = []

    for itemset, support_value in itemsets.items():

        if len(itemset) < 2:
            continue

        # Generate all non-empty proper subsets as antecedents.
        for size in range(1, len(itemset)):

            for antecedent_items in combinations(
                sorted(itemset), size
            ):

                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent

                confidence = (
                    support_value / itemsets[antecedent]
                )

                if confidence < min_confidence - _EPS:
                    continue

                lift = confidence / itemsets[consequent]

                rules.append({
                    "antecedent": tuple(sorted(antecedent)),
                    "consequent": tuple(sorted(consequent)),
                    "support": float(support_value),
                    "confidence": float(confidence),
                    "lift": float(lift),
                })

    # Deterministic sorting.
    rules.sort(
        key=lambda rule: (
            -rule["confidence"],
            -rule["lift"],
            rule["antecedent"],
            rule["consequent"]
        )
    )

    return rules


# ---------------------------------------------------------------------------
# Unconstrained baseline
# ---------------------------------------------------------------------------

def unconstrained_baseline(transactions, min_support, min_confidence):
    """Placeholder for the project's unconstrained baseline implementation."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Demo
# Run using: python src/gcarm.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    transactions = [
        {"dT=low", "P=low", "OA=cool"},
        {"dT=low", "P=low", "OA=cool"},
        {"dT=normal", "P=high", "OA=hot"},
        {"dT=low", "P=low", "OA=cool"},
        {"dT=normal", "P=high", "OA=hot"},
        {"dT=low", "P=high", "OA=hot"},
    ]

    min_support = 0.3
    min_confidence = 0.6

    itemsets = apriori(
        transactions,
        min_support
    )

    print(
        f"Frequent itemsets "
        f"(min_support={min_support}): {len(itemsets)}"
    )

    for itemset, support_value in sorted(
        itemsets.items(),
        key=lambda x: (
            len(x[0]),
            -x[1],
            sorted(x[0])
        )
    ):
        print(
            f"{len(itemset):>4}  "
            f"{support_value:>7.3f}  "
            f"{{{', '.join(sorted(itemset))}}}"
        )

    rules = generate_rules(itemsets, min_confidence)

    print(
        f"\nRules (min_confidence={min_confidence}): "
        f"{len(rules)}"
    )

    for rule in rules:
        print(
            f"{rule['antecedent']} -> {rule['consequent']} "
            f"| Support: {rule['support']:.3f} "
            f"| Confidence: {rule['confidence']:.3f} "
            f"| Lift: {rule['lift']:.3f}"
        )
