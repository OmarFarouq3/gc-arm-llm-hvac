"""Toy-data tests for src/gcarm.py. Run with:  python tests/test_gcarm.py

Expected values were computed by hand from the 6 toy transactions below.
"""

import os
import sys

# Make the repo root importable so "from src.gcarm import ..." works when the
# file is run directly (python puts tests/ on the path, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gcarm import apriori, generate_rules  # noqa: E402

TOL = 1e-6

TRANSACTIONS = [
    {"dT=low",    "P=low",  "OA=cool"},
    {"dT=low",    "P=low",  "OA=cool"},
    {"dT=normal", "P=high", "OA=hot"},
    {"dT=low",    "P=low",  "OA=cool"},
    {"dT=normal", "P=high", "OA=hot"},
    {"dT=low",    "P=high", "OA=hot"},
]
MIN_SUPPORT = 0.3
MIN_CONFIDENCE = 0.6


def close(a, b):
    return abs(a - b) < TOL


def find_rule(rules, antecedent, consequent):
    for r in rules:
        if r["antecedent"] == tuple(sorted(antecedent)) and r["consequent"] == tuple(sorted(consequent)):
            return r
    raise AssertionError(f"rule {antecedent} -> {consequent} not found")


def test_itemset_counts():
    itemsets = apriori(TRANSACTIONS, MIN_SUPPORT)
    multi = [s for s in itemsets if len(s) >= 2]
    pairs = [s for s in multi if len(s) == 2]
    triples = [s for s in multi if len(s) == 3]
    assert len(multi) == 8, f"expected 8 itemsets of size >= 2, got {len(multi)}"
    assert len(pairs) == 6, f"expected 6 pairs, got {len(pairs)}"
    assert len(triples) == 2, f"expected 2 triples, got {len(triples)}"


def test_supports():
    itemsets = apriori(TRANSACTIONS, MIN_SUPPORT)
    expected = {
        frozenset({"dT=low", "P=low"}): 0.5,
        frozenset({"dT=low", "OA=cool"}): 0.5,
        frozenset({"dT=low", "P=low", "OA=cool"}): 0.5,
        frozenset({"dT=normal", "P=high", "OA=hot"}): 1 / 3,
    }
    for itemset, sup in expected.items():
        assert itemset in itemsets, f"{set(itemset)} missing"
        assert close(itemsets[itemset], sup), f"{set(itemset)}: {itemsets[itemset]} != {sup}"


def test_rules():
    rules = generate_rules(apriori(TRANSACTIONS, MIN_SUPPORT), MIN_CONFIDENCE)

    r = find_rule(rules, ["dT=low"], ["P=low"])
    assert close(r["confidence"], 0.75), r
    assert close(r["lift"], 1.5), r

    r = find_rule(rules, ["P=low"], ["dT=low"])
    assert close(r["confidence"], 1.0), r
    assert close(r["lift"], 1.5), r

    # Output format other team members depend on.
    for r in rules:
        assert set(r) == {"antecedent", "consequent", "support", "confidence", "lift"}, r
        assert isinstance(r["antecedent"], tuple) and r["antecedent"] == tuple(sorted(r["antecedent"]))
        assert isinstance(r["consequent"], tuple) and r["consequent"] == tuple(sorted(r["consequent"]))
        assert all(isinstance(r[k], float) for k in ("support", "confidence", "lift"))
        assert r["confidence"] >= MIN_CONFIDENCE - TOL

    # Sorted by confidence desc, then lift desc.
    keys = [(-r["confidence"], -r["lift"]) for r in rules]
    assert keys == sorted(keys), "rules not sorted by confidence then lift"


def test_no_repeated_variable():
    itemsets = apriori(TRANSACTIONS, MIN_SUPPORT)
    for itemset in itemsets:
        variables = [item.split("=", 1)[0] for item in itemset]
        assert len(variables) == len(set(variables)), f"repeated variable in {set(itemset)}"


if __name__ == "__main__":
    tests = [test_itemset_counts, test_supports, test_rules, test_no_repeated_variable]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
