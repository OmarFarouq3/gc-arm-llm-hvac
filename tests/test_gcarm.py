"""Toy-data and graph-pruning tests for src/gcarm.py.

Run from the repository root with:
    python tests/test_gcarm.py

The expected values in the toy-data tests are computed by hand from the six
transactions below. The graph-pruning test reads graph.k from config.yaml.
"""

import os
import sys
import networkx as nx
import yaml

# Make the repository root importable when this file is run directly.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.gcarm import apriori, generate_rules  # noqa: E402
from src.graph import allowed_pairs, build_hvac_graph  # noqa: E402

TOL = 1e-6

TRANSACTIONS = [
    {"dT=low", "P=low", "OA=cool"},
    {"dT=low", "P=low", "OA=cool"},
    {"dT=normal", "P=high", "OA=hot"},
    {"dT=low", "P=low", "OA=cool"},
    {"dT=normal", "P=high", "OA=hot"},
    {"dT=low", "P=high", "OA=hot"},
]

MIN_SUPPORT = 0.3
MIN_CONFIDENCE = 0.6


def close(a, b):
    return abs(a - b) < TOL


def find_rule(rules, antecedent, consequent):
    for rule in rules:
        if (
            rule["antecedent"] == tuple(sorted(antecedent))
            and rule["consequent"] == tuple(sorted(consequent))
        ):
            return rule
    raise AssertionError(
        f"rule {antecedent} -> {consequent} not found"
    )


def load_graph_k():
    """Load and validate the graph hop limit from the repository config."""
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    assert os.path.isfile(config_path), (
        f"config.yaml was not found at: {config_path}"
    )

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert isinstance(config, dict), "config.yaml must contain a YAML mapping"
    assert isinstance(config.get("graph"), dict), (
        "config.yaml must contain a 'graph' section"
    )

    k = config["graph"].get("k")
    assert isinstance(k, int) and not isinstance(k, bool) and k >= 0, (
        f"config.yaml graph.k must be a non-negative integer; got {k!r}"
    )

    # The reviewer specifically requested the k=2 configuration for this test.
    assert k == 2, f"Expected graph.k = 2 in config.yaml, got {k}"
    return k


def find_connected_pair_beyond_k(graph, k):
    """Return a deterministic connected node pair with shortest path > k."""
    undirected_graph = graph.to_undirected() if graph.is_directed() else graph

    for source in sorted(undirected_graph.nodes):
        distances = nx.single_source_shortest_path_length(
            undirected_graph, source
        )
        for target in sorted(distances):
            if source < target and distances[target] > k:
                return source, target, distances[target]

    return None


def test_itemset_counts():
    itemsets = apriori(TRANSACTIONS, MIN_SUPPORT)
    multi = [itemset for itemset in itemsets if len(itemset) >= 2]
    pairs = [itemset for itemset in multi if len(itemset) == 2]
    triples = [itemset for itemset in multi if len(itemset) == 3]

    assert len(multi) == 8, (
        f"expected 8 itemsets of size >= 2, got {len(multi)}"
    )
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

    for itemset, expected_support in expected.items():
        assert itemset in itemsets, f"{set(itemset)} missing"
        assert close(itemsets[itemset], expected_support), (
            f"{set(itemset)}: {itemsets[itemset]} != {expected_support}"
        )


def test_rules():
    rules = generate_rules(
        apriori(TRANSACTIONS, MIN_SUPPORT), MIN_CONFIDENCE
    )

    rule = find_rule(rules, ["dT=low"], ["P=low"])
    assert close(rule["confidence"], 0.75), rule
    assert close(rule["lift"], 1.5), rule

    rule = find_rule(rules, ["P=low"], ["dT=low"])
    assert close(rule["confidence"], 1.0), rule
    assert close(rule["lift"], 1.5), rule

    # Output format other team members depend on.
    for rule in rules:
        assert set(rule) == {
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift",
        }, rule
        assert isinstance(rule["antecedent"], tuple)
        assert rule["antecedent"] == tuple(sorted(rule["antecedent"]))
        assert isinstance(rule["consequent"], tuple)
        assert rule["consequent"] == tuple(sorted(rule["consequent"]))
        assert all(
            isinstance(rule[key], float)
            for key in ("support", "confidence", "lift")
        )
        assert rule["confidence"] >= MIN_CONFIDENCE - TOL

    # Sorted by confidence descending, then lift descending.
    keys = [(-rule["confidence"], -rule["lift"]) for rule in rules]
    assert keys == sorted(keys), (
        "rules not sorted by confidence then lift"
    )


def test_no_repeated_variable():
    itemsets = apriori(TRANSACTIONS, MIN_SUPPORT)
    for itemset in itemsets:
        variables = [item.split("=", 1)[0] for item in itemset]
        assert len(variables) == len(set(variables)), (
            f"repeated variable in {set(itemset)}"
        )


def test_graph_pruning_uses_config_k():
    """Check that Apriori prunes distant graph pairs using config.yaml's k."""
    k = load_graph_k()
    graph = build_hvac_graph()
    undirected_graph = graph.to_undirected() if graph.is_directed() else graph

    # Confirm the graph can supply both a distant pair and an allowed pair.
    distant_pair = find_connected_pair_beyond_k(undirected_graph, k)
    assert distant_pair is not None, (
        f"No connected graph-node pair exists at a distance greater than k={k}"
    )

    far_node_a, far_node_b, far_distance = distant_pair
    allowed = allowed_pairs(graph, k)
    far_pair = frozenset((far_node_a, far_node_b))

    assert nx.has_path(undirected_graph, far_node_a, far_node_b)
    assert far_distance > k
    assert far_pair not in allowed, (
        "A pair farther than k hops was incorrectly included in allowed_pairs"
    )

    assert allowed, "allowed_pairs returned no pairs; cannot test retention"
    near_pair = min(allowed, key=lambda pair: tuple(sorted(pair)))
    near_node_a, near_node_b = sorted(near_pair)

    assert nx.shortest_path_length(
        undirected_graph, near_node_a, near_node_b
    ) <= k

    # Use repeated synthetic transactions so both individual items are
    # frequent. This isolates the graph-pruning behaviour of candidate
    # generation from support-threshold effects.
    far_transaction = {
        f"{far_node_a}=test_value",
        f"{far_node_b}=test_value",
    }
    far_transactions = [set(far_transaction) for _ in range(3)]

    far_itemsets = apriori(
        far_transactions,
        min_support=MIN_SUPPORT,
        max_size=2,
        allowed_pairs=allowed,
    )

    assert not any(
        itemset == frozenset(far_transaction)
        for itemset in far_itemsets
    ), (
        "Apriori failed to prune a connected pair whose shortest path "
        f"is {far_distance} hops (k={k}): {far_node_a}, {far_node_b}"
    )

    near_transaction = {
        f"{near_node_a}=test_value",
        f"{near_node_b}=test_value",
    }
    near_transactions = [set(near_transaction) for _ in range(3)]

    near_itemsets = apriori(
        near_transactions,
        min_support=MIN_SUPPORT,
        max_size=2,
        allowed_pairs=allowed,
    )

    assert frozenset(near_transaction) in near_itemsets, (
        "Apriori incorrectly pruned an allowed pair within k hops: "
        f"{near_node_a}, {near_node_b}"
    )

    print(f"  Configured graph.k: {k}")
    print(
        f"  Distant pair pruned ({far_distance} hops): "
        f"{far_node_a}, {far_node_b}"
    )
    print(
        f"  Nearby pair retained (within {k} hops): "
        f"{near_node_a}, {near_node_b}"
    )


if __name__ == "__main__":
    tests = [
        test_itemset_counts,
        test_supports,
        test_rules,
        test_no_repeated_variable,
        test_graph_pruning_uses_config_k,
    ]

    for test in tests:
        test()
        print(f"PASS  {test.__name__}")

    print(f"\nAll {len(tests)} tests passed.")
