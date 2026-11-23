"""
Association graph for the HVAC system.
"""

import networkx as nx


def build_graph(variables):
    """Build an undirected HVAC MultiGraph with typed nodes and edges."""
    required_keys = {
        "sensor_variables",
        "derived_metrics",
        "physical_edges",
        "thermodynamic_edges",
        "control_edges",
    }

    if not isinstance(variables, dict):
        raise TypeError("variables must be a dictionary")

    missing_keys = required_keys - set(variables)
    if missing_keys:
        raise ValueError(f"Missing input keys: {missing_keys}")

    sensor_nodes = set(variables["sensor_variables"])
    metric_nodes = set(variables["derived_metrics"])

    overlap = sensor_nodes & metric_nodes
    if overlap:
        raise ValueError(f"Nodes classified twice: {overlap}")

    G = nx.MultiGraph()

    G.add_nodes_from(
        (node, {"node_type": "sensor_or_control"})
        for node in sensor_nodes
    )
    G.add_nodes_from(
        (node, {"node_type": "derived_metric"})
        for node in metric_nodes
    )

    edge_groups = [
        ("physical", variables["physical_edges"]),
        ("thermodynamic", variables["thermodynamic_edges"]),
        ("control", variables["control_edges"]),
    ]

    for edge_type, edges in edge_groups:
        for edge in edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ValueError(f"Invalid edge: {edge}")

            source, target = edge
            if source not in G or target not in G:
                raise ValueError(
                    f"Unknown node in {edge_type} edge: {edge}"
                )

            G.add_edge(source, target, edge_type=edge_type)

    return G


def k_hop_neighbourhood(G, node, k):
    """Return nodes at shortest-path distance 1 through k from node."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")

    if node not in G:
        raise nx.NodeNotFound(f"Node '{node}' is not in the graph")

    graph = G.to_undirected() if G.is_directed() else G
    distances = nx.single_source_shortest_path_length(
        graph, source=node, cutoff=k
    )
    return set(distances) - {node}


def allowed_pairs(G, k):
    """Return unique unordered node pairs whose distance is at most k."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")

    graph = G.to_undirected() if G.is_directed() else G
    pairs = set()

    for source, distances in nx.all_pairs_shortest_path_length(
        graph, cutoff=k
    ):
        for target in distances:
            if source != target:
                pairs.add(frozenset((source, target)))

    return pairs

