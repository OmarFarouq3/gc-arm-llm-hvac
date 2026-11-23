
"""
Association graph for the HVAC system.

Node types:
    - Sensor and control variables
    - Derived metrics

Edge types:
    - Physical connections
    - Thermodynamic dependencies
    - Control dependencies

Graph distances are treated as undirected for the k-hop constraint.
"""

import networkx as nx


def build_graph(variables):
    """
    Build the HVAC association graph.

    Parameters
    ----------
    variables : dict
        Dictionary containing:
        - sensor_variables
        - derived_metrics
        - physical_edges
        - thermodynamic_edges
        - control_edges

    Returns
    -------
    nx.MultiGraph
        Undirected graph preserving different edge types.
    """

    # 1. Validate input
    required_keys = {
        "sensor_variables",
        "derived_metrics",
        "physical_edges",
        "thermodynamic_edges",
        "control_edges",
    }

    if not isinstance(variables, dict):
        raise TypeError("variables must be a dictionary")

    missing_keys = required_keys - set(variables.keys())

    if missing_keys:
        raise ValueError(f"Missing input keys: {missing_keys}")

    # 2. Identify node groups
    sensor_nodes = set(variables["sensor_variables"])
    metric_nodes = set(variables["derived_metrics"].keys())

    # A node cannot belong to both groups
    overlap = sensor_nodes & metric_nodes

    if overlap:
        raise ValueError(f"Nodes classified twice: {overlap}")

    # 3. Create an undirected MultiGraph
    G = nx.MultiGraph()

    # 4. Add sensor and control variables
    G.add_nodes_from(
        (node, {"node_type": "sensor_or_control"})
        for node in sensor_nodes
    )

    # 5. Add derived metric nodes
    G.add_nodes_from(
        (node, {"node_type": "derived_metric"})
        for node in metric_nodes
    )

    # 6. Prepare edge groups
    edge_groups = [
        ("physical", variables["physical_edges"]),
        ("thermodynamic", variables["thermodynamic_edges"]),
        ("control", variables["control_edges"]),
    ]

    # 7. Add edges with their relationship types
    for edge_type, edges in edge_groups:

        for edge in edges:

            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ValueError(f"Invalid edge: {edge}")

            source, target = edge

            if source not in G or target not in G:
                raise ValueError(
                    f"Unknown node in {edge_type} edge: {edge}"
                )

            G.add_edge(
                source,
                target,
                edge_type=edge_type
            )

    return G


def k_hop_neighbourhood(G, node, k):
    """
    Return all nodes within k graph hops of a given node.

    The graph is treated as undirected when calculating distances.

    Parameters
    ----------
    G : NetworkX graph
    node : str
        Starting node.
    k : int
        Maximum number of hops.

    Returns
    -------
    set
        Neighbouring nodes within k hops, excluding the starting node.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")

    if node not in G:
        raise nx.NodeNotFound(f"Node '{node}' is not in the graph")

    # Treat directed graphs as undirected for distance calculations
    graph = G.to_undirected() if G.is_directed() else G

    # Find shortest-path distances from the starting node
    distances = nx.single_source_shortest_path_length(
        graph,
        source=node,
        cutoff=k
    )

    # Exclude the starting node
    return set(distances.keys()) - {node}


def allowed_pairs(G, k):
    """
    Return all unique unordered node pairs whose shortest-path
    distance is at most k.

    Parameters
    ----------
    G : NetworkX graph
    k : int
        Maximum allowed graph distance.

    Returns
    -------
    set
        Set of frozensets containing allowed node pairs.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")

    # Treat the graph as undirected
    graph = G.to_undirected() if G.is_directed() else G

    pairs = set()

    # Calculate shortest-path distances from every node
    for source, distances in nx.all_pairs_shortest_path_length(
        graph,
        cutoff=k
    ):

        for target in distances:

            if source != target:
                pairs.add(frozenset((source, target)))

    return pairs
