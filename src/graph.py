
"""HVAC association graph for graph-constrained association rule mining.

Contains:
    - Generic graph construction
    - Complete HVAC node and edge definitions
    - k-hop neighbourhood calculation
    - Allowed variable-pair calculation

Graph distances are treated as undirected.
"""

import networkx as nx
from itertools import combinations


# ============================================================================
# 1. GENERIC GRAPH BUILDER
# ============================================================================

def build_graph(variables):
    """Build an undirected HVAC association graph.

    Parameters
    ----------
    variables : dict
        Required keys:
            sensor_variables
            derived_metrics
            physical_edges
            thermodynamic_edges
            control_edges

    Returns
    -------
    nx.MultiGraph
    """

    required_keys = {
        "sensor_variables",
        "derived_metrics",
        "physical_edges",
        "thermodynamic_edges",
        "control_edges",
    }

    if not isinstance(variables, dict):
        raise TypeError("variables must be a dictionary")

    missing_keys = required_keys - variables.keys()

    if missing_keys:
        raise ValueError(f"Missing input keys: {missing_keys}")

    sensor_nodes = set(variables["sensor_variables"])
    metric_nodes = set(variables["derived_metrics"].keys())

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


# ============================================================================
# 2. COMPLETE HVAC GRAPH
# ============================================================================

def build_hvac_graph():
    """Build the complete HVAC association graph.

    Returns
    -------
    nx.MultiGraph
        Graph containing HVAC sensor variables, derived metrics,
        physical relationships, thermodynamic relationships,
        and control relationships.
    """

    # ------------------------------------------------------------------------
    # A. SENSOR AND CONTROL VARIABLES
    # ------------------------------------------------------------------------

    # Chillers: 9 variables per chiller
    chiller_variables = []

    for i in range(1, 4):
        chiller_variables += [
            f"CHL_POW_{i}",
            f"CHL_SW_TEMP_{i}",
            f"CHL_RW_TEMP_{i}",
            f"CHL_CW_FLOW_{i}",
            f"CHL_SWCD_TEMP_{i}",
            f"CHL_RWCD_TEMP_{i}",
            f"CHL_CD_FLOW_{i}",
            f"CHL_COMP_SPD_CTRL_{i}",
            f"CHL_STA_{i}",
        ]

    # Cooling towers: 7 variables per tower + shared setpoint
    cooling_tower_variables = []

    for i in range(1, 4):
        cooling_tower_variables += [
            f"CT_POW_{i}",
            f"CT_FAN_SPD_{i}",
            f"CT_FLOW_{i}",
            f"CT_SW_TEMP_{i}",
            f"CT_RW_TEMP_{i}",
            f"CT_FAN_SPD_CTRL_{i}",
            f"CT_STA_{i}",
        ]

    cooling_tower_variables.append("CT_SW_TEMPSPT")

    # Condenser water loop
    condenser_variables = [
        "CDWL_CW_FLOW",
        "CDWL_SW_TEMP",
        "CDWL_RW_TEMP",
    ] + [
        f"CDWL_PM_POW_{i}" for i in range(1, 4)
    ]

    # Primary chilled-water loop
    primary_variables = [
        "CWL_PRI_CW_FLOW",
        "CWL_PRI_SW_TEMP",
        "CWL_PRI_RW_TEMP",
    ] + [
        f"CWL_PRI_PM_POW_{i}" for i in range(1, 4)
    ] + [
        "CWL_PRI_SW_TEMPSPT"
    ]

    # Secondary chilled-water loop
    secondary_variables = [
        "CWL_SEC_CW_FLOW",
        "CWL_SEC_SW_TEMP",
        "CWL_SEC_RW_TEMP",
        "CWL_SEC_DP",
        "CWL_SEC_LOAD",
    ] + [
        f"CWL_SEC_PM_POW_{i}" for i in range(1, 3)
    ] + [
        f"CWL_SEC_PM_SPD_{i}" for i in range(1, 3)
    ] + [
        "CWL_SEC_DPSPT",
        "CWL_SEC_PM_STA_1",
        "CWL_SEC_PM_STA_2",
    ]

    # Valve and outdoor air
    other_variables = [
        "TWV_CTRL",
        "OA_TEMP",
        "OA_TEMP_WB",
    ]

    # Organize all sensor/control variables
    raw_groups = {
        "Chillers": chiller_variables,
        "Cooling Towers": cooling_tower_variables,
        "Condenser Loop": condenser_variables,
        "Primary Chilled Water": primary_variables,
        "Secondary Chilled Water": secondary_variables,
        "Valve and Outdoor Air": other_variables,
    }

    sensor_variables = [
        variable
        for group in raw_groups.values()
        for variable in group
    ]

    if len(sensor_variables) != len(set(sensor_variables)):
        raise ValueError("Duplicate sensor variables detected")

    # ------------------------------------------------------------------------
    # B. DERIVED METRICS
    # ------------------------------------------------------------------------

    derived_metrics = {}

    # 1. Chilled-water Delta T
    for i in range(1, 4):
        derived_metrics[f"CHL_DT_{i}"] = [
            f"CHL_RW_TEMP_{i}",
            f"CHL_SW_TEMP_{i}",
        ]

    # 2. Chiller cooling capacity in kW (CHL_COOL_i).
    # This graph records formula dependencies; numerical feature
    # calculation belongs in the preprocessing/feature-engineering code.
    for i in range(1, 4):
        derived_metrics[f"CHL_COOL_{i}"] = [
            f"CHL_CW_FLOW_{i}",
            f"CHL_DT_{i}",
        ]

    # 3. Chiller COP
    for i in range(1, 4):
        derived_metrics[f"CHL_COP_{i}"] = [
            f"CHL_COOL_{i}",
            f"CHL_POW_{i}",
        ]

    # 4. Plant total power
    plant_power_inputs = (
        [f"CHL_POW_{i}" for i in range(1, 4)]
        + [f"CT_POW_{i}" for i in range(1, 4)]
        + [f"CDWL_PM_POW_{i}" for i in range(1, 4)]
        + [f"CWL_PRI_PM_POW_{i}" for i in range(1, 4)]
        + [f"CWL_SEC_PM_POW_{i}" for i in range(1, 3)]
    )

    derived_metrics["PLANT_POW"] = plant_power_inputs

    # 5. Plant COP
    derived_metrics["PLANT_COP"] = [
        "CWL_SEC_LOAD",
        "PLANT_POW",
    ]

    # 6. Cooling tower approach
    for i in range(1, 4):
        derived_metrics[f"CT_APPROACH_{i}"] = [
            f"CT_SW_TEMP_{i}",
            "OA_TEMP_WB",
        ]

    # 7. Condenser-water Delta T
    for i in range(1, 4):
        derived_metrics[f"CHL_CDDT_{i}"] = [
            f"CHL_SWCD_TEMP_{i}",
            f"CHL_RWCD_TEMP_{i}",
        ]

    # ------------------------------------------------------------------------
    # C. PHYSICAL EDGES
    # ------------------------------------------------------------------------

    physical_edges = []

    # Chillers <-> Primary chilled-water loop
    for i in range(1, 4):
        physical_edges.extend([
            (f"CHL_SW_TEMP_{i}", "CWL_PRI_SW_TEMP"),
            (f"CHL_RW_TEMP_{i}", "CWL_PRI_RW_TEMP"),
            (f"CHL_CW_FLOW_{i}", "CWL_PRI_CW_FLOW"),
        ])

    # Primary loop <-> Secondary loop
    physical_edges.extend([
        ("CWL_PRI_SW_TEMP", "CWL_SEC_SW_TEMP"),
        ("CWL_PRI_RW_TEMP", "CWL_SEC_RW_TEMP"),
        ("CWL_PRI_CW_FLOW", "CWL_SEC_CW_FLOW"),
    ])

    # Chillers <-> Condenser-water loop
    for i in range(1, 4):
        physical_edges.extend([
            (f"CHL_SWCD_TEMP_{i}", "CDWL_RW_TEMP"),
            (f"CHL_RWCD_TEMP_{i}", "CDWL_SW_TEMP"),
            (f"CHL_CD_FLOW_{i}", "CDWL_CW_FLOW"),
        ])

    # Cooling towers <-> Condenser-water loop
    for i in range(1, 4):
        physical_edges.extend([
            (f"CT_SW_TEMP_{i}", "CDWL_SW_TEMP"),
            (f"CT_RW_TEMP_{i}", "CDWL_RW_TEMP"),
            (f"CT_FLOW_{i}", "CDWL_CW_FLOW"),
        ])

    # ------------------------------------------------------------------------
    # D. ADDITIONAL PHYSICAL ASSOCIATIONS (15 EDGES)
    # ------------------------------------------------------------------------

    additional_physical_edges = []

    # Chiller condenser flow <-> condenser temperatures
    for i in range(1, 4):
        additional_physical_edges.extend([
            (f"CHL_CD_FLOW_{i}", f"CHL_SWCD_TEMP_{i}"),
            (f"CHL_CD_FLOW_{i}", f"CHL_RWCD_TEMP_{i}"),
        ])

    # Cooling tower flow <-> tower water temperatures
    for i in range(1, 4):
        additional_physical_edges.extend([
            (f"CT_FLOW_{i}", f"CT_SW_TEMP_{i}"),
            (f"CT_FLOW_{i}", f"CT_RW_TEMP_{i}"),
        ])

    # Condenser loop flow <-> loop temperatures
    additional_physical_edges.extend([
        ("CDWL_CW_FLOW", "CDWL_SW_TEMP"),
        ("CDWL_CW_FLOW", "CDWL_RW_TEMP"),
    ])

    # Secondary loop pressure <-> secondary loop flow
    additional_physical_edges.append(
        ("CWL_SEC_DP", "CWL_SEC_CW_FLOW")
    )

    physical_edges.extend(additional_physical_edges)

    # ------------------------------------------------------------------------
    # E. THERMODYNAMIC EDGES
    # ------------------------------------------------------------------------

    thermodynamic_edges = []

    # Formula inputs -> derived metrics
    for metric, inputs in derived_metrics.items():
        for input_variable in inputs:
            thermodynamic_edges.append((input_variable, metric))

    # Reviewer-requested chiller thermodynamic relationships
    for i in range(1, 4):
        thermodynamic_edges.extend([
            (f"CHL_RWCD_TEMP_{i}", f"CHL_POW_{i}"),
            (f"CHL_CDDT_{i}", f"CHL_COOL_{i}"),
            (f"CHL_CDDT_{i}", f"CHL_POW_{i}"),
        ])

    # ------------------------------------------------------------------------
    # F. CONTROL EDGES
    # ------------------------------------------------------------------------

    control_edges = []

    # Outdoor conditions -> HVAC setpoints
    control_edges.extend([
        ("OA_TEMP", "CWL_PRI_SW_TEMPSPT"),
        ("OA_TEMP_WB", "CT_SW_TEMPSPT"),
    ])

    # Cooling tower control chain
    for i in range(1, 4):
        control_edges.extend([
            ("CT_SW_TEMPSPT", f"CT_FAN_SPD_CTRL_{i}"),
            (f"CT_FAN_SPD_CTRL_{i}", f"CT_FAN_SPD_{i}"),
            (f"CT_FAN_SPD_{i}", f"CT_SW_TEMP_{i}"),
        ])

    # Reviewer-requested chiller compressor control chain
    for i in range(1, 4):
        control_edges.extend([
            (f"CHL_SW_TEMP_{i}", f"CHL_COMP_SPD_CTRL_{i}"),
            (f"CHL_COMP_SPD_CTRL_{i}", f"CHL_POW_{i}"),
        ])

    # Primary chilled-water supply temperature control
    control_edges.append(
        ("CWL_PRI_SW_TEMPSPT", "CWL_PRI_SW_TEMP")
    )

    # Secondary chilled-water pump pressure control
    for i in range(1, 3):
        control_edges.extend([
            ("CWL_SEC_DPSPT", f"CWL_SEC_PM_SPD_{i}"),
            (f"CWL_SEC_PM_SPD_{i}", "CWL_SEC_DP"),
            (f"CWL_SEC_PM_STA_{i}", f"CWL_SEC_PM_SPD_{i}"),
        ])

    # Thermal load -> chiller staging
    for i in range(1, 4):
        control_edges.append(
            ("CWL_SEC_LOAD", f"CHL_STA_{i}")
        )

    # Starting a chiller activates associated equipment
    for i in range(1, 4):
        control_edges.extend([
            (f"CHL_STA_{i}", f"CT_STA_{i}"),
            (f"CHL_STA_{i}", f"CDWL_PM_POW_{i}"),
            (f"CHL_STA_{i}", f"CWL_PRI_PM_POW_{i}"),
        ])

    # Chiller status gates its COP calculation
    for i in range(1, 4):
        control_edges.append(
            (f"CHL_STA_{i}", f"CHL_COP_{i}")
        )

    # Condenser three-way valve
    control_edges.append(
        ("TWV_CTRL", "CDWL_SW_TEMP")
    )

    # ------------------------------------------------------------------------
    # G. BUILD THE COMPLETE GRAPH
    # ------------------------------------------------------------------------

    graph_input = {
        "sensor_variables": sensor_variables,
        "derived_metrics": derived_metrics,
        "physical_edges": physical_edges,
        "thermodynamic_edges": thermodynamic_edges,
        "control_edges": control_edges,
    }

    G = build_graph(graph_input)

    return G


# ============================================================================
# 3. K-HOP NEIGHBOURHOOD
# ============================================================================

def k_hop_neighbourhood(G, node, k):
    """Return all nodes within k hops of a starting node.

    The graph is treated as undirected for distance calculations.
    The starting node is excluded from the result.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")

    if node not in G:
        raise nx.NodeNotFound(f"Node '{node}' is not in the graph")

    graph = G.to_undirected() if G.is_directed() else G

    distances = nx.single_source_shortest_path_length(
        graph,
        source=node,
        cutoff=k,
    )

    return set(distances) - {node}


# ============================================================================
# 4. ALLOWED VARIABLE PAIRS
# ============================================================================

def allowed_pairs(G, k):
    """Return unique unordered node pairs within k graph hops.

    Each pair is represented as a frozenset of two variable names.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")

    graph = G.to_undirected() if G.is_directed() else G

    pairs = set()

    for source, distances in nx.all_pairs_shortest_path_length(
        graph,
        cutoff=k,
    ):
        for target in distances:
            if source != target:
                pairs.add(frozenset((source, target)))

    return pairs
