"""Association graph.

Paper: Section 3.2. Node types: sensor measurements, derived energy-efficiency metrics,
environment/control variables. Edge types: physical connection, thermodynamic coupling,
control dependency. Distances are used undirected for the k-hop constraint.

Owner: Member A
"""


def build_graph(variables):
    raise NotImplementedError


def k_hop_neighbourhood(G, node, k):
    raise NotImplementedError


def allowed_pairs(G, k):
    raise NotImplementedError
