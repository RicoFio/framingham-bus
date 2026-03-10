from __future__ import annotations

import gurobipy as gp
import networkx as nx
from gurobipy import GRB


def build_toy_network() -> nx.DiGraph:
    graph = nx.DiGraph()
    grid = nx.grid_2d_graph(5, 5)

    for i, j in grid.edges():
        graph.add_edge(i, j, distance=1.0, emission=0.20)
        graph.add_edge(j, i, distance=1.0, emission=0.20)

    return graph


def build_model(
    graph: nx.DiGraph,
    source: tuple[int, int],
    sink: tuple[int, int],
    demand_items: int = 120,
    items_per_truck: int = 40,
    max_trucks: int = 10,
) -> gp.Model:
    arcs = list(graph.edges())
    nodes = list(graph.nodes())

    model = gp.Model("decomposition_test")

    # Route/path decisions.
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    # Fleet + shipping decisions.
    trucks = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_trucks, name="trucks")
    shipped = model.addVar(lb=0.0, ub=float(demand_items), name="shipped_items")
    lost = model.addVar(lb=0.0, ub=float(demand_items), name="lost_items")

    # n[i,j] = trucks * x[i,j] (linearized below) to price truck-miles emissions.
    n = model.addVars(arcs, vtype=GRB.INTEGER, lb=0, ub=max_trucks, name="n")

    # Single source -> sink path constraints (no arc capacities).
    for node in nodes:
        out_flow = gp.quicksum(x[node, j] for _, j in graph.out_edges(node))
        in_flow = gp.quicksum(x[i, node] for i, _ in graph.in_edges(node))

        if node == source:
            model.addConstr(out_flow - in_flow == 1, name=f"flow_source_{node}")
        elif node == sink:
            model.addConstr(out_flow - in_flow == -1, name=f"flow_sink_{node}")
        else:
            model.addConstr(out_flow - in_flow == 0, name=f"flow_bal_{node}")

        model.addConstr(out_flow <= 1, name=f"deg_out_{node}")
        model.addConstr(in_flow <= 1, name=f"deg_in_{node}")

    # Demand accounting.
    model.addConstr(lost == demand_items - shipped, name="loss_definition")
    model.addConstr(
        shipped <= items_per_truck * trucks,
        name="truck_payload_link",
    )

    # Linearize n[i,j] = trucks * x[i,j].
    for i, j in arcs:
        model.addConstr(n[i, j] <= trucks, name=f"n_le_trucks_{i}_{j}")
        model.addConstr(n[i, j] <= max_trucks * x[i, j], name=f"n_le_x_{i}_{j}")
        model.addConstr(
            n[i, j] >= trucks - max_trucks * (1 - x[i, j]),
            name=f"n_ge_link_{i}_{j}",
        )

    distance = gp.quicksum(graph[i][j]["distance"] * x[i, j] for i, j in arcs)
    emissions = gp.quicksum(graph[i][j]["emission"] * n[i, j] for i, j in arcs)

    # Simple weighted objective:
    # minimize route distance + trucks used + unmet demand + emissions.
    model.setObjective(
        1.0 * distance + 2.0 * trucks + 100.0 * lost + 1.0 * emissions,
        GRB.MINIMIZE,
    )

    model._routing_data = {
        "graph": graph,
        "source": source,
        "sink": sink,
        "x": x,
        "trucks": trucks,
        "shipped": shipped,
        "lost": lost,
        "distance_expr": distance,
        "emissions_expr": emissions,
    }
    return model


def extract_path(
    graph: nx.DiGraph,
    x: gp.tupledict,
    source: tuple[int, int],
    sink: tuple[int, int],
) -> list[tuple[int, int]]:
    next_by_node: dict[tuple[int, int], tuple[int, int]] = {}
    for i, j in graph.edges():
        if x[i, j].X > 0.5:
            next_by_node[i] = j

    path = [source]
    current = source
    max_steps = len(graph.nodes())
    for _ in range(max_steps):
        if current == sink:
            break
        if current not in next_by_node:
            break
        current = next_by_node[current]
        path.append(current)
    return path


if __name__ == "__main__":
    toy_graph = build_toy_network()
    origin = (0, 0)
    sink = (4, 4)

    model = build_model(toy_graph, origin, sink)
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        data = model._routing_data
        path = extract_path(toy_graph, data["x"], origin, sink)

        print(f"Objective: {model.ObjVal:.2f}")
        print(f"Path: {path}")
        print(f"Trucks: {data['trucks'].X:.0f}")
        print(f"Items shipped: {data['shipped'].X:.0f}")
        print(f"Items lost: {data['lost'].X:.0f}")
        print(f"Distance term: {data['distance_expr'].getValue():.2f}")
        print(f"Emissions term: {data['emissions_expr'].getValue():.2f}")
    else:
        print(f"No optimal solution. Status code: {model.Status}")
