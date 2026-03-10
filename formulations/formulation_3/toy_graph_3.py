from __future__ import annotations

import networkx as nx

from formulations.definition_commons import (
    Bus,
    Depot,
    School,
    Stop,
    Student,
)
from formulations.formulation_2.problem_definition_2 import ProblemDefinitionTwo


def _simple_base_graph() -> nx.DiGraph:
    # Road graph with intersections (4, 5) and service nodes:
    # depot=0, stops={1,2}, school={3}
    road = nx.DiGraph()
    weighted_edges = [
        (0, 1, 2.0),
        (1, 2, 2.0),
        (2, 3, 2.0),
        (3, 4, 2.0),
        (4, 5, 2.0),
        (5, 6, 2.0),
    ]
    for u, v, w in weighted_edges:
        road.add_edge(u, v, distance=w, travel_time=w)
        road.add_edge(v, u, distance=w, travel_time=w)

    # Depots
    road.nodes[0]["type"] = "depot"
    # Pickup stops
    road.nodes[1]["type"] = "stop"
    road.nodes[2]["type"] = "stop"
    road.nodes[4]["type"] = "stop"
    road.nodes[5]["type"] = "stop"
    # Shools
    road.nodes[3]["type"] = "school"
    road.nodes[3]["name"] = "THS"
    road.nodes[6]["type"] = "school"
    road.nodes[6]["name"] = "FHS"

    service_nodes = [
        node_id
        for node_id, data in road.nodes(data=True)
        if data.get("type") in {"depot", "stop", "school"}
    ]

    shortest_dist = {
        src: nx.single_source_dijkstra_path_length(road, src, weight="distance")
        for src in service_nodes
    }
    shortest_time = {
        src: nx.single_source_dijkstra_path_length(road, src, weight="travel_time")
        for src in service_nodes
    }

    reduced = nx.DiGraph()
    for node_id in service_nodes:
        reduced.add_node(node_id, **road.nodes[node_id])

    # Fully-connected reduced graph over {depot, stops, schools}
    for src in service_nodes:
        for dst in service_nodes:
            if src == dst:
                continue
            reduced.add_edge(
                src,
                dst,
                distance=float(shortest_dist[src][dst]),
                travel_time=float(shortest_time[src][dst]),
            )

    return reduced


def _build_toy_digraph() -> nx.DiGraph:
    return _simple_base_graph()


def build_toy_problem_definition_two() -> ProblemDefinitionTwo:
    graph = _build_toy_digraph()

    max_rounds = 2
    depot = Depot(node_id=0)

    buses = [Bus(id="bus_0", range=1_000.0, capacity=30, depot=depot)]
    stop_by_node = {
        node_id: Stop(node_id=node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "stop"
    }
    school_by_node = {
        node_id: School(name=str(data["name"]), node_id=node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "school"
    }

    students = [
        Student(
            id="stu_1",
            school=school_by_node[3],  # THS Node 3
            is_monitor=True,
            is_sped=False,
            stop=stop_by_node[1],  # Node 2
        ),
        Student(
            id="stu_2",
            school=school_by_node[6],  # FHS Node 6
            is_monitor=False,
            is_sped=True,
            stop=stop_by_node[2],  # Node 4
        ),
    ]

    problem = ProblemDefinitionTwo(
        graph=graph,
        Q_max=max_rounds,
        alpha=0.3,
        beta=0.5,
        H_ride=60.0,
        phi=1.0,
        eps=0.0,
        B=buses,
        M=students,
        h_s={3: 480.0, 6: 500.0},
        delta_s={3: 10.0, 6: 10.0},
        T_bar=0.0,
    )
    problem.build_service_graph()
    problem.T_bar = problem.compute_default_time_horizon()
    return problem


def build_toy_problem_definition_one() -> ProblemDefinitionTwo:
    # Backward-compatible alias while transitioning naming to formulation 2.
    return build_toy_problem_definition_two()


TOY_PROBLEM = build_toy_problem_definition_two()
