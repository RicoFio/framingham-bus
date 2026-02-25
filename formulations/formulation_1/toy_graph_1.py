from __future__ import annotations

import networkx as nx

from formulations.definition_commons import (
    Bus,
    Depot,
    School,
    Stop,
    Student,
)
from formulations.formulation_1.problem_definition_1 import ProblemDefinitionOne


def simplest_demo_graph_undirected():
    G = nx.Graph()
    G.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
        weight=5,
    )
    # Adapt the weights of the diagonals to make it interesting
    G.edges[0, 3]["weight"] = 10
    G.edges[2, 3]["weight"] = 10
    G.edges[1, 2]["weight"] = 10
    G.edges[0, 2]["weight"] = 1
    G.edges[1, 3]["weight"] = 1

    return G


def _build_toy_digraph() -> nx.DiGraph:
    base_graph = simplest_demo_graph_undirected()

    graph = nx.DiGraph()
    graph.add_nodes_from(base_graph.nodes())

    for u, v, data in base_graph.edges(data=True):
        weight = float(data["weight"])
        graph.add_edge(u, v, travel_time=weight, distance=weight)
        graph.add_edge(v, u, travel_time=weight, distance=weight)

    graph.nodes[0]["type"] = "depot"
    graph.nodes[1]["type"] = "stop"
    graph.nodes[2]["type"] = "stop"
    graph.nodes[3]["type"] = "school"
    graph.nodes[3]["name"] = "Toy School"
    return graph


def build_toy_problem_definition_one() -> ProblemDefinitionOne:
    graph = _build_toy_digraph()

    depot = Depot(node_id=0)
    stops = [Stop(node_id=1), Stop(node_id=2)]
    school = School(name="Toy School", node_id=3)

    buses = [Bus(id="bus_0", range=1_000.0, capacity=30, depot=depot)]
    students = [
        Student(
            id="stu_1",
            school=school,
            is_monitor=True,
            is_sped=False,
            stop=stops[0],
        ),
        Student(
            id="stu_2",
            school=school,
            is_monitor=False,
            is_sped=True,
            stop=stops[1],
        ),
    ]

    problem = ProblemDefinitionOne(
        graph=graph,
        Q_max=30,
        alpha=0.3,
        beta=0.5,
        H_ride=60.0,
        phi=1.0,
        eps=0.0,
        P=stops,
        S=[school],
        D=[depot],
        B=buses,
        M=students,
        h_s={3: 480.0},
        delta_s={3: 10.0},
        T_bar=0.0,
    )
    problem.build_service_graph()
    problem.T_bar = problem.compute_default_time_horizon()
    return problem


TOY_PROBLEM = build_toy_problem_definition_one()
