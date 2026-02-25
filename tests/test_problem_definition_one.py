import csv
import pickle
from pathlib import Path

import networkx as nx

from formulations.graph import simplest_demo_graph_undirected
from formulations.problem import ProblemDefinitionOne


def _write_toy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    base_graph = simplest_demo_graph_undirected()

    # Build a fully connected directed graph with the required edge attributes.
    graph = nx.DiGraph()
    graph.add_nodes_from(base_graph.nodes())
    nx.set_node_attributes(
        graph, {0: "depot", 1: "stop", 2: "stop", 3: "school"}, "type"
    )
    nx.set_node_attributes(graph, {3: "Toy School"}, "name")

    shortest = dict(nx.all_pairs_dijkstra_path_length(base_graph, weight="weight"))
    for u in base_graph.nodes():
        for v in base_graph.nodes():
            if u == v:
                continue
            weight = float(shortest[u][v])
            graph.add_edge(u, v, travel_time=weight, distance=weight)

    graph_path = tmp_path / "toy_graph.pkl"
    with graph_path.open("wb") as f:
        pickle.dump(graph, f)

    students_path = tmp_path / "students.csv"
    with students_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "stop_id", "school_id", "is_monitor", "is_sped"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "stu_1",
                "stop_id": 1,
                "school_id": 3,
                "is_monitor": "true",
                "is_sped": "false",
            }
        )
        writer.writerow(
            {
                "id": "stu_2",
                "stop_id": 2,
                "school_id": 3,
                "is_monitor": "false",
                "is_sped": "true",
            }
        )

    return graph_path, students_path


def test_problem_definition_one_from_files_with_toy_graph(tmp_path: Path) -> None:
    graph_path, students_path = _write_toy_inputs(tmp_path)

    problem = ProblemDefinitionOne.from_files(
        graph_path=graph_path,
        students_csv_path=students_path,
        Q_max=30,
        school_bell_times={3: 480.0},
        school_slacks={3: 10.0},
    )

    assert len(problem.P) == 2
    assert len(problem.S) == 1
    assert len(problem.D) == 1
    assert len(problem.M) == 2
    assert len(problem.K) == 1
    assert len(problem.F) == 1
    assert problem.M[0].stop.node_id == 1
    assert problem.M[1].stop.node_id == 2

    assert len(problem.N) == 5
    assert len(problem.A) == 13
    assert problem.t_ij[("p:1", "s:3")] == 1.0
    assert problem.d_ij[("p:1", "s:3")] == 1.0

    assert problem.l_s[3] == 470.0
    assert problem.M_time > 0.0
