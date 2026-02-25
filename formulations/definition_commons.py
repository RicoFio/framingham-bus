from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

NodeId = Hashable
Arc = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Node:
    node_id: NodeId
    x: float | None = field(default=None, kw_only=True)
    y: float | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class Depot(Node):
    pass


@dataclass(frozen=True, slots=True)
class Bus:
    id: str
    range: float  # in meters
    capacity: int  # in number of students
    # speed is ignored as we assume all buses can drive at urban speeds
    depot: Depot


@dataclass(frozen=True, slots=True)
class School(Node):
    name: str


@dataclass(frozen=True, slots=True)
class Stop(Node):
    pass


@dataclass(frozen=True, slots=True)
class Student:
    id: str
    school: School  # Destination school
    is_monitor: bool  # Student that supervises SPEDs
    is_sped: bool  # Disability status
    stop: Stop  # Stop assigned to student


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _read_pickle_graph(path: Path) -> nx.Graph:
    if path.suffix.lower() not in {".pickle", ".pkl", ".gpickle"}:
        raise ValueError(
            "Graph file must be a pickle file: .pickle, .pkl, or .gpickle."
        )

    with path.open("rb") as f:
        graph = pickle.load(f)

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{path} does not contain a NetworkX graph object.")

    return graph


def _to_digraph_with_edge_attrs(graph: nx.Graph) -> nx.DiGraph:
    def _validate_edge_attrs(
        u: NodeId,
        v: NodeId,
        data: Mapping[str, Any],
    ) -> tuple[float, float]:
        if "travel_time" not in data or "distance" not in data:
            raise ValueError(
                f"Edge ({u!r}, {v!r}) is missing required attributes "
                "'travel_time' and/or 'distance'."
            )

        return float(data["travel_time"]), float(data["distance"])

    if graph.is_multigraph():
        out = nx.DiGraph()
        out.add_nodes_from(graph.nodes(data=True))

        for u, v, edge_data in graph.edges(data=True):
            travel_time, distance = _validate_edge_attrs(u, v, edge_data)
            current = out.get_edge_data(u, v)
            if current is None or travel_time < float(current["travel_time"]):
                out.add_edge(u, v, travel_time=travel_time, distance=distance)
    elif graph.is_directed():
        out = nx.DiGraph(graph)
    else:
        out = nx.DiGraph(graph.to_directed())

    for u, v, data in out.edges(data=True):
        travel_time, distance = _validate_edge_attrs(u, v, data)
        data["travel_time"] = travel_time
        data["distance"] = distance

    return out


def _extract_typed_nodes(
    graph: nx.DiGraph,
) -> tuple[list[Stop], list[School], list[Depot]]:
    def _to_optional_float(
        node_id: NodeId,
        data: Mapping[str, Any],
        *,
        key: str,
    ) -> float | None:
        value = data.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Node {node_id!r} has non-numeric {key!r} coordinate: {value!r}."
            ) from exc

    stops: list[Stop] = []
    schools: list[School] = []
    depots: list[Depot] = []

    for node_id, data in graph.nodes(data=True):
        node_type = str(data.get("type", "")).strip().lower()
        x = _to_optional_float(node_id, data, key="x")
        y = _to_optional_float(node_id, data, key="y")
        if node_type == "stop":
            stops.append(Stop(node_id=node_id, x=x, y=y))
        elif node_type == "school":
            school_name = str(data.get("name", data.get("school_name", node_id)))
            schools.append(School(name=school_name, node_id=node_id, x=x, y=y))
        elif node_type == "depot":
            depots.append(Depot(node_id=node_id, x=x, y=y))

    stops.sort(key=lambda stop: str(stop.node_id))
    schools.sort(key=lambda school: str(school.node_id))
    depots.sort(key=lambda depot: str(depot.node_id))
    return stops, schools, depots


def _load_students_csv(
    csv_path: Path,
    *,
    stops_by_node: Mapping[NodeId, Stop],
    schools_by_node: Mapping[NodeId, School],
) -> list[Student]:
    students: list[Student] = []
    required_columns = {"id", "stop_id", "school_id", "is_monitor", "is_sped"}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} does not contain a header row.")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{csv_path} is missing required student columns: {missing_str}."
            )

        for row_idx, row in enumerate(reader, start=2):
            student_id = str(row["id"]).strip()
            if not student_id:
                raise ValueError(f"Row {row_idx}: student id is empty.")

            try:
                stop_node = int(str(row["stop_id"]).strip())
            except ValueError as exc:
                raise ValueError(f"Row {row_idx}: stop_id must be an integer.") from exc
            if stop_node not in stops_by_node:
                raise ValueError(
                    f"Row {row_idx}: stop_id={stop_node!r} is not in stop set P."
                )
            stop = stops_by_node[stop_node]

            try:
                school_node = int(str(row["school_id"]).strip())
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_idx}: school_id must be an integer."
                ) from exc
            if school_node not in schools_by_node:
                raise ValueError(
                    f"Row {row_idx}: school_id={school_node!r} is not in school set S."
                )
            school = schools_by_node[school_node]

            is_monitor = _to_bool(row["is_monitor"])
            is_sped = _to_bool(row["is_sped"])

            students.append(
                Student(
                    id=student_id,
                    school=school,
                    is_monitor=is_monitor,
                    is_sped=is_sped,
                    stop=stop,
                )
            )

    return students


def _build_school_value_map(
    schools: Iterable[School],
    *,
    school_values: Mapping[NodeId, float] | None,
    default_value: float,
) -> dict[NodeId, float]:
    output = {school.node_id: float(default_value) for school in schools}
    if not school_values:
        return output

    for key, value in school_values.items():
        if key not in output:
            raise ValueError(
                f"Unknown school key {key!r}. Keys must be school node IDs."
            )
        output[key] = float(value)

    return output
