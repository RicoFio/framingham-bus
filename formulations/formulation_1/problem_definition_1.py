from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import networkx as nx

from formulations.definition_commons import (
    Arc,
    Bus,
    Depot,
    NodeId,
    School,
    Stop,
    Student,
    _build_school_value_map,
    _extract_typed_nodes,
    _load_students_csv,
    _read_pickle_graph,
    _to_digraph_with_edge_attrs,
)


@dataclass(slots=True)
class ProblemDefinitionOne:
    graph: nx.DiGraph  # Directed road network graph with travel metrics on edges.

    Q_max: int  # Global upper bound used for capacity-related sizing.
    alpha: float = 0.3  # Boarding dwell time per student (minutes).
    beta: float = 0.5  # Alighting dwell time per student (minutes).
    H_ride: float = 60.0  # Maximum in-vehicle ride time per student (minutes).
    phi: float = 1.0  # Coverage ratio target (served/requested).
    eps: float = 0.0  # Small precedence/separation constant (minutes).

    # Sets / entity collections
    P: list[Stop] = field(default_factory=list)  # Pickup stops set.
    S: list[School] = field(default_factory=list)  # Schools set.
    D: list[Depot] = field(default_factory=list)  # Physical depots set.
    B: list[Bus] = field(default_factory=list)  # Fleet of buses.
    M: list[Student] = field(default_factory=list)  # Students requesting service.
    F: list[Student] = field(default_factory=list)  # Flagged/SPED students.
    K: list[Student] = field(default_factory=list)  # Monitor-eligible students.

    # Service graph sets and parameters
    D_plus: list[str] = field(default_factory=list)  # Start-depot copy nodes.
    D_minus: list[str] = field(default_factory=list)  # End-depot copy nodes.
    N: list[str] = field(default_factory=list)  # Service-node labels.
    A: list[Arc] = field(default_factory=list)  # Directed service arcs.
    t_ij: dict[Arc, float] = field(default_factory=dict)  # Arc travel times.
    d_ij: dict[Arc, float] = field(default_factory=dict)  # Arc travel distances.
    service_node_to_road_node: dict[str, NodeId] = field(default_factory=dict)  # Map from service-node label to road node id.
    service_node_kind: dict[str, str] = field(default_factory=dict)  # Service-node type tags.

    # School-time parameters
    h_s: dict[NodeId, float] = field(default_factory=dict)  # School bell/start times.
    delta_s: dict[NodeId, float] = field(default_factory=dict)  # Required slack before each school bell.
    T_bar: float = 0.0  # Global time horizon upper bound (also used as Big-M_time).

    def __post_init__(self) -> None:
        if not self.graph.is_directed():
            raise ValueError("ProblemDefinitionOne.graph must be directed.")

        self.F = [student for student in self.M if student.is_sped]
        self.K = [student for student in self.M if student.is_monitor]

        if self.B:
            max_capacity = max(bus.capacity for bus in self.B)
            if max_capacity > self.Q_max:
                raise ValueError(
                    f"Q_max={self.Q_max} is smaller than max bus capacity={max_capacity}."
                )

    @property
    def l_s(self) -> dict[NodeId, float]:
        return {
            school.node_id: self.h_s.get(school.node_id, 0.0)
            - self.delta_s.get(school.node_id, 0.0)
            for school in self.S
        }

    @property
    def M_time(self) -> float:
        return self.T_bar

    @property
    def M_cap(self) -> int:
        if self.B:
            return max(bus.capacity for bus in self.B)
        return self.Q_max

    def compute_default_time_horizon(self, *, extra_buffer: float = 0.0) -> float:
        latest_school_arrival = max(self.l_s.values(), default=0.0)
        max_travel = max(self.t_ij.values(), default=0.0)
        max_dwell = (self.alpha + self.beta) * len(self.M)
        return latest_school_arrival + max_travel + max_dwell + extra_buffer

    def build_service_graph(self) -> None:
        self.D_plus = []
        self.D_minus = []
        self.N = []
        self.A = []
        self.t_ij = {}
        self.d_ij = {}
        self.service_node_to_road_node = {}
        self.service_node_kind = {}

        for stop in self.P:
            label = f"p:{stop.node_id}"
            self.N.append(label)
            self.service_node_to_road_node[label] = stop.node_id
            self.service_node_kind[label] = "stop"

        for school in self.S:
            label = f"s:{school.node_id}"
            self.N.append(label)
            self.service_node_to_road_node[label] = school.node_id
            self.service_node_kind[label] = "school"

        for depot in self.D:
            start_label = f"d+:{depot.node_id}"
            end_label = f"d-:{depot.node_id}"

            self.D_plus.append(start_label)
            self.D_minus.append(end_label)

            self.N.append(start_label)
            self.N.append(end_label)

            self.service_node_to_road_node[start_label] = depot.node_id
            self.service_node_to_road_node[end_label] = depot.node_id
            self.service_node_kind[start_label] = "depot_start"
            self.service_node_kind[end_label] = "depot_end"

        for source in self.N:
            if source in self.D_minus:
                continue

            source_road_node = self.service_node_to_road_node[source]

            for target in self.N:
                if target == source:
                    continue
                if target in self.D_plus:
                    continue

                target_road_node = self.service_node_to_road_node[target]
                arc = (source, target)
                self.A.append(arc)

                if source_road_node == target_road_node:
                    self.t_ij[arc] = 0.0
                    self.d_ij[arc] = 0.0
                    continue

                edge_data = self.graph.get_edge_data(source_road_node, target_road_node)
                if edge_data is None:
                    raise ValueError(
                        f"Missing edge from {source_road_node!r} to {target_road_node!r}. "
                        "Expected a fully-connected graph."
                    )

                self.t_ij[arc] = float(edge_data["travel_time"])
                self.d_ij[arc] = float(edge_data["distance"])

    @classmethod
    def from_files(
        cls,
        *,
        graph_path: str | Path,
        students_csv_path: str | Path,
        Q_max: int,
        buses: Iterable[Bus] | None = None,
        alpha: float = 0.3,
        beta: float = 0.5,
        H_ride: float = 60.0,
        phi: float = 1.0,
        eps: float = 0.0,
        school_bell_times: Mapping[NodeId, float] | None = None,
        school_slacks: Mapping[NodeId, float] | float | None = None,
        default_school_bell_time: float = 0.0,
        default_school_slack: float = 0.0,
        T_bar: float | None = None,
        default_bus_range: float = math.inf,
    ) -> ProblemDefinitionOne:
        graph_file = Path(graph_path)
        students_file = Path(students_csv_path)

        graph = _to_digraph_with_edge_attrs(_read_pickle_graph(graph_file))

        stops, schools, depots = _extract_typed_nodes(graph)
        if not stops:
            raise ValueError(
                "No stop nodes found. Expected node attribute type='stop'."
            )
        if not schools:
            raise ValueError(
                "No school nodes found. Expected node attribute type='school'."
            )
        if not depots:
            raise ValueError(
                "No depot nodes found. Expected node attribute type='depot'."
            )

        stops_by_node = {stop.node_id: stop for stop in stops}
        schools_by_node = {school.node_id: school for school in schools}

        students = _load_students_csv(
            students_file,
            stops_by_node=stops_by_node,
            schools_by_node=schools_by_node,
        )

        bus_list = list(buses) if buses is not None else []
        if not bus_list:
            bus_list = [
                Bus(
                    id="bus_0",
                    range=default_bus_range,
                    capacity=Q_max,
                    depot=depots[0],
                )
            ]

        depot_nodes = {depot.node_id for depot in depots}
        for bus in bus_list:
            if bus.depot.node_id not in depot_nodes:
                raise ValueError(
                    f"Bus {bus.id!r} references depot node {bus.depot.node_id!r}, "
                    "which is not present in graph depot set D."
                )

        h_s = _build_school_value_map(
            schools,
            school_values=school_bell_times,
            default_value=default_school_bell_time,
        )
        if isinstance(school_slacks, (int, float)):
            delta_s = {school.node_id: float(school_slacks) for school in schools}
        else:
            delta_s = _build_school_value_map(
                schools,
                school_values=school_slacks,
                default_value=default_school_slack,
            )

        instance = cls(
            graph=graph,
            Q_max=Q_max,
            alpha=alpha,
            beta=beta,
            H_ride=H_ride,
            phi=phi,
            eps=eps,
            P=stops,
            S=schools,
            D=depots,
            B=bus_list,
            M=students,
            h_s=h_s,
            delta_s=delta_s,
            T_bar=0.0,
        )
        instance.build_service_graph()
        instance.T_bar = (
            float(T_bar)
            if T_bar is not None
            else instance.compute_default_time_horizon()
        )
        return instance
