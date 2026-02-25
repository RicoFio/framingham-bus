from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colors as mpl_colors

from formulations.definition_commons import Bus, Depot, NodeId, School, Stop


class NodeColors(Enum):
    DEPOT = "green"
    STOP = "blue"
    SCHOOL = "red"
    OTHER = "lightgray"


@dataclass
class Route:
    bus: Bus
    start: Depot
    stops: list[Stop]
    end: School | Depot
    color: str
    node_path: list[NodeId] | None = None
    trip_id: str | None = None
    edge_loads: list[float] | None = None


@dataclass
class RoutingSolution:
    network: nx.MultiDiGraph
    routes: list[Route]
    objective: float

    def _node_color(self, node: NodeId) -> str:
        node_type = str(self.network.nodes[node].get("type", "")).strip().lower()
        if node_type == "depot":
            return NodeColors.DEPOT.value
        if node_type == "stop":
            return NodeColors.STOP.value
        if node_type == "school":
            return NodeColors.SCHOOL.value
        return NodeColors.OTHER.value

    def _ordered_route_edges(self, route: Route) -> list[tuple[NodeId, NodeId]]:
        if route.node_path and len(route.node_path) >= 2:
            return [
                (route.node_path[idx], route.node_path[idx + 1])
                for idx in range(len(route.node_path) - 1)
            ]

        stop_ids = [stop.node_id for stop in route.stops]
        if not stop_ids:
            return [(route.start.node_id, route.end.node_id)]

        edges: list[tuple[int, int]] = [(route.start.node_id, stop_ids[0])]
        edges.extend((stop_ids[i], stop_ids[i + 1]) for i in range(len(stop_ids) - 1))
        edges.append((stop_ids[-1], route.end.node_id))
        return edges

    def _draw_base(
        self,
        *,
        node_colors: list[str],
        pos: dict[NodeId, tuple[float, float]],
    ) -> None:
        nx.draw_networkx_nodes(self.network, pos, node_color=node_colors)
        nx.draw_networkx_labels(self.network, pos)
        nx.draw_networkx_edges(
            self.network,
            pos,
            edgelist=list(self.network.edges()),
            edge_color="lightgray",
            width=1.0,
            alpha=0.6,
            arrows=False,
        )

    def _draw_highlighted_edges(
        self,
        *,
        pos: dict[NodeId, tuple[float, float]],
        edge_groups: Iterable[tuple[list[tuple[NodeId, NodeId]], str]],
    ) -> None:
        for edge_list, color in edge_groups:
            if not edge_list:
                continue
            nx.draw_networkx_edges(
                self.network,
                pos,
                edgelist=edge_list,
                edge_color=color,
                width=3.5,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=18,
            )

    def _draw_capacity_weighted_route(
        self,
        *,
        route: Route,
        route_edges: list[tuple[NodeId, NodeId]],
        pos: dict[NodeId, tuple[float, float]],
    ) -> None:
        if not route_edges:
            return

        if not route.edge_loads:
            self._draw_highlighted_edges(
                pos=pos,
                edge_groups=[(route_edges, route.color)],
            )
            return

        loads = [float(load) for load in route.edge_loads]
        if len(loads) < len(route_edges):
            loads.extend([0.0] * (len(route_edges) - len(loads)))
        elif len(loads) > len(route_edges):
            loads = loads[: len(route_edges)]

        vmax = max(float(route.bus.capacity), max(loads, default=0.0))
        norm = mpl_colors.Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.cm.viridis
        edge_colors = [cmap(norm(load)) for load in loads]
        widths = [
            2.5 + 2.5 * (load / max(1.0, float(route.bus.capacity))) for load in loads
        ]

        nx.draw_networkx_edges(
            self.network,
            pos,
            edgelist=route_edges,
            edge_color=edge_colors,
            width=widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
        )
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array(loads)
        colorbar = plt.colorbar(scalar_mappable, ax=plt.gca())
        colorbar.set_label("Students on bus")

    def plot_route(self, route: Route, *, capacity_weighted: bool = False) -> None:
        node_colors = [self._node_color(node) for node in self.network.nodes()]

        route_edges = [
            (u, v)
            for (u, v) in self._ordered_route_edges(route)
            if self.network.has_edge(u, v)
        ]

        pos = nx.spring_layout(self.network, seed=42)
        self._draw_base(node_colors=node_colors, pos=pos)
        if capacity_weighted:
            self._draw_capacity_weighted_route(
                route=route,
                route_edges=route_edges,
                pos=pos,
            )
        else:
            self._draw_highlighted_edges(
                pos=pos, edge_groups=[(route_edges, route.color)]
            )
        plt.show()

    def plot_all_routes(self, *, capacity_weighted: bool = False) -> None:
        if capacity_weighted and len(self.routes) == 1:
            self.plot_route(self.routes[0], capacity_weighted=True)
            return

        node_colors = [self._node_color(node) for node in self.network.nodes()]

        edge_groups: list[tuple[list[tuple[NodeId, NodeId]], str]] = []
        for route in self.routes:
            route_edges = [
                (u, v)
                for (u, v) in self._ordered_route_edges(route)
                if self.network.has_edge(u, v)
            ]
            edge_groups.append((route_edges, route.color))

        pos = nx.spring_layout(self.network, seed=42)
        self._draw_base(node_colors=node_colors, pos=pos)
        self._draw_highlighted_edges(pos=pos, edge_groups=edge_groups)
        plt.show()
