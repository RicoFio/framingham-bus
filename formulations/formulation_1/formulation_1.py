"""
This formulation allows us to route each bus *once* from the depot to a school. Thus, there is no notion of trip chaining or repeated trips between schools. Once the bus has picked up students and dropped them off at their respective shool, the bus' trip ends there.

"""

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from formulations.formulation_1.toy_graph_1 import build_toy_problem_definition_one
from formulations.solution import Route, RoutingSolution


def build_model_from_definition(problem, formulation_name):

    # Sets and indices
    P = [stop.node_id for stop in problem.P]
    S = [school.node_id for school in problem.S]
    D_plus = list(problem.D_plus)
    D_minus = list(problem.D_minus)
    N = list(problem.N)
    A = list(problem.A)

    B = [bus.id for bus in problem.B]
    M = [student.id for student in problem.M]
    F = [student.id for student in problem.F]
    K = [student.id for student in problem.K]

    student_by_id = {student.id: student for student in problem.M}
    bus_by_id = {bus.id: bus for bus in problem.B}

    # Instance data / parameters
    p_m = {m_id: student_by_id[m_id].stop.node_id for m_id in M}
    s_m = {m_id: student_by_id[m_id].school.node_id for m_id in M}

    d_b = {b_id: bus_by_id[b_id].depot.node_id for b_id in B}
    C_b = {b_id: int(bus_by_id[b_id].capacity) for b_id in B}
    R_b = {b_id: float(bus_by_id[b_id].range) for b_id in B}

    t_ij = dict(problem.t_ij)
    d_ij = dict(problem.d_ij)
    ell_s = dict(problem.l_s)
    T_bar = float(problem.T_bar)

    # Global constants
    alpha = float(problem.alpha)
    beta = float(problem.beta)
    H_ride = float(problem.H_ride)
    phi = float(problem.phi)
    eps = float(problem.eps)
    M_time = float(problem.M_time)
    M_cap = int(problem.M_cap)

    # Build model
    model = gp.Model(formulation_name)

    # Decision variables
    z = model.addVars(B, vtype=GRB.BINARY, name="z")

    x_keys = [(b_id, i, j) for b_id in B for (i, j) in A]
    x = model.addVars(x_keys, vtype=GRB.BINARY, name="x")

    v_keys = [(b_id, i) for b_id in B for i in N]
    v = model.addVars(v_keys, vtype=GRB.BINARY, name="v")

    a_keys = [(m_id, b_id) for m_id in M for b_id in B]
    a = model.addVars(a_keys, vtype=GRB.BINARY, name="a")

    T = model.addVars(v_keys, lb=0.0, ub=T_bar, vtype=GRB.CONTINUOUS, name="T")

    L_ub = {(b_id, i): C_b[b_id] for b_id in B for i in N}
    L = model.addVars(v_keys, lb=0.0, ub=L_ub, vtype=GRB.INTEGER, name="L")

    r_mon = model.addVars(B, vtype=GRB.BINARY, name="r_mon")

    w_keys = [(m_id, b_id) for m_id in K for b_id in B]
    w = model.addVars(w_keys, vtype=GRB.BINARY, name="w")

    T_mon = model.addVars(B, lb=0.0, ub=T_bar, vtype=GRB.CONTINUOUS, name="T_mon")

    #############
    # Constraints
    #############
    # Student assignment constraints
    model.addConstrs(
        (gp.quicksum(a[m, b] for b in B) == 1 for m in M), name="student_bus_assignment"
    )
    model.addConstrs((a[m, b] == z[b] for m in M for b in B), name="bus_usage_flag")
    model.addConstrs(
        (z[b] <= gp.quicksum(a[m, b] for m in M) for b in B), name="ub_bus_usage_flag"
    )
    model.addConstrs(
        (z[b] <= gp.quicksum(a[m, b] for m in M) for b in B), name="ub_bus_usage_flag"
    )
    # Depot management constraints
    ps_nodes = [f"p:{p}" for p in P] + [f"s:{s}" for s in S]
    d_plus_by_depot = {problem.service_node_to_road_node[n]: n for n in D_plus}
    d_minus_by_depot = {problem.service_node_to_road_node[n]: n for n in D_minus}

    for b in B:
        d_plus_b = d_plus_by_depot[d_b[b]]
        d_minus_b = d_minus_by_depot[d_b[b]]

        model.addConstr(
            gp.quicksum(x[b, i, j] for (i, j) in A if i == d_plus_b) == z[b],
            name=f"depot_depart_{b}",
        )
        model.addConstr(
            gp.quicksum(x[b, i, j] for (i, j) in A if j == d_minus_b) == z[b],
            name=f"depot_arrive_{b}",
        )

    model.addConstrs(
        (
            gp.quicksum(x[b, i_from, j] for (i_from, j) in A if i_from == i) == v[b, i]
            for b in B
            for i in ps_nodes
        ),
        name="flow_out_equals_visit",
    )
    model.addConstrs(
        (
            gp.quicksum(x[b, j, i_to] for (j, i_to) in A if i_to == i) == v[b, i]
            for b in B
            for i in ps_nodes
        ),
        name="flow_in_equals_visit",
    )
    model.addConstrs(
        (v[b, i] <= z[b] for b in B for i in ps_nodes), name="visit_le_bus"
    )
    model.addConstrs(
        (a[m, b] <= v[b, f"p:{p_m[m]}"] for m in M for b in B),
        name="assign_implies_pickup_visit",
    )
    model.addConstrs(
        (a[m, b] <= v[b, f"s:{s_m[m]}"] for m in M for b in B),
        name="assign_implies_school_visit",
    )
    # Time constraints
    for b in B:
        d_plus_b = d_plus_by_depot[d_b[b]]
        d_minus_b = d_minus_by_depot[d_b[b]]

        model.addConstr(T[b, d_plus_b] == 0.0, name=f"start_time_zero_{b}")

    model.addConstrs(
        (
            T[b, j]
            >= T[b, i]
            + t_ij[(i, j)]
            + alpha * gp.quicksum(a[m, b] for m in M if i == f"p:{p_m[m]}")
            + beta * gp.quicksum(a[m, b] for m in M if i == f"s:{s_m[m]}")
            - M_time * (1 - x[b, i, j])
            for b in B
            for (i, j) in A
        ),
        name="time_propagation",
    )
    model.addConstrs(
        (
            T[b, f"s:{s}"] + beta * gp.quicksum(a[m, b] for m in M if s_m[m] == s)
            <= ell_s[s] + M_time * (1 - v[b, f"s:{s}"])
            for b in B
            for s in S
        ),
        name="school_latest_arrival",
    )
    model.addConstrs(
        (
            T[b, f"s:{s_m[m]}"] >= T[b, f"p:{p_m[m]}"] + eps - M_time * (1 - a[m, b])
            for b in B
            for m in M
        ),
        name="student_pickup_precedes_dropoff",
    )
    model.addConstrs(
        (
            T[b, f"s:{s_m[m]}"] - T[b, f"p:{p_m[m]}"] <= H_ride + M_time * (1 - a[m, b])
            for b in B
            for m in M
        ),
        name="student_max_ride_time",
    )
    model.addConstrs(
        (
            gp.quicksum(d_ij[(i, j)] * x[b, i, j] for (i, j) in A) <= R_b[b] * z[b]
            for b in B
        ),
        name="bus_range",
    )

    for b in B:
        d_plus_b = d_plus_by_depot[d_b[b]]
        d_minus_b = d_minus_by_depot[d_b[b]]

        model.addConstr(L[b, d_plus_b] == 0.0, name=f"start_load_zero_{b}")
        model.addConstr(L[b, d_minus_b] == 0.0, name=f"end_load_zero_{b}")

    model.addConstrs(
        (
            L[b, j]
            >= L[b, i]
            + gp.quicksum(a[m, b] for m in M if f"p:{p_m[m]}" == j)
            - gp.quicksum(a[m, b] for m in M if f"s:{s_m[m]}" == j)
            - M_cap * (1 - x[b, i, j])
            for b in B
            for (i, j) in A
        ),
        name="load_propagation_lb",
    )
    model.addConstrs(
        (
            L[b, j]
            <= L[b, i]
            + gp.quicksum(a[m, b] for m in M if f"p:{p_m[m]}" == j)
            - gp.quicksum(a[m, b] for m in M if f"s:{s_m[m]}" == j)
            + M_cap * (1 - x[b, i, j])
            for b in B
            for (i, j) in A
        ),
        name="load_propagation_ub",
    )
    model.addConstrs((L[b, i] <= C_b[b] for b in B for i in N), name="load_cap")
    model.addConstrs(
        (a[m, b] <= r_mon[b] for b in B for m in F), name="flagged_need_mon"
    )
    model.addConstrs(
        (r_mon[b] <= gp.quicksum(a[m, b] for m in F) for b in B), name="mon_if_flagged"
    )
    model.addConstrs(
        (gp.quicksum(w[m, b] for m in K) == r_mon[b] for b in B),
        name="select_one_monitor",
    )
    model.addConstrs(
        (w[m, b] <= a[m, b] for b in B for m in K), name="monitor_assigned"
    )
    model.addConstrs(
        (
            T_mon[b] <= T[b, f"p:{p_m[m]}"] + M_time * (1 - w[m, b])
            for b in B
            for m in K
        ),
        name="monitor_time_ub",
    )
    model.addConstrs(
        (
            T_mon[b] >= T[b, f"p:{p_m[m]}"] - M_time * (1 - w[m, b])
            for b in B
            for m in K
        ),
        name="monitor_time_lb",
    )
    model.addConstrs(
        (
            T_mon[b] + eps <= T[b, f"p:{p_m[m]}"] + M_time * (1 - a[m, b])
            for b in B
            for m in F
        ),
        name="monitor_precedes_flagged_pickup",
    )

    #############
    # Objective
    #############
    # Minimize total travel distance
    model.setObjective(
        gp.quicksum(d_ij[(i, j)] * x[b, i, j] for b in B for (i, j) in A),
        GRB.MINIMIZE,
    )

    model._routing_data = {
        "problem": problem,
        "B": B,
        "M": M,
        "A": A,
        "z": z,
        "x": x,
        "a": a,
        "T": T,
        "L": L,
        "bus_by_id": bus_by_id,
        "student_by_id": student_by_id,
    }

    return model


def build_routing_solution_from_model(model: gp.Model) -> RoutingSolution:
    if model.SolCount == 0:
        raise ValueError("Model has no feasible solution to extract.")

    routing_data = model._routing_data
    problem = routing_data["problem"]
    B = routing_data["B"]
    M = routing_data["M"]
    A = routing_data["A"]
    z = routing_data["z"]
    x = routing_data["x"]
    a = routing_data["a"]
    T = routing_data["T"]
    L = routing_data["L"]
    bus_by_id = routing_data["bus_by_id"]
    student_by_id = routing_data["student_by_id"]

    stop_by_id = {stop.node_id: stop for stop in problem.P}
    school_by_id = {school.node_id: school for school in problem.S}
    d_plus_by_depot = {problem.service_node_to_road_node[n]: n for n in problem.D_plus}

    route_colors = [
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]
    routes: list[Route] = []

    def _ordered_service_path(
        active_arcs: list[tuple[str, str]],
        preferred_start: str,
    ) -> list[str]:
        if not active_arcs:
            return []

        succ: dict[str, list[str]] = {}
        indeg: dict[str, int] = {}
        outdeg: dict[str, int] = {}
        for i, j in active_arcs:
            succ.setdefault(i, []).append(j)
            indeg[j] = indeg.get(j, 0) + 1
            outdeg[i] = outdeg.get(i, 0) + 1

        if preferred_start in succ:
            start = preferred_start
        else:
            flow_starts = [
                node for node in succ if outdeg.get(node, 0) > indeg.get(node, 0)
            ]
            start = flow_starts[0] if flow_starts else active_arcs[0][0]

        path = [start]
        used_edges: set[tuple[str, str]] = set()
        current = start
        while current in succ:
            next_node = None
            for candidate in succ[current]:
                edge = (current, candidate)
                if edge not in used_edges:
                    used_edges.add(edge)
                    next_node = candidate
                    break
            if next_node is None:
                break
            path.append(next_node)
            current = next_node
            if len(used_edges) >= len(active_arcs):
                break

        return path

    for route_idx, b in enumerate(B):
        if z[b].X < 0.5:
            continue

        bus = bus_by_id[b]
        active_arcs = [(i, j) for (i, j) in A if x[b, i, j].X > 0.5]
        if not active_arcs:
            continue
        preferred_start = d_plus_by_depot[bus.depot.node_id]
        service_path = _ordered_service_path(active_arcs, preferred_start)
        if not service_path:
            continue

        assigned_students = [student_by_id[m] for m in M if a[m, b].X > 0.5]
        if not assigned_students:
            continue

        raw_node_path = [
            problem.service_node_to_road_node[label] for label in service_path
        ]
        node_path = [raw_node_path[0]]
        edge_loads: list[float] = []
        for idx in range(len(service_path) - 1):
            source_label = service_path[idx]
            source_road = problem.service_node_to_road_node[source_label]
            target_road = problem.service_node_to_road_node[service_path[idx + 1]]
            if source_road == target_road:
                continue
            if node_path[-1] != source_road:
                node_path.append(source_road)
            node_path.append(target_road)
            edge_loads.append(float(L[b, source_label].X))

        if len(node_path) < 2:
            continue

        seen_stops = set()
        stop_ids = []
        for node in node_path:
            if node in stop_by_id and node not in seen_stops:
                stop_ids.append(node)
                seen_stops.add(node)
        stops = [stop_by_id[stop_id] for stop_id in stop_ids]

        end_node = node_path[-1]
        end = school_by_id[end_node] if end_node in school_by_id else bus.depot

        routes.append(
            Route(
                bus=bus,
                start=bus.depot,
                stops=stops,
                end=end,
                color=route_colors[route_idx % len(route_colors)],
                node_path=node_path,
                trip_id=b,
                edge_loads=edge_loads,
            )
        )

    return RoutingSolution(
        network=nx.MultiDiGraph(problem.graph),
        routes=routes,
        objective=float(model.ObjVal),
    )


if __name__ == "__main__":
    try:
        definition = build_toy_problem_definition_one()
        formulation_name = "formulation_1_toy"
        model = build_model_from_definition(definition, formulation_name)
        model.update()

        print(f"Model name: {model.ModelName}")
        print(f"Decision variables created: {model.NumVars}")

        model.optimize()
        if model.Status == GRB.OPTIMAL:
            solution = build_routing_solution_from_model(model)
            print(f"Objective value: {solution.objective}")
            solution.plot_all_routes()

    except gp.GurobiError as e:
        print(f"Error: {e}")
