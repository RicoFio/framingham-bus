"""
Formulation 2 allows trip chaining: each bus can execute multiple sequential rounds.
"""

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from formulations.formulation_2.toy_graph_2 import build_toy_problem_definition_two
from formulations.solution import Route, RoutingSolution


def build_model_from_definition(problem, formulation_name):

    # Sets and indices
    P = [stop.node_id for stop in problem.P]
    S = [school.node_id for school in problem.S]
    S_plus = list(problem.S_plus)
    D_plus = list(problem.D_plus)
    D_minus = list(problem.D_minus)
    N = list(problem.N)
    A = list(problem.A)
    Q = list(problem.Q)

    B = [bus.id for bus in problem.B]
    M = [student.id for student in problem.M]
    F = [student.id for student in problem.F]
    K = [student.id for student in problem.K]

    student_by_id = {student.id: student for student in problem.M}
    bus_by_id = {bus.id: bus for bus in problem.B}

    # Instance data / parameters
    p_m_node = {m_id: student_by_id[m_id].stop.node_id for m_id in M}
    s_m_node = {m_id: student_by_id[m_id].school.node_id for m_id in M}
    p_m_label = {m_id: f"p:{p_m_node[m_id]}" for m_id in M}
    s_m_label = {m_id: f"s:{s_m_node[m_id]}" for m_id in M}

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

    # Helper mappings
    pickup_nodes = [f"p:{p}" for p in P]
    school_nodes = [f"s:{s}" for s in S]

    d_plus_by_depot = {problem.service_node_to_road_node[n]: n for n in D_plus}
    d_minus_by_depot = {problem.service_node_to_road_node[n]: n for n in D_minus}
    school_start_by_school = dict(problem.school_to_start_copy)

    q_first = min(Q)
    q_last = max(Q)
    q_prev = {Q[idx]: Q[idx - 1] for idx in range(1, len(Q))}
    q_next = {Q[idx]: Q[idx + 1] for idx in range(len(Q) - 1)}
    Q_without_first = [q for q in Q if q != q_first]
    Q_without_last = [q for q in Q if q != q_last]

    out_arcs = {}
    in_arcs = {}
    for i, j in A:
        out_arcs.setdefault(i, []).append((i, j))
        in_arcs.setdefault(j, []).append((i, j))

    students_by_pickup_label = {}
    students_by_school_label = {}
    students_by_school_node = {}
    for m in M:
        students_by_pickup_label.setdefault(p_m_label[m], []).append(m)
        students_by_school_label.setdefault(s_m_label[m], []).append(m)
        students_by_school_node.setdefault(s_m_node[m], []).append(m)

    # Build model
    model = gp.Model(formulation_name)

    # Decision variables
    z = model.addVars(B, vtype=GRB.BINARY, name="z")

    zq_keys = [(b_id, q) for b_id in B for q in Q]
    z_q = model.addVars(zq_keys, vtype=GRB.BINARY, name="z_q")

    x_keys = [(b_id, q, i, j) for b_id in B for q in Q for (i, j) in A]
    x = model.addVars(x_keys, vtype=GRB.BINARY, name="x")

    v_keys = [(b_id, q, i) for b_id in B for q in Q for i in N]
    v = model.addVars(v_keys, vtype=GRB.BINARY, name="v")

    a_keys = [(m_id, b_id, q) for m_id in M for b_id in B for q in Q]
    a = model.addVars(a_keys, vtype=GRB.BINARY, name="a")

    T = model.addVars(v_keys, lb=0.0, ub=T_bar, vtype=GRB.CONTINUOUS, name="T")

    L_ub = {(b_id, q, i): C_b[b_id] for b_id in B for q in Q for i in N}
    L = model.addVars(v_keys, lb=0.0, ub=L_ub, vtype=GRB.INTEGER, name="L")

    e_keys = [(b_id, q, s) for b_id in B for q in Q for s in S]
    e = model.addVars(e_keys, vtype=GRB.BINARY, name="e")

    r_mon_keys = [(b_id, q) for b_id in B for q in Q]
    r_mon = model.addVars(r_mon_keys, vtype=GRB.BINARY, name="r_mon")

    w_keys = [(m_id, b_id, q) for m_id in K for b_id in B for q in Q]
    w = model.addVars(w_keys, vtype=GRB.BINARY, name="w")

    T_mon = model.addVars(
        r_mon_keys, lb=0.0, ub=T_bar, vtype=GRB.CONTINUOUS, name="T_mon"
    )

    #############
    # Constraints
    #############

    # -------------------------
    # Student assignment (all served; phi=1)
    # -------------------------
    model.addConstrs(
        (gp.quicksum(a[m, b, q] for b in B for q in Q) <= 1 for m in M),
        name="assign_at_most_one_bus_round",
    )
    model.addConstr(
        gp.quicksum(a[m, b, q] for m in M for b in B for q in Q) >= phi * len(M),
        name="assign_minimum_coverage",
    )
    model.addConstrs(
        (a[m, b, q] <= z_q[b, q] for m in M for b in B for q in Q),
        name="assign_implies_round_use",
    )
    model.addConstrs(
        (z_q[b, q] <= gp.quicksum(a[m, b, q] for m in M) for b in B for q in Q),
        name="round_use_implies_some_assignment",
    )
    model.addConstrs(
        (z[b] <= gp.quicksum(a[m, b, q] for m in M for q in Q) for b in B),
        name="bus_use_implies_some_assignment",
    )

    # -------------------------
    # Routing / tour structure
    # -------------------------
    for b in B:
        d_plus_b = d_plus_by_depot[d_b[b]]
        model.addConstr(
            gp.quicksum(x[b, q_first, i, j] for (i, j) in out_arcs.get(d_plus_b, []))
            == z_q[b, q_first],
            name=f"route_start_from_depot_round1_{b}",
        )

    model.addConstrs(
        (
            gp.quicksum(
                x[b, q, i, j] for (i, j) in out_arcs.get(d_plus_by_depot[d_b[b]], [])
            )
            == 0
            for b in B
            for q in Q_without_first
        ),
        name="route_no_depot_start_after_round1",
    )

    model.addConstrs(
        (
            gp.quicksum(e[b, q, s] for s in S) == z_q[b, q_next[q]]
            for b in B
            for q in Q_without_last
        ),
        name="route_round_end_school_select",
    )
    model.addConstrs(
        (gp.quicksum(e[b, q_last, s] for s in S) == 0 for b in B),
        name="route_last_round_no_school_end",
    )

    model.addConstrs(
        (
            gp.quicksum(
                x[b, q_first, i, j]
                for (i, j) in out_arcs.get(school_start_by_school[s], [])
            )
            == 0
            for b in B
            for s in S
        ),
        name="route_no_school_start_round1",
    )
    model.addConstrs(
        (
            gp.quicksum(
                x[b, q, i, j] for (i, j) in out_arcs.get(school_start_by_school[s], [])
            )
            == e[b, q_prev[q], s]
            for b in B
            for q in Q_without_first
            for s in S
        ),
        name="route_school_start_from_prev_round",
    )
    model.addConstrs(
        (
            gp.quicksum(
                x[b, q, i, j] for (i, j) in in_arcs.get(school_start_by_school[s], [])
            )
            == 0
            for b in B
            for q in Q
            for s in S
        ),
        name="route_no_incoming_school_start_copy",
    )

    model.addConstrs(
        (
            gp.quicksum(
                x[b, q, i, j] for (i, j) in in_arcs.get(d_minus_by_depot[d_b[b]], [])
            )
            == z_q[b, q] - z_q[b, q_next[q]]
            for b in B
            for q in Q_without_last
        ),
        name="route_return_depot_only_last_used_round",
    )
    model.addConstrs(
        (
            gp.quicksum(
                x[b, q_last, i, j]
                for (i, j) in in_arcs.get(d_minus_by_depot[d_b[b]], [])
            )
            == z_q[b, q_last]
            for b in B
        ),
        name="route_return_depot_last_round",
    )

    model.addConstrs(
        (
            gp.quicksum(x[b, q, i, j] for (i, j) in out_arcs.get(f"p:{p}", []))
            == v[b, q, f"p:{p}"]
            for b in B
            for q in Q
            for p in P
        ),
        name="flow_pickup_out_equals_visit",
    )
    model.addConstrs(
        (
            gp.quicksum(x[b, q, i, j] for (i, j) in in_arcs.get(f"p:{p}", []))
            == v[b, q, f"p:{p}"]
            for b in B
            for q in Q
            for p in P
        ),
        name="flow_pickup_in_equals_visit",
    )

    model.addConstrs(
        (
            gp.quicksum(x[b, q, i, j] for (i, j) in in_arcs.get(f"s:{s}", []))
            == v[b, q, f"s:{s}"]
            for b in B
            for q in Q
            for s in S
        ),
        name="flow_school_in_equals_visit",
    )
    model.addConstrs(
        (
            gp.quicksum(x[b, q, i, j] for (i, j) in out_arcs.get(f"s:{s}", []))
            == v[b, q, f"s:{s}"] - e[b, q, s]
            for b in B
            for q in Q
            for s in S
        ),
        name="flow_school_out_with_round_end",
    )
    model.addConstrs(
        (
            v[b, q, i] <= z_q[b, q]
            for b in B
            for q in Q
            for i in pickup_nodes + school_nodes
        ),
        name="visit_le_round_use",
    )

    model.addConstrs(
        (a[m, b, q] <= v[b, q, p_m_label[m]] for m in M for b in B for q in Q),
        name="assign_implies_pickup_visit",
    )
    model.addConstrs(
        (a[m, b, q] <= v[b, q, s_m_label[m]] for m in M for b in B for q in Q),
        name="assign_implies_school_visit",
    )
    model.addConstrs(
        (
            v[b, q, p_label]
            <= gp.quicksum(
                a[m, b, q] for m in students_by_pickup_label.get(p_label, [])
            )
            for b in B
            for q in Q
            for p_label in pickup_nodes
        ),
        name="pickup_visit_implies_assigned_student",
    )

    # -------------------------
    # Time anchoring (planning horizon origin at depot copies)
    # -------------------------
    for b in B:
        d_plus_b = d_plus_by_depot[d_b[b]]
        model.addConstr(
            T[b, q_first, d_plus_b] == 0.0, name=f"time_anchor_round1_depot_{b}"
        )

    model.addConstrs(
        (
            T[b, q, school_start_by_school[s]]
            >= T[b, q_prev[q], f"s:{s}"]
            + beta
            * gp.quicksum(
                a[m, b, q_prev[q]] for m in students_by_school_node.get(s, [])
            )
            - M_time * (1 - e[b, q_prev[q], s])
            for b in B
            for q in Q_without_first
            for s in S
        ),
        name="time_round_to_round_chain",
    )

    # -------------------------
    # Time propagation with explicit dwell times
    # -------------------------
    model.addConstrs(
        (
            T[b, q, j]
            >= T[b, q, i]
            + t_ij[(i, j)]
            + alpha
            * gp.quicksum(a[m, b, q] for m in students_by_pickup_label.get(i, []))
            + beta
            * gp.quicksum(a[m, b, q] for m in students_by_school_label.get(i, []))
            - M_time * (1 - x[b, q, i, j])
            for b in B
            for q in Q
            for (i, j) in A
        ),
        name="time_propagation",
    )

    # -------------------------
    # School latest-arrival constraints
    # -------------------------
    model.addConstrs(
        (
            T[b, q, f"s:{s}"]
            + beta * gp.quicksum(a[m, b, q] for m in students_by_school_node.get(s, []))
            <= ell_s[s] + M_time * (1 - v[b, q, f"s:{s}"])
            for b in B
            for q in Q
            for s in S
        ),
        name="time_school_latest_arrival",
    )

    # -------------------------
    # Pickup-before-dropoff and max ride-time
    # -------------------------
    model.addConstrs(
        (
            T[b, q, s_m_label[m]]
            >= T[b, q, p_m_label[m]] + eps - M_time * (1 - a[m, b, q])
            for b in B
            for m in M
            for q in Q
        ),
        name="time_pickup_precedes_dropoff",
    )
    model.addConstrs(
        (
            T[b, q, s_m_label[m]] - T[b, q, p_m_label[m]]
            <= H_ride + M_time * (1 - a[m, b, q])
            for b in B
            for m in M
            for q in Q
        ),
        name="time_max_ride",
    )

    # -------------------------
    # Distance-range constraint (across all rounds)
    # -------------------------
    model.addConstrs(
        (
            gp.quicksum(d_ij[(i, j)] * x[b, q, i, j] for q in Q for (i, j) in A)
            <= R_b[b] * z[b]
            for b in B
        ),
        name="range_across_rounds",
    )

    # -------------------------
    # Load (capacity) constraints (per round)
    # -------------------------
    for b in B:
        d_plus_b = d_plus_by_depot[d_b[b]]
        model.addConstr(L[b, q_first, d_plus_b] == 0.0, name=f"load_start_round1_{b}")

    model.addConstrs(
        (L[b, q, d_minus_by_depot[d_b[b]]] == 0.0 for b in B for q in Q),
        name="load_end_depot_zero",
    )
    model.addConstrs(
        (L[b, q, school_start_by_school[s]] == 0.0 for b in B for q in Q for s in S),
        name="load_school_start_copy_zero",
    )

    model.addConstrs(
        (
            L[b, q, f"s:{s}"] <= C_b[b] * (1 - e[b, q, s])
            for b in B
            for q in Q
            for s in S
        ),
        name="load_empty_if_round_ends_at_school",
    )

    model.addConstrs(
        (
            L[b, q, j]
            >= L[b, q, i]
            + gp.quicksum(a[m, b, q] for m in students_by_pickup_label.get(j, []))
            - gp.quicksum(a[m, b, q] for m in students_by_school_label.get(j, []))
            - M_cap * (1 - x[b, q, i, j])
            for b in B
            for q in Q
            for (i, j) in A
        ),
        name="load_propagation_lb",
    )
    model.addConstrs(
        (
            L[b, q, j]
            <= L[b, q, i]
            + gp.quicksum(a[m, b, q] for m in students_by_pickup_label.get(j, []))
            - gp.quicksum(a[m, b, q] for m in students_by_school_label.get(j, []))
            + M_cap * (1 - x[b, q, i, j])
            for b in B
            for q in Q
            for (i, j) in A
        ),
        name="load_propagation_ub",
    )
    model.addConstrs(
        (L[b, q, i] <= C_b[b] for b in B for q in Q for i in N),
        name="load_cap",
    )
    model.addConstrs(
        (L[b, q, i] >= 0 for b in B for q in Q for i in N),
        name="load_noneg",
    )

    # -------------------------
    # Monitor feasibility constraints (per round)
    # -------------------------
    model.addConstrs(
        (a[m, b, q] <= r_mon[b, q] for b in B for q in Q for m in F),
        name="monitor_flagged_implies_monitor",
    )
    model.addConstrs(
        (r_mon[b, q] <= gp.quicksum(a[m, b, q] for m in F) for b in B for q in Q),
        name="monitor_only_if_flagged_present",
    )

    model.addConstrs(
        (gp.quicksum(w[m, b, q] for m in K) == r_mon[b, q] for b in B for q in Q),
        name="monitor_select_one_if_needed",
    )
    model.addConstrs(
        (w[m, b, q] <= a[m, b, q] for b in B for q in Q for m in K),
        name="monitor_must_be_assigned_student",
    )

    model.addConstrs(
        (
            T_mon[b, q] <= T[b, q, p_m_label[m]] + M_time * (1 - w[m, b, q])
            for b in B
            for q in Q
            for m in K
        ),
        name="monitor_board_time_ub",
    )
    model.addConstrs(
        (
            T_mon[b, q] >= T[b, q, p_m_label[m]] - M_time * (1 - w[m, b, q])
            for b in B
            for q in Q
            for m in K
        ),
        name="monitor_board_time_lb",
    )

    model.addConstrs(
        (
            T_mon[b, q] + eps <= T[b, q, p_m_label[m]] + M_time * (1 - a[m, b, q])
            for b in B
            for q in Q
            for m in F
        ),
        name="monitor_precedes_flagged_pickup",
    )

    # -------------------------
    # Variable domains (explicit time bounds)
    # -------------------------
    model.addConstrs(
        (T[b, q, i] <= T_bar for b in B for q in Q for i in N), name="domain_T_ub"
    )
    model.addConstrs(
        (T[b, q, i] >= 0.0 for b in B for q in Q for i in N), name="domain_T_lb"
    )
    model.addConstrs(
        (T_mon[b, q] <= T_bar for b in B for q in Q), name="domain_T_mon_ub"
    )
    model.addConstrs((T_mon[b, q] >= 0.0 for b in B for q in Q), name="domain_T_mon_lb")

    #############
    # Objective
    #############
    round_tiebreak_weight = 1e-4
    model.setObjective(
        gp.quicksum(d_ij[(i, j)] * x[b, q, i, j] for b in B for q in Q for (i, j) in A)
        + round_tiebreak_weight * gp.quicksum(z_q[b, q] for b in B for q in Q),
        GRB.MINIMIZE,
    )

    model._routing_data = {
        "problem": problem,
        "B": B,
        "M": M,
        "Q": Q,
        "z": z,
        "z_q": z_q,
        "x": x,
        "A": A,
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
    Q = routing_data["Q"]
    z = routing_data["z"]
    z_q = routing_data["z_q"]
    x = routing_data["x"]
    A = routing_data["A"]
    a = routing_data["a"]
    T = routing_data["T"]
    L = routing_data["L"]
    bus_by_id = routing_data["bus_by_id"]
    student_by_id = routing_data["student_by_id"]

    stop_by_id = {stop.node_id: stop for stop in problem.P}
    school_by_id = {school.node_id: school for school in problem.S}
    d_plus_by_depot = {problem.service_node_to_road_node[n]: n for n in problem.D_plus}
    q_first = min(Q)

    route_colors = [
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]
    routes: list[Route] = []
    route_idx = 0

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

    for b in B:
        if z[b].X < 0.5:
            continue

        bus = bus_by_id[b]
        for q in Q:
            if z_q[b, q].X < 0.5:
                continue

            active_arcs = [(i, j) for (i, j) in A if x[b, q, i, j].X > 0.5]
            if not active_arcs:
                continue

            preferred_start = (
                d_plus_by_depot[bus.depot.node_id]
                if q == q_first
                else next(
                    (i for (i, _) in active_arcs if i.startswith("s+:")),
                    active_arcs[0][0],
                )
            )
            service_path = _ordered_service_path(active_arcs, preferred_start)
            if not service_path:
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
                edge_loads.append(float(L[b, q, source_label].X))

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
                    trip_id=f"{b}-q{q}",
                    edge_loads=edge_loads,
                )
            )
            route_idx += 1

    return RoutingSolution(
        network=nx.MultiDiGraph(problem.graph),
        routes=routes,
        objective=float(model.ObjVal),
    )


if __name__ == "__main__":
    try:
        definition = build_toy_problem_definition_two()
        formulation_name = "formulation_2_toy"
        model = build_model_from_definition(definition, formulation_name)
        model.update()

        print(f"Model name: {model.ModelName}")
        print(f"Decision variables created: {model.NumVars}")

        model.optimize()
        if model.Status == GRB.OPTIMAL:
            solution = build_routing_solution_from_model(model)
            print(f"Objective value: {solution.objective}")
            for idx, route in enumerate(solution.routes, start=1):
                route_label = route.trip_id or f"route_{idx}"
                print(f"Plotting {route_label}")
                solution.plot_route(route)

    except gp.GurobiError as e:
        print(f"Error: {e}")
