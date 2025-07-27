import random
import os

import itertools
import pandas as pd
import gurobipy as gp
import pandas as pd
import ast
import numpy as np
import argparse
from gurobipy import Env, Model, GRB, quicksum


LICENSE_PARAMS = {
    "WLSACCESSID": "17b613a5-d70f-4885-94aa-c54cf45cf07a",
    "WLSSECRET": "d053d738-acec-41e3-bd66-bbc460d6b31e",
    "LICENSEID": 2654947,
}


def setup():

    print("in the else")
    exam_map_df = pd.read_csv("exam_map.csv")
    exam_map = {}
    for r in exam_map_df.iterrows():
        # print("r", r)
        exam_map[r[1]["Unnamed: 0"]] = r[1]["num"]
        # exam_map = dict(
        #    exam_map_df
        # )  # zip(list(exam_map_df.index), list(exam_map_df["num"])))
    return exam_map


def simulate_triplet_coenrol_simple(
    n_courses: int,
    t_co_file: str,
    m_samples: int,
    random_seed: int = None,
    output_file: str = "sim_t_co.csv",
):
    # 0) reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    # 1) load your historical triples (columns: a,b,c,co)
    df = pd.read_csv(t_co_file)

    # 2) build & filter the `(i,j,k) -> co` lookup
    df["trip"] = df[["a", "b", "c"]].apply(lambda row: tuple(sorted(row)), axis=1)
    df = df[df["trip"].apply(lambda t: t[2] < n_courses)]  # keep max index < n_courses
    hist = dict(zip(df["trip"], df["co"]))

    # 3) initialize counts for every unique triple in 0..n_courses-1
    counts = {
        trip: hist.get(trip, 0)
        for trip in itertools.combinations(range(0, n_courses), 3)
    }

    # 4) add m_samples uniform random draws
    for _ in range(m_samples):
        i, j, k = sorted(random.sample(range(0, n_courses), 3))
        counts[(i, j, k)] += 1

    # 5) output only non-zero
    rows = [
        {"course1": i, "course2": j, "course3": k, "count": c}
        for (i, j, k), c in counts.items()
        if c > 0
    ]
    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(output_file, index=False)
    return sim_df


def scheduling_IP(
    num_courses,
    random_seed,
    triple_24_start,
    triple_day_start,
    eve_morn_start,
    other_b2b_start,
    alpha,
    beta,
    gamma1,
    gamma2,
    delta,
    vega,
    theta,
    t,
    p,
    blocks,
    lambda_large1=100,
    lambda_large2=50,
    lambda_big=100,
    read=False,
    readfile=None,
    slow=False,
):
    """
    alpha := weight penalty for 3 exams within the same day
    beta := weight penalty for 3 exams within 24 hours
    gamma1 := weight penalty for back-to-back exams (eve→morn)
    gamma2 := weight penalty for back-to-back exams (other)
    delta := weight penalty for 4‑exam sequences
    vega := weight penalty for first exam in later slot
    theta := weight penalty for large gap between exams
    t := triplet coenrollment dictionary
    p := pairwise coenrollment dictionary
    blocks := list of block IDs
    slow := if True, include z‑variables for 4‑block sequences (slower)
    """
    slots = blocks

    # build all index tuples
    block_slot = [(i, s) for i in blocks for s in slots]
    block_pair = [(i, j) for i in blocks for j in blocks]
    block_sequence_trip = [(i, j, k) for i in blocks for j in blocks for k in blocks]
    block_sequence_quad = [
        (i, j, k, l) for i in blocks for j in blocks for k in blocks for l in blocks
    ]
    block_sequence_slot = [
        (i, j, k, s) for i in blocks for j in blocks for k in blocks for s in slots
    ]

    # slot‐to‐slot successor mapping
    shifted_slots = np.roll(slots, -1)
    next_slot = dict(zip(slots, shifted_slots))

    # penalty slot‐sets
    triple_in_day = triple_day_start
    triple_in_24hr = triple_24_start
    triple_slots = np.sort(triple_in_day + triple_in_24hr)
    b2b_eveMorn = eve_morn_start
    b2b_other = other_b2b_start

    # model
    env = gp.Env()
    m = gp.Model("Scheduler", env=env)
    m.setParam("Timelimit", 3600 * 10)

    # decision vars
    x = m.addVars(block_sequence_slot, vtype=GRB.BINARY, name="x")
    y = m.addVars(block_sequence_trip, vtype=GRB.BINARY, name="y")
    if slow:
        z = m.addVars(block_sequence_quad, vtype=GRB.BINARY, name="z")

    # output vars
    schedule = m.addVars(slots, vtype=GRB.INTEGER, name="slot_assignment")
    b = m.addVars(block_slot, vtype=GRB.BINARY, name="b")
    block_assigned = m.addVars(blocks, vtype=GRB.INTEGER, name="block_assigned")
    block_diff = m.addVars(block_pair, vtype=GRB.INTEGER, name="block_diff")
    block_diff_large = m.addVars(block_pair, vtype=GRB.BINARY, name="block_diff_large")

    # penalty counters
    triple_in_day_var = m.addVar(vtype=GRB.INTEGER, name="triple_in_day")
    triple_in_24hr_var = m.addVar(vtype=GRB.INTEGER, name="triple_in_24hr")
    b2b_eveMorn_var = m.addVar(vtype=GRB.INTEGER, name="b2b_eveMorn")
    b2b_other_var = m.addVar(vtype=GRB.INTEGER, name="b2b_other")
    three_exams_four_slots_var = m.addVar(
        vtype=GRB.INTEGER, name="three_exams_four_slots"
    )

    # core sequencing constraints
    m.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for j in blocks for k in blocks for s in slots)
            == 1
            for i in blocks
        ),
        name="each_i",
    )
    m.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for i in blocks for k in blocks for s in slots)
            == 1
            for j in blocks
        ),
        name="each_j",
    )
    m.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for i in blocks for j in blocks for s in slots)
            == 1
            for k in blocks
        ),
        name="each_k",
    )
    m.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for i in blocks for j in blocks for k in blocks)
            == 1
            for s in slots
        ),
        name="each_slot",
    )

    # no repeats
    m.addConstrs(
        (x[i, i, k, s] == 0 for i in blocks for k in blocks for s in slots),
        name="no_ii",
    )
    m.addConstrs(
        (x[i, j, i, s] == 0 for i in blocks for j in blocks for s in slots),
        name="no_ik",
    )
    m.addConstrs(
        (x[i, j, j, s] == 0 for i in blocks for j in blocks for s in slots),
        name="no_jj",
    )

    # continuity
    m.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for i in blocks)
            == gp.quicksum(x[j, k, l, next_slot[s]] for l in blocks)
            for j in blocks
            for k in blocks
            for s in slots
        ),
        name="continuity",
    )

    # define y
    m.addConstrs(
        (y[i, j, k] == gp.quicksum(x[i, j, k, s] for s in list(triple_slots)))
        for i, j, k in block_sequence_trip
    )

    # define z if requested
    if slow:
        m.addConstrs(
            (z[i, j, k, l] >= y[i, j, k] + y[j, k, l] - 1)
            for i, j, k, l in block_sequence_quad
        )

    # write out schedule
    m.addConstrs(
        (
            schedule[s]
            == gp.quicksum(
                i * x[i, j, k, s] for i in blocks for j in blocks for k in blocks
            )
        )
        for s in slots
    )

    # block_slot linkage
    m.addConstrs(
        (b[i, s] == gp.quicksum(x[i, j, k, s] for j in blocks for k in blocks))
        for i in blocks
        for s in slots
    )

    # block_assigned and diff
    m.addConstrs(
        (block_assigned[i] == gp.quicksum(s * b[i, s] for s in slots)) for i in blocks
    )
    m.addConstrs(
        (block_diff[i, j] >= block_assigned[i] - block_assigned[j])
        for i, j in block_pair
    )
    m.addConstrs(
        (block_diff[i, j] >= block_assigned[j] - block_assigned[i])
        for i, j in block_pair
    )
    c = 16
    big_m = 20
    m.addConstrs(
        (block_diff[i, j] >= c * block_diff_large[i, j]) for i, j in block_pair
    )
    m.addConstrs(
        (block_diff[i, j] <= c - 1 + big_m * block_diff_large[i, j])
        for i, j in block_pair
    )

    # penalty definitions
    m.addConstr(
        gp.quicksum(
            t[i, j, k] * x[i, j, k, s]
            for i in blocks
            for j in blocks
            for k in blocks
            for s in list(triple_day_start)
        )
        == triple_in_day_var,
        name="pen1",
    )
    m.addConstr(
        gp.quicksum(
            t[i, j, k] * x[i, j, k, s]
            for i in blocks
            for j in blocks
            for k in blocks
            for s in list(triple_24_start)
        )
        == triple_in_24hr_var,
        name="pen2",
    )
    if slow:
        m.addConstr(
            gp.quicksum(
                (t[i, j, l] + t[i, k, l]) * z[i, j, k, l]
                for i in blocks
                for j in blocks
                for k in blocks
                for l in blocks
            )
            == three_exams_four_slots_var,
            name="pen3",
        )
    else:
        # unconstrained ⇒ will default to 0
        m.addConstr(three_exams_four_slots_var == 0, name="pen3_zero")
    m.addConstr(
        gp.quicksum(
            p[i, j] * x[i, j, k, s]
            for i in blocks
            for j in blocks
            for k in blocks
            for s in list(eve_morn_start)
        )
        == b2b_eveMorn_var,
        name="pen4",
    )
    m.addConstr(
        gp.quicksum(
            p[i, j] * x[i, j, k, s]
            for i in blocks
            for j in blocks
            for k in blocks
            for s in list(other_b2b_start)
        )
        == b2b_other_var,
        name="pen5",
    )

    # objective
    m.setObjective(
        alpha * triple_in_day_var
        + beta * triple_in_24hr_var
        + gamma1 * b2b_eveMorn_var
        + gamma2 * b2b_other_var
        + delta * three_exams_four_slots_var,
        GRB.MINIMIZE,
    )

    # file I/O if requested
    m.update()
    m.write(f"{num_courses}{random_seed}.lp")
    if read:
        m.read(readfile)
    # m.optimize()

    # collect solution
    output = []
    for s in slots:
        output.append((s, int(schedule[s].x)))
    penalty = {
        "triple_in_day": triple_in_day_var.x,
        "triple_in_24hr": triple_in_24hr_var.x,
        "b2b_eveMorn": b2b_eveMorn_var.x,
        "b2b_other": b2b_other_var.x,
        "three_exams_four_slots": three_exams_four_slots_var.x,
    }
    return m.ObjVal, output, penalty


def run_simulation(n_courses, seed, slots, slow):

    sim_df = simulate_triplet_coenrol_simple(
        n_courses=n_courses,
        t_co_file="anon_t_co.csv",
        m_samples=100_000,
        random_seed=seed,
        output_file="sim_t_co.csv",
    )
    print(f"Wrote {len(sim_df)} non-zero triplets")
    print(sim_df.head())
    slots_per_day = 3
    slots = list(range(1, slots + 1))
    blocks = slots
    # Translation to different slot notation
    slots_n = range(1, len(slots) + 1)
    d = dict(zip(slots, slots_n))
    slots_e = list(slots) + [np.inf] * 10

    triple_24_start = []
    triple_day_start = []
    eve_morn_start = []
    other_b2b_start = []

    for j in range(len(slots)):
        s = slots[j]
        if s + 1 == slots_e[j + 1]:  # 11
            if s % slots_per_day == 0:
                eve_morn_start.append(d[s])
            else:
                other_b2b_start.append(d[s])
            if s + 2 == slots_e[j + 2]:  # 111
                if (
                    slots_per_day - s % slots_per_day >= 2
                    and slots_per_day - s % slots_per_day != slots_per_day
                ):
                    triple_day_start.append(d[s])
                else:
                    triple_24_start.append(d[s])

    # — your existing zero‐inits —
    t = {}  # triplet
    p = {}  # pair
    q = {}  # quadruple

    for i in blocks:
        for j in blocks:
            for k in blocks:
                for l in blocks:
                    q[(i, j, k, l)] = 0
                t[(i, j, k)] = 0
            p[(i, j)] = 0

    # — load the exam→block mapping —
    # assumes a CSV with columns "Exam Group","Exam Block"

    # Read the CSV and convert to dictionary
    block_map = pd.read_csv("anon24.csv").set_index("AnonExam")["Exam Block"].to_dict()
    # print("block_map_from_csv", block_map)
    # — load exam‐level co‐enrollment tables —
    # t_co = pd.read_csv('gen_lp/t_co.csv', converters={'triplets': ast.literal_eval})
    p_co = pd.read_csv("anon_coenrol.csv", index_col=0)

    # — update pair counts p[(block_i,block_j)] += exam_pair_count —
    for (exam_i, exam_j), cnt in p_co.stack().items():
        bi = block_map.get(int(exam_i))
        bj = block_map.get(int(exam_j))
        # print("(exam_i, exam_j), cnt ", (exam_i, exam_j), cnt, bi, bj)
        if bi is not None and bj is not None:
            p[(int(bi), int(bj))] += cnt

    # — update triplet counts t[(b1,b2,b3)] += exam_triplet_count —
    for r in sim_df.iterrows():
        # print(r)
        # print(r[1]["course1"])
        # print(r[1]["course2"])

        b1 = block_map.get(r[1]["course1"], random.choice(slots))
        b2 = block_map.get(r[1]["course2"], random.choice(slots))
        b3 = block_map.get(r[1]["course3"], random.choice(slots))
        if None not in (b1, b2, b3):
            t[(int(b1), int(b2), int(b3))] += r[1]["count"]
    alpha1 = 10  # int(lic_param['3_in_1_day'].iloc[0])
    beta1 = 5  # int(lic_param['3_in_24h'].iloc[0])
    gamma3 = 3  # int(lic_param['b_to_b'].iloc[0])
    delta1 = 2  # int(lic_param['2_in_3'].iloc[0])
    vega1 = 1  # int(lic_param['late_first'].iloc[0])

    lambda_big_exam_1 = 1  # int(lic_param['big_exam'].iloc[0]) * size_cutoff_1_weight
    lambda_big_exam_2 = 1  # int(lic_param['big_exam'].iloc[0]) * size_cutoff_2_weight

    lambda_big_block = 1000  # int(lic_param['big_block'].iloc[0])
    theta1 = 2  # int(lic_param['large_gap'].iloc[0])
    first1 = 1  # string_to_int_list(lic_param['first'].iloc[0])
    print("alpha1 =", alpha1)
    print("beta1 =", beta1)
    print("gamma3 =", gamma3)
    print("delta1 =", delta1)
    print("vega1 =", vega1)
    print("lambda_big_exam 1=", lambda_big_exam_1)
    print("lambda_big_exam 2=", lambda_big_exam_2)
    print("lambda_big_block 1 =", lambda_big_block)
    print("theta1 =", theta1)
    print("first1 =", first1)
    scheduling_IP(
        n_courses,
        seed,
        triple_24_start,
        triple_day_start,
        eve_morn_start,
        other_b2b_start,
        alpha=alpha1,
        beta=beta1,
        gamma1=gamma3,
        gamma2=gamma3,
        delta=delta1,
        vega=vega1,
        theta=theta1,
        t=t,
        p=p,
        lambda_large1=lambda_big_exam_1,
        lambda_large2=lambda_big_exam_2,
        lambda_big=lambda_big_block,
        blocks=blocks,
        slow=slow,
    )


def main():

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("size", type=int, help="Number of courses")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--slots", type=int, default=24)
    p.add_argument("--slow", type=bool, default=False)
    args = p.parse_args()
    if args.slots < 24:
        print("Warning: slots must be more than 24")
    args = p.parse_args()
    run_simulation(args.size, seed=args.seed, slots=args.slots, slow=args.slow)


if __name__ == "__main__":
    main()
