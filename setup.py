# anonymize.py
#!/usr/bin/env python3
import os
import pandas as pd
import ast

ORIG_FILE = "p_co.csv"
ANON_FILE = "anon_coenrol.csv"
HIST_TOTALS_FILE = "hist_totals.csv"


def load_original_coenrol(fp=ORIG_FILE):
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    return pd.read_csv(fp, index_col=0)


def anonymize_by_importance(df):
    # 1) compute total co-enrol per course
    totals = df.sum(axis=1).sort_values(ascending=False)
    # 2) map original → ranked IDs starting at 1
    mapping = {orig: rank + 1 for rank, orig in enumerate(totals.index)}
    # 3) remap both axes
    df_anon = df.rename(index=mapping, columns=mapping)
    df_anon = df_anon.sort_index(axis=0).sort_index(axis=1)
    df_anon.to_csv(ANON_FILE)
    return df_anon


def compute_hist_totals(df_anon):
    # stack into (course1,course2)->count
    stacked = df_anon.stack().rename("count")
    # drop self-pairs
    stacked = stacked[
        stacked.index.get_level_values(0) != stacked.index.get_level_values(1)
    ]
    stacked.index.names = ["course1", "course2"]
    stacked.to_csv(HIST_TOTALS_FILE)
    return stacked


def setup():

    def map_triplet(s):
        tup = ast.literal_eval(s)  # turns the string into ('mlg22','mlg33','1-8990')
        return tuple(e for e in tup)  # maps each entry

    df = pd.read_csv("../sp24/t_co.csv")

    df["triplets_mapped"] = df["triplets"].apply(map_triplet)

    ord_set = set()
    add_order = []
    n = 0
    for i in df["triplets_mapped"]:
        # print(i)
        # print(type(i))
        old = ord_set.copy()
        ord_set.add(i[0])
        ord_set.add(i[1])
        ord_set.add(i[2])
        if len(old) < len(ord_set):
            a = ord_set - old.intersection(ord_set)
            for i in range(len(a)):
                add_order.append(a.pop())
    exam_map = {}
    for i, e in enumerate(add_order):
        exam_map[e] = i
    pd.DataFrame(data=exam_map.values(), index=exam_map.keys(), columns=["num"]).to_csv(
        "exam_map.csv"
    )

    def map_triplet(s):
        tup = ast.literal_eval(s)  # turns the string into ('mlg22','mlg33','1-8990')
        return tuple(exam_map[e] for e in tup)  # maps each entry

    df["triplets_mapped"] = df["triplets"].apply(map_triplet)
    df = df.set_index(
        pd.MultiIndex.from_tuples(df["triplets_mapped"], names=["a", "b", "c"])
    )
    # print(df)
    df[["triplets_mapped", "co"]].to_csv("anon_t_co.csv")
    ba = pd.read_csv("block24.csv")
    ba["AnonExam"] = ba["Exam Group"].map(exam_map)
    block_map = dict(zip(ba["AnonExam"], ba["Exam Block"]))
    # Save block_map to annon24.csv
    pd.DataFrame(list(block_map.items()), columns=["AnonExam", "Exam Block"]).to_csv(
        "anon24.csv", index=False
    )


def main():
    orig = load_original_coenrol()
    df_anon = anonymize_by_importance(orig)
    compute_hist_totals(df_anon)
    print(f"Anonymization complete. Files saved: {ANON_FILE}, {HIST_TOTALS_FILE}")
    setup()


if __name__ == "__main__":

    main()
