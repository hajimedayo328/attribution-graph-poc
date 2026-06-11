# 実験H: 幻覚(誤答)の構造シグネチャ再挑戦
# 同一フォーマット "Fact: The capital of X is" のF(メジャー10カ国)+K(マイナー12カ国)で
# 正答/誤答のグラフ構造を比較する。フォーマット交絡なし。
import csv
import json
import math
import re
from pathlib import Path

import networkx as nx

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
MANIFEST = BASE / "data" / "manifest.csv"


def mann_whitney(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return None, None
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(ranks[k] for k, (_, g) in enumerate(combined) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu) / sigma if sigma else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u1, p


def gini(values):
    v = sorted(values)
    n = len(v)
    s = sum(v)
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(v)) / (n * s) if s else 0.0


def main() -> None:
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        manifest = [r for r in csv.DictReader(f) if r["category"] in ("F", "K", "L")]

    rows = []
    for m in manifest:
        path = BATCH / f"{m['slug']}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        G = nx.DiGraph()
        for n in d["nodes"]:
            G.add_node(n["node_id"], ftype=n["feature_type"],
                       prob=float(n["token_prob"]) if n.get("token_prob") else 0.0,
                       clerp=n.get("clerp", ""))
        for e in d["links"]:
            G.add_edge(e["source"], e["target"], weight=abs(e["weight"]))
        logits = [k for k, a in G.nodes(data=True) if a["ftype"] == "logit"]
        top = max(logits, key=lambda k: G.nodes[k]["prob"])
        mt = re.search(r'Output "(.*?)"', G.nodes[top]["clerp"])
        token = (mt.group(1) if mt else "").strip().lower().strip(".,!?")
        expected = [e for e in m["expected"].split("|") if e]
        correct = any(token.startswith(e) or e.startswith(token) for e in expected) if token else False
        errors = [k for k, a in G.nodes(data=True) if a["ftype"] == "mlp reconstruction error"]
        degs = [dg for _, dg in G.degree()]
        U = G.to_undirected()
        comms = nx.community.louvain_communities(U, weight="weight", seed=42)
        probs = [G.nodes[k]["prob"] for k in logits if G.nodes[k]["prob"] > 0]
        n_tok = len(d["metadata"]["prompt_tokens"])
        rows.append({
            "slug": m["slug"], "cat": m["category"], "token": token, "correct": correct,
            "top1_prob": G.nodes[top]["prob"],
            "entropy": -sum(p * math.log(p) for p in probs),
            "density": nx.density(G), "gini": gini(degs),
            "modularity": nx.community.modularity(U, comms, weight="weight"),
            "error_share": len(errors) / G.number_of_nodes(),
            "nodes_per_token": G.number_of_nodes() / n_tok,
            "edges_per_token": G.number_of_edges() / n_tok,
        })
        print(f"{m['slug']} {m['category']} -> {token!r} correct={correct} p={rows[-1]['top1_prob']:.2f}")

    # 方向比較: K(順方向: 国→首都) vs L(逆方向: 首都→国)
    k_rows = [r for r in rows if r["cat"] == "K"]
    l_rows = [r for r in rows if r["cat"] == "L"]
    if l_rows:
        print(f"\n===== 逆転の呪い: 順方向K({len(k_rows)}) vs 逆方向L({len(l_rows)}) =====")
        print(f"K正答率: {sum(r['correct'] for r in k_rows)}/{len(k_rows)}")
        print(f"L正答率: {sum(r['correct'] for r in l_rows)}/{len(l_rows)}")
        keys_d = ["density", "gini", "modularity", "error_share", "nodes_per_token",
                  "edges_per_token", "top1_prob", "entropy"]
        print(f"{'指標':16s} {'K mean':>10s} {'L mean':>10s} {'p':>8s}")
        for k in keys_d:
            a = [r[k] for r in k_rows]
            b = [r[k] for r in l_rows]
            _, p = mann_whitney(a, b)
            print(f"{k:16s} {sum(a)/len(a):10.3f} {sum(b)/len(b):10.3f} {p:8.4f}")

    cor = [r for r in rows if r["correct"]]
    inc = [r for r in rows if not r["correct"]]
    print(f"\n正答={len(cor)} 誤答={len(inc)}")
    if len(inc) < 3:
        print("誤答が3未満のため検定不能。プロンプトをさらに難化させる必要あり")
        return

    keys = ["density", "gini", "modularity", "error_share", "nodes_per_token",
            "edges_per_token", "top1_prob", "entropy"]
    tests = []
    print(f"\n{'指標':16s} {'正答mean':>10s} {'誤答mean':>10s} {'p':>8s}")
    for k in keys:
        a = [r[k] for r in cor]
        b = [r[k] for r in inc]
        _, p = mann_whitney(a, b)
        tests.append((k, p))
        print(f"{k:16s} {sum(a)/len(a):10.3f} {sum(b)/len(b):10.3f} {p:8.4f}")
    # BH補正
    m_ = len(tests)
    order = sorted(range(m_), key=lambda i: tests[i][1])
    prev = 1.0
    adj = [0.0] * m_
    for ri in range(m_ - 1, -1, -1):
        i = order[ri]
        prev = min(prev, tests[i][1] * m_ / (ri + 1))
        adj[i] = prev
    print("\nBH補正後 q<0.05:")
    for (k, p), q in zip(tests, adj):
        if q < 0.05:
            print(f"  {k} q={q:.4f}")
    # 構造のみでのAUC(出力系除外)
    print("\n単一構造特徴のAUC(誤答検出):")
    for k in ["density", "gini", "modularity", "error_share", "nodes_per_token", "edges_per_token"]:
        pos = [r[k] for r in inc]
        neg = [r[k] for r in cor]
        u1, _ = mann_whitney(pos, neg)
        auc = u1 / (len(pos) * len(neg))
        print(f"  {k:16s} AUC={max(auc, 1-auc):.3f}")


if __name__ == "__main__":
    main()
