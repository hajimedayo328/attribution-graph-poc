# 追加実験A〜D
# A: レイヤー分布(計算の重心・深さ) B: 埋め込み→logit最強経路のホップ数
# C: グラフ間Jaccard類似度(回路の普遍性) D: 特徴追加版LOO分類
import csv
import json
import math
import random
from pathlib import Path

import networkx as nx

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
MANIFEST = BASE / "data" / "manifest.csv"
OUT = BASE / "data" / "results2.csv"


def mann_whitney(a, b):
    n1, n2 = len(a), len(b)
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


def bh(tests):
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i][1])
    adj = [0.0] * m
    prev = 1.0
    for ri in range(m - 1, -1, -1):
        i = order[ri]
        prev = min(prev, tests[i][1] * m / (ri + 1))
        adj[i] = prev
    return adj


def load_graph(slug):
    d = json.loads((BATCH / f"{slug}.json").read_text(encoding="utf-8"))
    G = nx.DiGraph()
    for n in d["nodes"]:
        G.add_node(n["node_id"], ftype=n["feature_type"], layer=n["layer"],
                   feature=n["feature"],
                   prob=float(n["token_prob"]) if n.get("token_prob") else 0.0)
    edges = [(e["source"], e["target"], abs(e["weight"])) for e in d["links"] if abs(e["weight"]) > 0]
    wmax = max(w for _, _, w in edges)
    for s, t, w in edges:
        # 重み積最大経路 = -log(w/wmax)最小経路(非負距離)
        G.add_edge(s, t, weight=w, dist=-math.log(w / wmax) + 1e-9)
    return G, d


def graph_features(G):
    feats = [(k, a) for k, a in G.nodes(data=True) if a["ftype"] == "cross layer transcoder"]
    logits = [k for k, a in G.nodes(data=True) if a["ftype"] == "logit"]
    embs = [k for k, a in G.nodes(data=True) if a["ftype"] == "embedding"]
    top_logit = max(logits, key=lambda k: G.nodes[k]["prob"])

    # A: PageRank質量のレイヤー分布
    RG = G.reverse(copy=False)
    pers = {n: (1.0 if n in logits else 0.0) for n in G.nodes}
    pr = nx.pagerank(RG, weight="weight", personalization=pers, alpha=0.9)
    mass = [(int(a["layer"]), pr[k]) for k, a in feats]
    total = sum(m for _, m in mass) or 1e-12
    centroid = sum(l * m for l, m in mass) / total
    late = sum(m for l, m in mass if l >= 18) / total      # 層18-25
    early = sum(m for l, m in mass if l <= 8) / total      # 層0-8

    # B: 最強経路(重み積最大=対数距離最小)のホップ数と無重み最短路
    dist_min, hops_strong = None, None
    # 逆グラフでlogitから埋め込みへ(向きを遡る)
    RGd = G.reverse(copy=False)
    sp_len = nx.single_source_dijkstra_path_length(RGd, top_logit, weight="dist")
    sp_path = nx.single_source_dijkstra_path(RGd, top_logit, weight="dist")
    best_emb = min((e for e in embs if e in sp_len), key=lambda e: sp_len[e], default=None)
    if best_emb is not None:
        hops_strong = len(sp_path[best_emb]) - 1
        dist_min = sp_len[best_emb]
    try:
        min_hops = min(nx.shortest_path_length(RGd, top_logit, e) for e in embs
                       if nx.has_path(RGd, top_logit, e))
    except ValueError:
        min_hops = None

    # C用: 特徴量集合(位置非依存)
    fset = {(a["layer"], a["feature"]) for _, a in feats}

    return {
        "layer_centroid": centroid,
        "late_share": late,
        "early_share": early,
        "hops_strong": hops_strong,
        "dist_strong": dist_min,
        "min_hops": min_hops,
    }, fset


def main() -> None:
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f))
    with open(BASE / "data" / "results.csv", encoding="utf-8", newline="") as f:
        old = {r["slug"]: r for r in csv.DictReader(f)}

    rows, fsets = [], {}
    for m in manifest:
        slug = m["slug"]
        G, d = load_graph(slug)
        ft, fset = graph_features(G)
        ft["slug"] = slug
        ft["category"] = m["category"]
        fsets[slug] = fset
        # 既存特徴をマージ
        for k in ["density", "degree_gini", "modularity", "error_share", "mean_degree",
                  "n_nodes", "n_edges", "n_communities"]:
            ft[k] = float(old[slug][k])
        dd = json.loads((BATCH / f"{slug}.json").read_text(encoding="utf-8"))
        ft["n_tokens"] = len(dd["metadata"].get("prompt_tokens", dd["metadata"].get("promptTokens", [])))
        ft["nodes_per_token"] = ft["n_nodes"] / ft["n_tokens"]
        ft["edges_per_token"] = ft["n_edges"] / ft["n_tokens"]
        rows.append(ft)
        print(f"{slug} cat={m['category']} centroid={ft['layer_centroid']:.1f} "
              f"late={ft['late_share']:.2f} hops={ft['hops_strong']} min_hops={ft['min_hops']}", flush=True)

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def grp(c):
        return [r for r in rows if r["category"] == c]

    def ms(vs):
        vs = [v for v in vs if v is not None]
        m = sum(vs) / len(vs)
        s = (sum((v - m) ** 2 for v in vs) / max(1, len(vs) - 1)) ** 0.5
        return m, s

    print("\n===== 実験A/B: カテゴリ別のレイヤー重心と経路 =====")
    print("cat  centroid  late_share  early_share  hops_strong  min_hops")
    for c in "ABCDEF":
        g = grp(c)
        print(f"{c}   {ms([r['layer_centroid'] for r in g])[0]:8.2f}"
              f"  {ms([r['late_share'] for r in g])[0]:9.3f}"
              f"  {ms([r['early_share'] for r in g])[0]:10.3f}"
              f"  {ms([r['hops_strong'] for r in g])[0]:10.2f}"
              f"  {ms([r['min_hops'] for r in g])[0]:8.2f}")

    print("\n検定(BH補正):")
    tests, labels = [], []
    for k in ["layer_centroid", "late_share", "early_share", "hops_strong", "dist_strong"]:
        for name, g1, g2 in [("A vs B", grp("A"), grp("B")), ("C vs F", grp("C"), grp("F")),
                             ("A vs F", grp("A"), grp("F"))]:
            v1 = [r[k] for r in g1 if r[k] is not None]
            v2 = [r[k] for r in g2 if r[k] is not None]
            _, p = mann_whitney(v1, v2)
            tests.append((f"{name}: {k}", p))
    for (name, p), q in sorted(zip(tests, bh(tests)), key=lambda t: t[0][1]):
        print(f"  {name:28s} p={p:.4f} q={q:.4f}{' *' if q < 0.05 else ''}")

    print("\n===== 実験C: グラフ間Jaccard類似度(回路の共有度) =====")
    slugs = [r["slug"] for r in rows]
    cat = {r["slug"]: r["category"] for r in rows}
    within, between = [], []
    sim = {}
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            a, b = fsets[slugs[i]], fsets[slugs[j]]
            jac = len(a & b) / len(a | b)
            sim[(i, j)] = jac
            (within if cat[slugs[i]] == cat[slugs[j]] else between).append(jac)
    mw, mb = ms(within)[0], ms(between)[0]
    print(f"同カテゴリ平均Jaccard={mw:.3f}  異カテゴリ平均={mb:.3f}  比={mw/mb:.2f}")
    # 置換検定: カテゴリラベルをシャッフルして within-between 差の帰無分布
    obs = mw - mb
    random.seed(42)
    cnt = 0
    labels_list = [cat[s] for s in slugs]
    for _ in range(2000):
        perm = labels_list[:]
        random.shuffle(perm)
        w_, b_ = [], []
        for (i, j), v in sim.items():
            (w_ if perm[i] == perm[j] else b_).append(v)
        if (sum(w_) / len(w_) - sum(b_) / len(b_)) >= obs:
            cnt += 1
    print(f"置換検定 p={cnt/2000:.4f} (within>between)")
    # 1-NN分類
    correct = 0
    for i in range(len(slugs)):
        best_j, best = None, -1
        for j in range(len(slugs)):
            if i == j:
                continue
            v = sim[(min(i, j), max(i, j))]
            if v > best:
                best, best_j = v, j
        if cat[slugs[best_j]] == cat[slugs[i]]:
            correct += 1
    print(f"1-NN(Jaccard)による6カテゴリ分類: {correct}/{len(slugs)} (チャンス約10/60)")

    print("\n===== 実験D: 特徴追加版 LOO分類 (B vs その他) =====")
    feats_d = ["nodes_per_token", "edges_per_token", "density", "degree_gini",
               "modularity", "error_share", "mean_degree",
               "layer_centroid", "late_share", "early_share", "hops_strong"]
    data = [r for r in rows if all(r[k] is not None for k in feats_d)]
    y_all = [1 if r["category"] == "B" else 0 for r in data]
    X_all = [[float(r[k]) for k in feats_d] for r in data]

    def logistic_train(X, y, epochs=3000, lr=0.5, l2=0.01):
        n, dd_ = len(X), len(X[0])
        w = [0.0] * dd_
        b = 0.0
        for _ in range(epochs):
            gw = [0.0] * dd_
            gb = 0.0
            for xi, yi in zip(X, y):
                z = sum(wj * xj for wj, xj in zip(w, xi)) + b
                p = 1 / (1 + math.exp(-max(-30, min(30, z))))
                err = p - yi
                for jj in range(dd_):
                    gw[jj] += err * xi[jj]
                gb += err
            w = [wj - lr * (gwj / n + l2 * wj) for wj, gwj in zip(w, gw)]
            b -= lr * gb / n
        return w, b

    loo = []
    for hold in range(len(data)):
        X_tr = [x for i, x in enumerate(X_all) if i != hold]
        y_tr = [v for i, v in enumerate(y_all) if i != hold]
        dd_ = len(feats_d)
        mus = [sum(x[j] for x in X_tr) / len(X_tr) for j in range(dd_)]
        sds = [max(1e-9, (sum((x[j] - mus[j]) ** 2 for x in X_tr) / len(X_tr)) ** 0.5) for j in range(dd_)]
        Xs = [[(x[j] - mus[j]) / sds[j] for j in range(dd_)] for x in X_tr]
        w, b = logistic_train(Xs, y_tr)
        xh = [(X_all[hold][j] - mus[j]) / sds[j] for j in range(dd_)]
        z = sum(wj * xj for wj, xj in zip(w, xh)) + b
        loo.append(1 / (1 + math.exp(-max(-30, min(30, z)))))
    pos = [s for s, yv in zip(loo, y_all) if yv == 1]
    neg = [s for s, yv in zip(loo, y_all) if yv == 0]
    u1, _ = mann_whitney(pos, neg)
    print(f"AUC = {u1/(len(pos)*len(neg)):.3f}  (旧7特徴: 0.756)")


if __name__ == "__main__":
    main()
