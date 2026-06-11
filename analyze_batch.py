# バッチ生成したattribution graph群の統計分析
# 1) influence再現対決: 個人化PageRank vs 重み付き次数ベースライン
# 2) グラフ不変量のカテゴリ間(A〜E)・正誤間の比較(Mann-Whitney U + BH補正)
import csv
import json
import math
import re
from pathlib import Path

import networkx as nx

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
MANIFEST = BASE / "data" / "manifest.csv"
RESULTS = BASE / "data" / "results.csv"


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def mann_whitney(a, b):
    """両側Mann-Whitney U (正規近似)。(U, p) を返す"""
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
    if sigma == 0:
        return u1, 1.0
    z = (u1 - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u1, p


def gini(values):
    v = sorted(values)
    n = len(v)
    s = sum(v)
    if s == 0:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(v)) / (n * s)


def analyze_graph(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    G = nx.DiGraph()
    for n in d["nodes"]:
        G.add_node(
            n["node_id"],
            ftype=n["feature_type"],
            cum_infl=float(n["influence"]) if n["influence"] is not None else 1.0,
            prob=float(n["token_prob"]) if n.get("token_prob") else 0.0,
            clerp=n.get("clerp", ""),
        )
    for e in d["links"]:
        G.add_edge(e["source"], e["target"], weight=abs(e["weight"]))

    feats = [k for k, a in G.nodes(data=True) if a["ftype"] == "cross layer transcoder"]
    errors = [k for k, a in G.nodes(data=True) if a["ftype"] == "mlp reconstruction error"]
    logits = [k for k, a in G.nodes(data=True) if a["ftype"] == "logit"]

    # モデルの出力: top-1 logitトークン
    top_logit = max(logits, key=lambda k: G.nodes[k]["prob"])
    m = re.search(r'Output "(.*?)"', G.nodes[top_logit]["clerp"])
    top_token = (m.group(1) if m else "").strip().lower().strip(".,!?")
    top1_prob = G.nodes[top_logit]["prob"]
    probs = [G.nodes[k]["prob"] for k in logits if G.nodes[k]["prob"] > 0]
    entropy = -sum(p * math.log(p) for p in probs)

    # 不変量
    U = G.to_undirected()
    comms = nx.community.louvain_communities(U, weight="weight", seed=42)
    modularity = nx.community.modularity(U, comms, weight="weight")
    degs = [dg for _, dg in G.degree()]

    # influence再現対決(featureノードのみ、importance = -累積influence)
    imp = {k: -G.nodes[k]["cum_infl"] for k in feats}
    RG = G.reverse(copy=False)
    pers = {n: (1.0 if n in logits else 0.0) for n in G.nodes}
    pr = nx.pagerank(RG, weight="weight", personalization=pers, alpha=0.9)
    w_in = dict(G.in_degree(weight="weight"))
    w_out = dict(G.out_degree(weight="weight"))
    w_tot = {k: w_in[k] + w_out[k] for k in G.nodes}

    ys = [imp[k] for k in feats]
    rho_pr = spearman([pr[k] for k in feats], ys)
    rho_in = spearman([w_in[k] for k in feats], ys)
    rho_out = spearman([w_out[k] for k in feats], ys)
    rho_tot = spearman([w_tot[k] for k in feats], ys)

    def top10_overlap(score):
        top_s = set(sorted(feats, key=lambda k: -score[k])[:10])
        top_i = set(sorted(feats, key=lambda k: -imp[k])[:10])
        return len(top_s & top_i)

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "n_features": len(feats),
        "error_share": len(errors) / G.number_of_nodes(),
        "modularity": modularity,
        "n_communities": len(comms),
        "mean_degree": sum(degs) / len(degs),
        "degree_gini": gini(degs),
        "top_token": top_token,
        "top1_prob": top1_prob,
        "logit_entropy": entropy,
        "rho_pagerank": rho_pr,
        "rho_w_indeg": rho_in,
        "rho_w_outdeg": rho_out,
        "rho_w_totdeg": rho_tot,
        "ov10_pagerank": top10_overlap(pr),
        "ov10_w_outdeg": top10_overlap(w_out),
        "ov10_w_totdeg": top10_overlap(w_tot),
    }


def main() -> None:
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f))

    rows = []
    for mrow in manifest:
        path = BATCH / f"{mrow['slug']}.json"
        if not path.exists():
            continue
        r = analyze_graph(path)
        expected = [e for e in mrow["expected"].split("|") if e]
        r["slug"] = mrow["slug"]
        r["category"] = mrow["category"]
        r["prompt"] = mrow["prompt"]
        # Aは「Fact:なし」フレーミングで説明文に流れるためラベル対象外(Fが正誤測定枠)
        r["correct"] = (r["top_token"] in expected) if (expected and mrow["category"] != "A") else None
        rows.append(r)
        print(f"{mrow['slug']} cat={mrow['category']} top={r['top_token']!r} "
              f"correct={r['correct']} rho_pr={r['rho_pagerank']:.3f} mod={r['modularity']:.3f}",
              flush=True)

    fields = list(rows[0].keys())
    with open(RESULTS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ===== 集計 =====
    def mean_std(vs):
        m = sum(vs) / len(vs)
        s = (sum((v - m) ** 2 for v in vs) / (len(vs) - 1)) ** 0.5 if len(vs) > 1 else 0.0
        return m, s

    print("\n===== 1) influence再現対決 (n=%d) =====" % len(rows))
    for key in ["rho_pagerank", "rho_w_indeg", "rho_w_outdeg", "rho_w_totdeg"]:
        m, s = mean_std([r[key] for r in rows])
        print(f"{key:16s} mean={m:.3f} sd={s:.3f}")
    for key in ["ov10_pagerank", "ov10_w_outdeg", "ov10_w_totdeg"]:
        m, s = mean_std([r[key] for r in rows])
        print(f"{key:16s} mean={m:.2f}/10 sd={s:.2f}")

    print("\n===== 2) カテゴリ別不変量 =====")
    inv_keys = ["modularity", "density", "n_nodes", "error_share", "top1_prob",
                "logit_entropy", "degree_gini", "rho_pagerank"]
    cats = sorted({r["category"] for r in rows})
    header = "cat n  " + "  ".join(f"{k[:10]:>10s}" for k in inv_keys)
    print(header)
    for c in cats:
        sub = [r for r in rows if r["category"] == c]
        vals = "  ".join(f"{mean_std([r[k] for r in sub])[0]:10.3f}" for k in inv_keys)
        print(f"{c}  {len(sub):2d} {vals}")

    print("\n===== 3) 検定 (Mann-Whitney U, BH補正) =====")
    tests = []
    a_rows = [r for r in rows if r["category"] == "A"]
    b_rows = [r for r in rows if r["category"] == "B"]
    f_rows = [r for r in rows if r["category"] == "F"]
    cor = [r for r in rows if r["correct"] is True]
    inc = [r for r in rows if r["correct"] is False]
    print(f"A={len(a_rows)} B={len(b_rows)} F={len(f_rows)} correct={len(cor)} incorrect={len(inc)}")
    for k in inv_keys:
        for name, g1, g2 in [("A vs B (fact vs fiction)", a_rows, b_rows),
                             ("F vs B (fact vs fiction)", f_rows, b_rows),
                             ("A vs F (framing)", a_rows, f_rows),
                             ("correct vs incorrect", cor, inc)]:
            u, p = mann_whitney([r[k] for r in g1], [r[k] for r in g2])
            if p is not None:
                tests.append((f"{name}: {k}", p))
    # Benjamini-Hochberg
    m_tests = len(tests)
    order = sorted(range(m_tests), key=lambda i: tests[i][1])
    adj = [0.0] * m_tests
    prev = 1.0
    for rank_i in range(m_tests - 1, -1, -1):
        i = order[rank_i]
        val = min(prev, tests[i][1] * m_tests / (rank_i + 1))
        adj[i] = val
        prev = val
    for (name, p), q in sorted(zip(tests, adj), key=lambda t: t[0][1]):
        flag = " *" if q < 0.05 else ""
        print(f"{name:42s} p={p:.4f} q(BH)={q:.4f}{flag}")


if __name__ == "__main__":
    main()
