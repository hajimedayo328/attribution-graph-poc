# attribution graph のグラフ統計 PoC
# 入力: Neuronpedia形式のattribution graph JSON
import json
import sys
from collections import Counter

import networkx as nx

path = sys.argv[1] if len(sys.argv) > 1 else "data/dallas_austin.json"
d = json.load(open(path, encoding="utf-8"))

G = nx.DiGraph()
for n in d["nodes"]:
    G.add_node(
        n["node_id"],
        layer=n["layer"],
        ctx=n["ctx_idx"],
        ftype=n["feature_type"],
        influence=float(n["influence"]) if n["influence"] is not None else 0.0,
    )
for e in d["links"]:
    G.add_edge(e["source"], e["target"], weight=abs(e["weight"]), raw=e["weight"])

print(f"=== {d['metadata'].get('slug', path)} ===")
print(f"prompt: {d['metadata'].get('prompt', '')!r}")
print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} density={nx.density(G):.4f}")
print("feature_type:", dict(Counter(nx.get_node_attributes(G, 'ftype').values())))
print(f"DAG?: {nx.is_directed_acyclic_graph(G)}")
print(f"weakly connected components: {nx.number_weakly_connected_components(G)}")

deg = [dg for _, dg in G.degree()]
deg_sorted = sorted(deg, reverse=True)
print(f"degree: max={deg_sorted[0]} top5={deg_sorted[:5]} mean={sum(deg)/len(deg):.1f}")

# コミュニティ検出(無向化・重み付き)
U = G.to_undirected()
comms = nx.community.louvain_communities(U, weight="weight", seed=42)
mod = nx.community.modularity(U, comms, weight="weight")
sizes = sorted((len(c) for c in comms), reverse=True)
print(f"louvain: {len(comms)} communities, modularity={mod:.3f}, sizes top10={sizes[:10]}")

# logit起点の個人化PageRank(出力から逆向きに影響を遡る) vs circuit-tracer公式influence
# 注意: JSONのinfluenceは枝刈り用の累積スコアで「小さいほど重要」(circuit-tracer/create_graph_files.py)
RG = G.reverse(copy=False)
logits = [n for n, a in G.nodes(data=True) if a["ftype"] == "logit"]
pers = {n: (1.0 if n in logits else 0.0) for n in G.nodes}
pr = nx.pagerank(RG, weight="weight", personalization=pers, alpha=0.9)
infl = {k: -v for k, v in nx.get_node_attributes(G, "influence").items()}  # 符号反転で「大=重要」
common = [k for k in G.nodes if G.nodes[k]["ftype"] == "cross layer transcoder"]

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
    return cov / (vx * vy)

rho = spearman([pr[k] for k in common], [infl[k] for k in common])
print(f"Spearman(reverse-PageRank, official influence): rho={rho:.3f} (n={len(common)})")

# 上位ノードの顔ぶれ比較
top_pr = sorted(common, key=lambda k: -pr[k])[:10]
top_in = sorted(common, key=lambda k: -infl[k])[:10]
print("top10 PageRank:", top_pr)
print("top10 influence:", top_in)
print(f"top10 overlap: {len(set(top_pr) & set(top_in))}/10")
