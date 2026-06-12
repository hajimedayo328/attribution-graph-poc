# 実験L: universality検証 — gemma-2-2bで見つけた3現象がQwen3-1.7Bでも再現するか
# 1) 回路同一性の言い回し支配(全位置) vs 国位置での内容優位への逆転
# 2) participation ratio: 汎用語=広く浅い vs 内容語=狭く深い
# 3) 読み出しレジーム(どの言い回しで首都を答えるか)
import json
import re
from itertools import combinations
from pathlib import Path

OUT = Path(__file__).parent / "graphs_qwen06"
COUNTRIES = {"France": "paris", "Japan": "tokyo", "Germany": "berlin", "Egypt": "cairo"}
PHRASINGS = ["F", "G", "H"]


def norm_tok(t):
    return re.sub(r"[^a-z]", "", t.lower())


def load(slug, country):
    d = json.loads((OUT / f"{slug}.json").read_text(encoding="utf-8"))
    toks = d["metadata"]["prompt_tokens"]
    cpos = {i for i, t in enumerate(toks) if country.lower() in norm_tok(t)}
    last = len(toks) - 1
    fs_all, fs_cpos = set(), set()
    for n in d["nodes"]:
        if n["feature_type"] != "cross layer transcoder":
            continue
        layer, feat, ctx = n["node_id"].split("_")
        fs_all.add((layer, feat))
        if int(ctx) in cpos:
            fs_cpos.add((layer, feat))
    # logit情報
    logits = {n["node_id"]: (n.get("clerp", ""), float(n["token_prob"] or 0))
              for n in d["nodes"] if n["feature_type"] == "logit"}
    kind = {n["node_id"]: n["feature_type"] for n in d["nodes"]}
    inflow = {}
    for e in d["links"]:
        if e["target"] in logits and kind.get(e["source"]) == "cross layer transcoder":
            w = abs(e["weight"])
            if w > 0:
                inflow.setdefault(e["target"], []).append(w)
    top = max(logits, key=lambda k: logits[k][1])
    cap = COUNTRIES[country]
    cap_nodes = [k for k, (c, _) in logits.items() if cap in c.lower()]
    m = re.search(r'Output "(.*?)"', logits[top][0])
    top_tok = (m.group(1) if m else "").strip().lower()

    def pr_ratio(ws):
        s1, s2 = sum(ws), sum(w * w for w in ws)
        return (s1 * s1) / s2 if s2 else 0.0

    return {
        "all": fs_all, "cpos": fs_cpos,
        "top_tok": top_tok, "answered": bool(cap_nodes) and top == cap_nodes[0],
        "cap_in_top10": bool(cap_nodes),
        "pr_top": pr_ratio(inflow.get(top, [])),
        "pr_cap": pr_ratio(inflow.get(cap_nodes[0], [])) if cap_nodes else None,
    }


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    data = {}
    for p in PHRASINGS:
        for c in COUNTRIES:
            slug = f"qwen06-{p.lower()}-{c.lower()}"
            if (OUT / f"{slug}.json").exists():
                data[(c, p)] = load(slug, c)
    print(f"loaded {len(data)}/12 graphs\n")

    print("===== 3) 読み出しレジーム =====")
    for (c, p), r in sorted(data.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"{c:8s} {p}  top={r['top_tok']!r:14s} answered={r['answered']} cap_in_top10={r['cap_in_top10']}")

    print("\n===== 1) Jaccard: 言い回し vs 内容 =====")
    for scope, label in [("all", "全位置"), ("cpos", "国トークン位置のみ")]:
        same_c, same_p, diff = [], [], []
        for k1, k2 in combinations(data, 2):
            j = jac(data[k1][scope], data[k2][scope])
            if k1[0] == k2[0]:
                same_c.append(j)
            elif k1[1] == k2[1]:
                same_p.append(j)
            else:
                diff.append(j)
        cn = pn = 0
        for k1 in data:
            best, bj = None, -1
            for k2 in data:
                if k1 == k2:
                    continue
                j = jac(data[k1][scope], data[k2][scope])
                if j > bj:
                    bj, best = j, k2
            cn += best[0] == k1[0]
            pn += best[1] == k1[1]
        print(f"[{label}] 同じ国={sum(same_c)/len(same_c):.3f} 同じ言い回し={sum(same_p)/len(same_p):.3f}"
              f" 両方違う={sum(diff)/len(diff):.3f} | 1-NN: 国{cn}/{len(data)} 言い回し{pn}/{len(data)}")

    print("\n===== 2) Participation Ratio =====")
    generic, content = [], []
    for r in data.values():
        if r["answered"]:
            content.append(r["pr_top"])
        else:
            generic.append(r["pr_top"])
            if r["pr_cap"] is not None:
                content.append(r["pr_cap"])
    if generic:
        print(f"汎用語トークンPR: mean={sum(generic)/len(generic):.1f} (n={len(generic)}) range=[{min(generic):.0f},{max(generic):.0f}]")
    if content:
        print(f"首都トークンPR:   mean={sum(content)/len(content):.1f} (n={len(content)}) range=[{min(content):.0f},{max(content):.0f}]")


if __name__ == "__main__":
    main()
