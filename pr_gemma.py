# 実験K(再現): participation ratio — qwen06_compare.py と完全同一の手法をgemmaに適用
# inflow = transcoderソース・abs(w)>0。generic=pr_top(答えない群)、content=pr_top(答えた群)+pr_cap(答えない群)
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"

SLUGS = {
    ("France", "F"): "agpoc-f40-hj328", ("Japan", "F"): "agpoc-f41-hj328",
    ("Germany", "F"): "agpoc-f43-hj328", ("Egypt", "F"): "agpoc-f47-hj328",
    ("France", "G"): "agpoc-g60-hj328", ("Japan", "G"): "agpoc-g61-hj328",
    ("Germany", "G"): "agpoc-g62-hj328", ("Egypt", "G"): "agpoc-g63-hj328",
    ("France", "H"): "agpoc-h64-hj328", ("Japan", "H"): "agpoc-h65-hj328",
    ("Germany", "H"): "agpoc-h66-hj328", ("Egypt", "H"): "agpoc-h67-hj328",
}
CAP = {"France": "paris", "Japan": "tokyo", "Germany": "berlin", "Egypt": "cairo"}


def pr_ratio(ws):
    s1, s2 = sum(ws), sum(w * w for w in ws)
    return (s1 * s1) / s2 if s2 else 0.0


def load(slug, country):
    d = json.loads((BATCH / f"{slug}.json").read_text(encoding="utf-8"))
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
    cap = CAP[country]
    cap_nodes = [k for k, (c, _) in logits.items() if cap in c.lower()]
    return {
        "answered": bool(cap_nodes) and top == cap_nodes[0],
        "pr_top": pr_ratio(inflow.get(top, [])),
        "pr_cap": pr_ratio(inflow.get(cap_nodes[0], [])) if cap_nodes else None,
    }


def main() -> None:
    generic, content = [], []
    for (country, phr), slug in sorted(SLUGS.items(), key=lambda x: (x[0][1], x[0][0])):
        r = load(slug, country)
        if r["answered"]:
            content.append(r["pr_top"])
        else:
            generic.append(r["pr_top"])
            if r["pr_cap"] is not None:
                content.append(r["pr_cap"])
        print(f"{country:8s} {phr}  answered={r['answered']!s:5s} pr_top={r['pr_top']:6.1f}"
              f" pr_cap={('%.1f' % r['pr_cap']) if r['pr_cap'] is not None else '-':>6s}")
    print(f"\n汎用語トークンPR: mean={sum(generic)/len(generic):.1f} (n={len(generic)}) "
          f"range=[{min(generic):.0f},{max(generic):.0f}]   (ページ主張: mean224, range197-281)")
    print(f"内容(首都)トークンPR: mean={sum(content)/len(content):.1f} (n={len(content)}) "
          f"range=[{min(content):.0f},{max(content):.0f}]   (ページ主張: mean75, range67-85)")


if __name__ == "__main__":
    main()
