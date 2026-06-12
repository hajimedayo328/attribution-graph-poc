# 実験I-2: 情報輸送のQwen3-0.6B再現
# gemmaの発見: 答えた群はlogit直結シェアが高く、H失敗は「輸送は届くが読み出されない」
import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "graphs_qwen06"
CAPS = {"France": "paris", "Japan": "tokyo", "Germany": "berlin", "Egypt": "cairo"}
PHRASINGS = ["F", "G", "H"]


def norm_tok(t):
    return re.sub(r"[^a-z]", "", t.lower())


def analyze(slug, country):
    d = json.loads((OUT / f"{slug}.json").read_text(encoding="utf-8"))
    toks = d["metadata"]["prompt_tokens"]
    cpos = {i for i, t in enumerate(toks) if country.lower() in norm_tok(t)}
    last = len(toks) - 1

    pos_of, kind_of = {}, {}
    for n in d["nodes"]:
        kind_of[n["node_id"]] = n["feature_type"]
        if n["feature_type"] == "cross layer transcoder":
            pos_of[n["node_id"]] = int(n["node_id"].split("_")[2])
        elif n["feature_type"] == "embedding":
            pos_of[n["node_id"]] = int(n["ctx_idx"])

    logits = {n["node_id"]: (n.get("clerp", ""), float(n["token_prob"] or 0))
              for n in d["nodes"] if n["feature_type"] == "logit"}
    top = max(logits, key=lambda k: logits[k][1])
    m = re.search(r'Output "(.*?)"', logits[top][0])
    top_tok = (m.group(1) if m else "").strip().lower()
    answered = top_tok == CAPS[country]

    into_last = from_cpos_last = 0.0
    top_in = top_in_cpos = 0.0
    for e in d["links"]:
        s, t, w = e["source"], e["target"], abs(e["weight"])
        sk = kind_of.get(s)
        if kind_of.get(t) == "cross layer transcoder" and pos_of.get(t) == last:
            into_last += w
            if sk in ("cross layer transcoder", "embedding") and pos_of.get(s) in cpos:
                from_cpos_last += w
        if t == top and sk in ("cross layer transcoder", "embedding"):
            top_in += w
            if pos_of.get(s) in cpos:
                top_in_cpos += w

    return {
        "answered": answered, "top_tok": top_tok,
        "transport": from_cpos_last / into_last if into_last else 0,
        "logit_direct": top_in_cpos / top_in if top_in else 0,
    }


def main() -> None:
    print(f"{'国':10s} {'表現':4s} {'top':10s} {'答えた':6s} {'輸送':>8s} {'logit直結':>10s}")
    res = {}
    for p in PHRASINGS:
        for c in CAPS:
            slug = f"qwen06-{p.lower()}-{c.lower()}"
            r = analyze(slug, c)
            res[(c, p)] = r
            print(f"{c:10s} {p:4s} {r['top_tok']:10s} {('YES' if r['answered'] else 'no'):6s}"
                  f" {r['transport']:8.3f} {r['logit_direct']:10.3f}")

    for key in ["transport", "logit_direct"]:
        yes = [r[key] for r in res.values() if r["answered"]]
        no = [r[key] for r in res.values() if not r["answered"]]
        print(f"\n{key}: 答えた群 mean={sum(yes)/len(yes):.3f} (n={len(yes)})"
              f" / 答えない群 mean={sum(no)/len(no):.3f} (n={len(no)})")

    # H失敗(Germany)の「考えたのに負ける」検証: 首都logitへの国位置流入
    d = json.loads((OUT / "qwen06-h-germany.json").read_text(encoding="utf-8"))
    toks = d["metadata"]["prompt_tokens"]
    cpos = {i for i, t in enumerate(toks) if "germany" in norm_tok(t)}
    pos_of, kind_of = {}, {}
    for n in d["nodes"]:
        kind_of[n["node_id"]] = n["feature_type"]
        if n["feature_type"] == "cross layer transcoder":
            pos_of[n["node_id"]] = int(n["node_id"].split("_")[2])
        elif n["feature_type"] == "embedding":
            pos_of[n["node_id"]] = int(n["ctx_idx"])
    logits = {n["node_id"]: (n.get("clerp", ""), float(n["token_prob"] or 0))
              for n in d["nodes"] if n["feature_type"] == "logit"}
    cap_nodes = [k for k, (c, _) in logits.items() if "berlin" in c.lower()]
    inflow_tot, inflow_c = defaultdict(float), defaultdict(float)
    for e in d["links"]:
        t = e["target"]
        if t in logits:
            w = abs(e["weight"])
            inflow_tot[t] += w
            if pos_of.get(e["source"]) in cpos:
                inflow_c[t] += w
    print("\nH-Germany(失敗例)の首都候補:")
    if cap_nodes:
        for cn in cap_nodes:
            print(f"  {logits[cn][0]} | 国位置流入シェア={inflow_c[cn]/inflow_tot[cn]:.3f}")
    else:
        print("  Berlinはtop10に不在")


if __name__ == "__main__":
    main()
