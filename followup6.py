# 実験I: 国トークン位置 → 最終位置への情報輸送
# 1) 輸送量: 国位置ノード→最終位置ノードへのエッジ重み総量(正規化)
# 2) 言い回しF/G/H(正答/誤答)で輸送量を比較 — 「答えられない言い回し」は輸送不足か
# 3) 輸送がどの層で起きるか(層プロファイル)
# 4) 最終位置で国情報を最も受け取る特徴の意味照合
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
CACHE = BASE / "data" / "feature_explanations.json"

SLUGS = {
    ("France", "F"): "agpoc-f40-hj328", ("Japan", "F"): "agpoc-f41-hj328",
    ("Germany", "F"): "agpoc-f43-hj328", ("Egypt", "F"): "agpoc-f47-hj328",
    ("France", "G"): "agpoc-g60-hj328", ("Japan", "G"): "agpoc-g61-hj328",
    ("Germany", "G"): "agpoc-g62-hj328", ("Egypt", "G"): "agpoc-g63-hj328",
    ("France", "H"): "agpoc-h64-hj328", ("Japan", "H"): "agpoc-h65-hj328",
    ("Germany", "H"): "agpoc-h66-hj328", ("Egypt", "H"): "agpoc-h67-hj328",
}
# モデルが首都名を答えたか(followup4/実験Fで確認済み)
ANSWERED = {("France", "F"): 1, ("Japan", "F"): 1, ("Germany", "F"): 1, ("Egypt", "F"): 1,
            ("France", "G"): 0, ("Japan", "G"): 0, ("Germany", "G"): 0, ("Egypt", "G"): 0,
            ("France", "H"): 0, ("Japan", "H"): 1, ("Germany", "H"): 0, ("Egypt", "H"): 1}


def fetch_explanation(layer, feature, cache):
    key = f"{layer}/{feature}"
    if key in cache:
        return cache[key]
    url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feature}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.loads(r.read())
        descs = [e.get("description", "") for e in d.get("explanations", [])]
        cache[key] = descs[0] if descs else "(no explanation)"
    except Exception as e:  # noqa: BLE001
        cache[key] = f"(fetch error: {e})"
    time.sleep(1.0)
    return cache[key]


def analyze(slug, country):
    d = json.loads((BATCH / f"{slug}.json").read_text(encoding="utf-8"))
    toks = d["metadata"]["prompt_tokens"]
    cpos = {i for i, t in enumerate(toks) if country.lower() in t.lower().replace("▁", "")}
    last = len(toks) - 1

    # ノード位置・種別の索引
    pos_of, kind_of = {}, {}
    for n in d["nodes"]:
        kind_of[n["node_id"]] = n["feature_type"]
        if n["feature_type"] == "cross layer transcoder":
            _, _, ctx = n["node_id"].split("_")
            pos_of[n["node_id"]] = int(ctx)
        elif n["feature_type"] == "embedding":
            pos_of[n["node_id"]] = int(n["ctx_idx"])
        elif n["feature_type"] == "logit":
            pos_of[n["node_id"]] = last

    into_last_total = 0.0          # 最終位置の特徴ノードへの流入総量
    from_cpos_to_last = 0.0        # うち国位置の特徴/埋め込み発
    layer_hist = Counter()         # 輸送エッジのsource層
    receiver_inflow = defaultdict(float)  # 最終位置の特徴ごとの国発流入
    top_logit_in_total = 0.0
    top_logit_in_cpos = 0.0

    logit_probs = {n["node_id"]: float(n["token_prob"] or 0) for n in d["nodes"]
                   if n["feature_type"] == "logit"}
    top_logit = max(logit_probs, key=logit_probs.get)

    for e in d["links"]:
        s, t, w = e["source"], e["target"], abs(e["weight"])
        sk, tk = kind_of.get(s), kind_of.get(t)
        # 最終位置の特徴ノードへの流入
        if tk == "cross layer transcoder" and pos_of.get(t) == last:
            into_last_total += w
            if sk in ("cross layer transcoder", "embedding") and pos_of.get(s) in cpos:
                from_cpos_to_last += w
                if sk == "cross layer transcoder":
                    layer_hist[s.split("_")[0]] += 1
                receiver_inflow[t] += w
        # top logitへの直接流入
        if t == top_logit and sk in ("cross layer transcoder", "embedding"):
            top_logit_in_total += w
            if pos_of.get(s) in cpos:
                top_logit_in_cpos += w

    return {
        "transport_share": from_cpos_to_last / into_last_total if into_last_total else 0,
        "logit_direct_share": top_logit_in_cpos / top_logit_in_total if top_logit_in_total else 0,
        "layer_hist": layer_hist,
        "receivers": receiver_inflow,
    }


def main() -> None:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    res = {k: analyze(slug, k[0]) for k, slug in SLUGS.items()}

    print("===== 輸送量(国位置→最終位置の流入シェア) =====")
    print(f"{'国':10s} {'表現':4s} {'答えた?':6s} {'輸送シェア':>10s} {'logit直結シェア':>14s}")
    for (c, p), r in sorted(res.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"{c:10s} {p:4s} {('YES' if ANSWERED[(c,p)] else 'no'):6s}"
              f" {r['transport_share']:10.3f} {r['logit_direct_share']:14.3f}")

    for key in ["transport_share", "logit_direct_share"]:
        yes = [r[key] for k, r in res.items() if ANSWERED[k]]
        no = [r[key] for k, r in res.items() if not ANSWERED[k]]
        print(f"\n{key}: 答えた群 mean={sum(yes)/len(yes):.3f} (n={len(yes)})"
              f" / 答えない群 mean={sum(no)/len(no):.3f} (n={len(no)})")

    print("\n===== 輸送エッジのsource層分布(全12グラフ合算) =====")
    total_hist = Counter()
    for r in res.values():
        total_hist.update(r["layer_hist"])
    for layer in sorted(total_hist, key=int):
        print(f"  L{layer:>2}: {'#' * (total_hist[layer] // 20)} {total_hist[layer]}")

    print("\n===== 最終位置で国情報を最も受け取る特徴(F表現4グラフ、意味照合) =====")
    recv = defaultdict(float)
    for (c, p), r in res.items():
        if p != "F":
            continue
        for node, w in r["receivers"].items():
            layer, feat, _ = node.split("_")
            recv[(layer, feat)] += w
    for (layer, feat), w in sorted(recv.items(), key=lambda x: -x[1])[:10]:
        print(f"  L{layer:>2}/{feat:<6} inflow={w:7.1f}  {fetch_explanation(layer, feat, cache)}")

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
