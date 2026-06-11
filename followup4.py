# 実験G: 内容回路の分離
# 仮説: 「国トークンの位置」に限定すれば、言い回しに埋もれた内容(国)の回路が見える
# 1) 国トークン位置の特徴だけでJaccard → 同じ国vs同じ言い回しの逆転を確認
# 2) 最終トークン位置(答えが形成される場所)でも同様に比較
# 3) フランス固有回路を抽出してNeuronpedia説明と照合
import json
import time
import urllib.request
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
CACHE = BASE / "data" / "feature_explanations.json"

COUNTRIES = {"France": "paris", "Japan": "tokyo", "Germany": "berlin", "Egypt": "cairo"}
PHRASINGS = {"F": "Fact: The capital of {} is", "G": "{}'s capital city is",
             "H": "The city that serves as the capital of {} is"}
SLUGS = {
    ("France", "F"): "agpoc-f40-hj328", ("Japan", "F"): "agpoc-f41-hj328",
    ("Germany", "F"): "agpoc-f43-hj328", ("Egypt", "F"): "agpoc-f47-hj328",
    ("France", "G"): "agpoc-g60-hj328", ("Japan", "G"): "agpoc-g61-hj328",
    ("Germany", "G"): "agpoc-g62-hj328", ("Egypt", "G"): "agpoc-g63-hj328",
    ("France", "H"): "agpoc-h64-hj328", ("Japan", "H"): "agpoc-h65-hj328",
    ("Germany", "H"): "agpoc-h66-hj328", ("Egypt", "H"): "agpoc-h67-hj328",
}


def load(slug, country):
    d = json.loads((BATCH / f"{slug}.json").read_text(encoding="utf-8"))
    toks = d["metadata"]["prompt_tokens"]
    # 国トークンの位置(部分一致、'France'や' France')
    cpos = [i for i, t in enumerate(toks) if country.lower() in t.lower().replace("▁", "")]
    last = len(toks) - 1
    at_cpos, at_last, allf = set(), set(), set()
    for n in d["nodes"]:
        if n["feature_type"] != "cross layer transcoder":
            continue
        layer, feat, ctx = n["node_id"].split("_")
        f = (layer, feat)
        allf.add(f)
        if int(ctx) in cpos:
            at_cpos.add(f)
        if int(ctx) == last:
            at_last.add(f)
    return {"cpos": at_cpos, "last": at_last, "all": allf, "n_cpos_positions": len(cpos)}


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


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


def main() -> None:
    data = {k: load(slug, k[0]) for k, slug in SLUGS.items()}
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    for scope in ["all", "cpos", "last"]:
        same_c, same_p, diff = [], [], []
        for k1, k2 in combinations(SLUGS, 2):
            j = jac(data[k1][scope], data[k2][scope])
            if k1[0] == k2[0]:
                same_c.append(j)
            elif k1[1] == k2[1]:
                same_p.append(j)
            else:
                diff.append(j)
        # 1-NN
        cn = pn = 0
        for k1 in SLUGS:
            best, bj = None, -1
            for k2 in SLUGS:
                if k1 == k2:
                    continue
                j = jac(data[k1][scope], data[k2][scope])
                if j > bj:
                    bj, best = j, k2
            cn += best[0] == k1[0]
            pn += best[1] == k1[1]
        label = {"all": "全位置(再掲)", "cpos": "国トークン位置のみ", "last": "最終トークン位置のみ"}[scope]
        print(f"\n===== {label} =====")
        print(f"同じ国・違う言い回し: {sum(same_c)/len(same_c):.3f}")
        print(f"同じ言い回し・違う国: {sum(same_p)/len(same_p):.3f}")
        print(f"両方違う:            {sum(diff)/len(diff):.3f}")
        print(f"1-NN: 同じ国 {cn}/12, 同じ言い回し {pn}/12")

    # フランス固有回路: 国位置で、フランス3表現すべてに出て、他国9枚に出ない特徴
    print("\n===== 国固有回路の抽出(国トークン位置) =====")
    for country in COUNTRIES:
        own = set.intersection(*[data[(country, p)]["cpos"] for p in "FGH"])
        others = set.union(*[data[(c, p)]["cpos"] for c in COUNTRIES for p in "FGH" if c != country])
        specific = own - others
        print(f"\n{country}固有(3表現共通かつ他国に無し): {len(specific)}個")
        for layer, feat in sorted(specific, key=lambda x: -int(x[0]))[:8]:
            print(f"  L{layer:>2}/{feat:<6} {fetch_explanation(layer, feat, cache)}")

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
