# 実験O: countingエラーのノードレベル分析
# 要約統計で捕まらなかった幻覚を、「どの特徴が活性したか」のレベルで探す
# 1) 正答/誤答グラフを弁別する特徴(出現頻度差)を抽出し意味照合
# 2) 「7に飽和」現象: 7と答えた誤答グラフに共通する特徴
import csv
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
MANIFEST = BASE / "data" / "manifest.csv"
CACHE = BASE / "data" / "feature_explanations.json"


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
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        manifest = [r for r in csv.DictReader(f) if r["category"] == "N"]

    graphs = []
    for m in manifest:
        path = BATCH / f"{m['slug']}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        toks = d["metadata"]["prompt_tokens"]
        last = len(toks) - 1
        logits = [n for n in d["nodes"] if n["feature_type"] == "logit"]
        top = max(logits, key=lambda n: float(n["token_prob"] or 0))
        mt = re.search(r'Output "(.*?)"', top.get("clerp", ""))
        pred = (mt.group(1) if mt else "").strip()
        expected = m["expected"].split("|")[0]
        fs_all, fs_last = set(), set()
        for n in d["nodes"]:
            if n["feature_type"] != "cross layer transcoder":
                continue
            layer, feat, ctx = n["node_id"].split("_")
            fs_all.add((layer, feat))
            if int(ctx) == last:
                fs_last.add((layer, feat))
        graphs.append({"slug": m["slug"], "pred": pred, "expected": expected,
                       "correct": pred == expected, "all": fs_all, "last": fs_last})

    cor = [g for g in graphs if g["correct"]]
    inc = [g for g in graphs if not g["correct"]]
    print(f"n={len(graphs)} 正答={len(cor)} 誤答={len(inc)}")

    # 1) 弁別特徴: 誤答の70%以上に出現 かつ 正答の30%以下(とその逆)
    def freq(group, scope):
        c = Counter()
        for g in group:
            c.update(g[scope])
        return c

    for scope, label in [("all", "全位置"), ("last", "最終位置のみ")]:
        fc, fi = freq(cor, scope), freq(inc, scope)
        err_only = [(f, fi[f] / len(inc), fc[f] / len(cor)) for f in fi
                    if fi[f] / len(inc) >= 0.7 and fc[f] / len(cor) <= 0.3]
        cor_only = [(f, fc[f] / len(cor), fi[f] / len(inc)) for f in fc
                    if fc[f] / len(cor) >= 0.7 and fi[f] / len(inc) <= 0.3]
        print(f"\n===== [{label}] 誤答特異的特徴: {len(err_only)}個 / 正答特異的: {len(cor_only)}個 =====")
        for name, items in [("誤答に出る(正答に出ない)", err_only), ("正答に出る(誤答に出ない)", cor_only)]:
            print(f"--- {name} ---")
            for (layer, feat), p1, p2 in sorted(items, key=lambda x: -(x[1] - x[2]))[:6]:
                desc = fetch_explanation(layer, feat, cache)
                print(f"  L{layer:>2}/{feat:<6} ({p1:.0%} vs {p2:.0%}) {desc}")

    # 2) 「7に飽和」: 7と誤答したグラフ vs 7が正解で7と答えたグラフ
    sat7 = [g for g in graphs if g["pred"] == "7" and not g["correct"]]
    true7 = [g for g in graphs if g["pred"] == "7" and g["correct"]]
    print(f"\n===== 「7」誤答={len(sat7)}枚 vs 「7」正答={len(true7)}枚 =====")
    if sat7 and true7:
        f_sat, f_true = freq(sat7, "last"), freq(true7, "last")
        only_sat = [(f, f_sat[f] / len(sat7)) for f in f_sat
                    if f_sat[f] / len(sat7) >= 0.8 and f_true[f] / len(true7) <= 0.2]
        print(f"飽和誤答に特異的(最終位置): {len(only_sat)}個")
        for (layer, feat), p in sorted(only_sat, key=lambda x: -x[1])[:6]:
            print(f"  L{layer:>2}/{feat:<6} ({p:.0%}) {fetch_explanation(layer, feat, cache)}")

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
