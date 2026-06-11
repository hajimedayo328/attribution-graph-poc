# 実験E: 共有回路の意味照合
# 1) カテゴリのコア特徴(9/10以上のグラフで活性)を抽出
# 2) Fact:回路 = A/F双子ペアの差分で一貫して現れる特徴
# 3) 作話候補 = Bのコア特徴のうち他カテゴリにほぼ出ないもの
# 4) Neuronpedia APIでauto-interp説明を取得して意味を照合
import csv
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
MANIFEST = BASE / "data" / "manifest.csv"
CACHE = BASE / "data" / "feature_explanations.json"

EXPLAIN_LIMIT = 12  # 1セットあたりの説明取得上限(API礼儀)


def load_fsets():
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f))
    fsets, cats = {}, {}
    for m in manifest:
        path = BATCH / f"{m['slug']}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        # node_id = "{layer}_{層内特徴index}_{ctx}" — APIで照会できるのは層内index
        fs = set()
        for n in d["nodes"]:
            if n["feature_type"] != "cross layer transcoder":
                continue
            layer, feat, _ = n["node_id"].split("_")
            fs.add((layer, feat))
        fsets[m["slug"]] = fs
        cats[m["slug"]] = m["category"]
    return fsets, cats


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
    fsets, cats = load_fsets()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    by_cat = defaultdict(list)
    for slug, c in cats.items():
        if c in "ABCDEF":
            by_cat[c].append(slug)

    # 1) カテゴリコア特徴
    core = {}
    for c, slugs in by_cat.items():
        cnt = Counter()
        for s in slugs:
            cnt.update(fsets[s])
        core[c] = {f for f, n in cnt.items() if n >= 9}
        print(f"core[{c}]: {len(core[c])} features (9/10以上で活性)")

    # 全カテゴリ共通(背景回路)
    universal = set.intersection(*core.values())
    print(f"\n全6カテゴリ共通のコア特徴(背景回路): {len(universal)}個")

    # 2) Fact:回路: A/F双子ペア(国が同じ)の差分
    a_slugs = sorted(by_cat["A"])  # a00..a09 国順
    f_slugs = sorted(by_cat["F"])  # f40..f49 同じ国順
    diff_cnt, rem_cnt = Counter(), Counter()
    for a, f in zip(a_slugs, f_slugs):
        diff_cnt.update(fsets[f] - fsets[a])  # Fact:で増えた
        rem_cnt.update(fsets[a] - fsets[f])   # Fact:で消えた
    fact_added = [f for f, n in diff_cnt.items() if n >= 8]
    fact_removed = [f for f, n in rem_cnt.items() if n >= 8]
    print(f"\nFact:付与で一貫して追加される特徴(8/10ペア以上): {len(fact_added)}個")
    print(f"Fact:付与で一貫して消える特徴(8/10ペア以上): {len(fact_removed)}個")

    # 3) 作話候補: Bコアのうち、他カテゴリのグラフにほぼ出ない特徴
    other_cnt = Counter()
    for c in "ACDEF":
        for s in by_cat[c]:
            other_cnt.update(fsets[s])
    confab = [f for f in core["B"] if other_cnt[f] <= 5]  # 他50枚中5枚以下
    print(f"B(架空)コアのうち他カテゴリにほぼ出ない特徴: {len(confab)}個")

    # 4) 説明の取得と表示
    def show(title, feats):
        print(f"\n===== {title} =====")
        # L0はトークン検出器なので、意味処理が起きる深い層から表示
        for layer, feat in sorted(feats, key=lambda x: -int(x[0]))[:EXPLAIN_LIMIT]:
            desc = fetch_explanation(layer, feat, cache)
            print(f"  L{layer:>2}/{feat:<6} {desc}")
        if len(feats) > EXPLAIN_LIMIT:
            print(f"  ...他{len(feats)-EXPLAIN_LIMIT}個は data/circuits.json 参照")

    show("Fact:付与で追加される回路", fact_added)
    show("Fact:付与で消える回路", fact_removed)
    show("作話(confabulation)候補回路 [B架空のみ]", confab)
    show("全カテゴリ共通の背景回路(参考)", sorted(universal)[:EXPLAIN_LIMIT])

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    (BASE / "data" / "circuits.json").write_text(json.dumps({
        "universal": sorted(map(list, universal)),
        "fact_added": sorted(map(list, fact_added)),
        "fact_removed": sorted(map(list, fact_removed)),
        "confab_candidates": sorted(map(list, confab)),
        "core_sizes": {c: len(v) for c, v in core.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsaved: data/circuits.json, data/feature_explanations.json")


if __name__ == "__main__":
    main()
