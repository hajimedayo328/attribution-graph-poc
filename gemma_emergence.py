# 実験T-3(A): gemmaでの「答えが組み上がる層」— logit lens普遍性の別アーキ代理検証
# gemma本体はF32でローカルロード不可(ram_guard)。代わりに既存のgemma counting帰属グラフ(category N, 40枚)
# から「答えロジットを支える特徴の層分布」を測る。
# 注意: これはlogit lens(残差の復号層)そのものではなく「答えがどの層の特徴で組み上がるか」の代理。
#       Qwenのlogit lens emergence(相対深度82-89%)と"整合するか"を見る目的。
import csv
import json
from pathlib import Path

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
N_LAYERS = 26  # gemma-2-2b (層0-25)


def digit_of(clerp):
    try:
        return clerp.split('"')[1].strip()
    except Exception:  # noqa: BLE001
        return ""


def analyze(slug, expected):
    d = json.loads((BATCH / f"{slug}.json").read_text(encoding="utf-8"))
    kind = {n["node_id"]: n["feature_type"] for n in d["nodes"]}
    logits = {n["node_id"]: (n.get("clerp", ""), float(n["token_prob"] or 0))
              for n in d["nodes"] if n["feature_type"] == "logit"}
    if not logits:
        return None
    top = max(logits, key=lambda k: logits[k][1])
    correct = digit_of(logits[top][0]) == str(expected)
    # top logitへ流入する transcoder特徴の層ごと|weight|合計
    per_layer = {}
    for e in d["links"]:
        if e["target"] == top and kind.get(e["source"]) == "cross layer transcoder":
            L = int(e["source"].split("_")[0])
            per_layer[L] = per_layer.get(L, 0.0) + abs(e["weight"])
    if not per_layer:
        return None
    tot = sum(per_layer.values())
    centroid = sum(L * w for L, w in per_layer.items()) / tot      # 加重平均層
    peak = max(per_layer, key=per_layer.get)                        # 最大支持層
    # 後半(相対深度>=0.8)の層からの支持シェア
    late_share = sum(w for L, w in per_layer.items() if L / (N_LAYERS - 1) >= 0.8) / tot
    return {"correct": correct, "centroid": centroid, "peak": peak, "late_share": late_share}


def main() -> None:
    rows = [r for r in csv.DictReader(open(BASE / "data" / "manifest.csv", encoding="utf-8"))
            if r["category"] == "N"]
    res = [r2 for r2 in (analyze(r["slug"], r["expected"]) for r in rows) if r2]
    n = len(res)

    def stat(items, key):
        v = [x[key] for x in items]
        return sum(v) / len(v) if v else 0.0

    print(f"gemma-2-2b counting帰属グラフ n={n} (層0-{N_LAYERS-1})")
    print("「答えロジットを支える特徴」の層分布 = 答えがどこで組み上がるかの代理指標\n")
    for label, sub in [("全体", res),
                       ("正答", [x for x in res if x["correct"]]),
                       ("誤答", [x for x in res if not x["correct"]])]:
        if not sub:
            continue
        c = stat(sub, "centroid")
        p = stat(sub, "peak")
        ls = stat(sub, "late_share")
        print(f"[{label} n={len(sub)}] 加重平均層={c:.1f}(相対{c/(N_LAYERS-1)*100:.0f}%) "
              f"最大支持層(平均)={p:.1f}(相対{p/(N_LAYERS-1)*100:.0f}%) 後半80%からの支持={ls*100:.0f}%")

    print("\n--- Qwen logit lensとの比較 ---")
    print("Qwen 0.6/1.7/4B: 正解の浮上(logit lens)は相対深度82-89%")
    print("→ gemmaの『答えが組み上がる層』も終盤に寄れば、late-emergenceは別アーキでも整合(代理指標)")


if __name__ == "__main__":
    main()
