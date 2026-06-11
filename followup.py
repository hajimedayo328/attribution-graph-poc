# 追試1: トークン数正規化 — 「事実vs架空」の構造差はプロンプト長の交絡を除いても残るか
# 追試2: 構造指標だけで架空プロンプト(B)を当てられるか — LOO交差検証ロジスティック回帰 + AUC
import csv
import json
import math
from pathlib import Path

BASE = Path(__file__).parent
BATCH = BASE / "data" / "batch"
RESULTS = BASE / "data" / "results.csv"


def mann_whitney(a, b):
    n1, n2 = len(a), len(b)
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
    z = (u1 - mu) / sigma if sigma else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u1, p


def auc_from_scores(scores_pos, scores_neg):
    """AUC = P(score_pos > score_neg)、Mann-Whitney Uから算出"""
    u1, _ = mann_whitney(scores_pos, scores_neg)
    return u1 / (len(scores_pos) * len(scores_neg))


def logistic_train(X, y, epochs=3000, lr=0.5, l2=0.01):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = sum(wj * xj for wj, xj in zip(w, xi)) + b
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            err = p - yi
            for j in range(d):
                gw[j] += err * xi[j]
            gb += err
        w = [wj - lr * (gwj / n + l2 * wj) for wj, gwj in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def main() -> None:
    with open(RESULTS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # プロンプトのトークン数をJSONメタデータから取得
    for r in rows:
        d = json.loads((BATCH / f"{r['slug']}.json").read_text(encoding="utf-8"))
        r["n_tokens"] = len(d["metadata"]["prompt_tokens"]) if "prompt_tokens" in d["metadata"] else len(d["metadata"].get("promptTokens", []))

    for r in rows:
        r["n_nodes"] = float(r["n_nodes"])
        r["n_edges"] = float(r["n_edges"])
        r["nodes_per_token"] = r["n_nodes"] / r["n_tokens"]
        r["edges_per_token"] = r["n_edges"] / r["n_tokens"]
        for k in ["density", "degree_gini", "modularity", "error_share",
                  "mean_degree", "logit_entropy", "n_communities"]:
            r[k] = float(r[k])

    a_rows = [r for r in rows if r["category"] == "A"]
    b_rows = [r for r in rows if r["category"] == "B"]

    print("===== 追試1: トークン数の交絡チェック (A vs B) =====")
    ta = [r["n_tokens"] for r in a_rows]
    tb = [r["n_tokens"] for r in b_rows]
    print(f"A tokens: mean={sum(ta)/len(ta):.1f} {sorted(ta)}")
    print(f"B tokens: mean={sum(tb)/len(tb):.1f} {sorted(tb)}")
    _, p = mann_whitney(ta, tb)
    print(f"n_tokens A vs B: p={p:.4f}  ← 有意ならカテゴリ差はプロンプト長の疑いあり")

    print("\n正規化後の A vs B 検定:")
    tests = []
    for k in ["nodes_per_token", "edges_per_token", "density", "degree_gini",
              "modularity", "error_share", "logit_entropy"]:
        _, p = mann_whitney([r[k] for r in a_rows], [r[k] for r in b_rows])
        tests.append((k, p))
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i][1])
    adj = [0.0] * m
    prev = 1.0
    for ri in range(m - 1, -1, -1):
        i = order[ri]
        prev = min(prev, tests[i][1] * m / (ri + 1))
        adj[i] = prev
    for (k, p), q in zip(tests, adj):
        am = sum(r[k] for r in a_rows) / len(a_rows)
        bm = sum(r[k] for r in b_rows) / len(b_rows)
        flag = " *" if q < 0.05 else ""
        print(f"  {k:16s} A={am:9.3f} B={bm:9.3f} p={p:.4f} q={q:.4f}{flag}")

    print("\n===== 追試2: 構造指標だけで架空(B)を当てる LOO分類 =====")
    # 出力系(top1_prob, logit_entropy)は使わない。純構造のみ
    feats = ["nodes_per_token", "edges_per_token", "density", "degree_gini",
             "modularity", "error_share", "mean_degree"]
    y_all = [1 if r["category"] == "B" else 0 for r in rows]
    X_all = [[r[k] for k in feats] for r in rows]

    # 単一特徴のAUC(参考)
    print("単一特徴AUC (B vs その他):")
    for k in feats + ["n_tokens"]:
        pos = [r[k] for r in rows if r["category"] == "B"]
        neg = [r[k] for r in rows if r["category"] != "B"]
        auc = auc_from_scores(pos, neg)
        print(f"  {k:16s} AUC={max(auc, 1-auc):.3f} ({'B高' if auc>0.5 else 'B低'})")

    # LOO交差検証
    loo_scores = []
    for hold in range(len(rows)):
        X_tr = [x for i, x in enumerate(X_all) if i != hold]
        y_tr = [v for i, v in enumerate(y_all) if i != hold]
        # 標準化(訓練foldの統計のみ使用)
        d = len(feats)
        mus = [sum(x[j] for x in X_tr) / len(X_tr) for j in range(d)]
        sds = [max(1e-9, (sum((x[j] - mus[j]) ** 2 for x in X_tr) / len(X_tr)) ** 0.5) for j in range(d)]
        Xs = [[(x[j] - mus[j]) / sds[j] for j in range(d)] for x in X_tr]
        w, b = logistic_train(Xs, y_tr)
        xh = [(X_all[hold][j] - mus[j]) / sds[j] for j in range(d)]
        z = sum(wj * xj for wj, xj in zip(w, xh)) + b
        loo_scores.append(1 / (1 + math.exp(-max(-30, min(30, z)))))

    pos = [s for s, yv in zip(loo_scores, y_all) if yv == 1]
    neg = [s for s, yv in zip(loo_scores, y_all) if yv == 0]
    auc = auc_from_scores(pos, neg)
    print(f"\nLOOロジスティック回帰(7構造特徴) AUC = {auc:.3f}")
    thr_correct = sum(1 for s, yv in zip(loo_scores, y_all) if (s > 0.5) == bool(yv))
    print(f"閾値0.5での的中: {thr_correct}/{len(rows)} (ベースライン: 多数派予測={len(rows)-sum(y_all)}/{len(rows)})")
    miss = [(rows[i]['slug'], rows[i]['category'], round(loo_scores[i],3))
            for i in range(len(rows)) if (loo_scores[i] > 0.5) != bool(y_all[i])]
    print("誤分類:", miss)


if __name__ == "__main__":
    main()
