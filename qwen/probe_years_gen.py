# 実験S-2: 数値2系統説の追検証 — 年は「生成途中」なら線形に読めるか
# 実験Sは prompt最終位置(=生成1トークン目を予測する位置)で年をプローブ→R²0.51(読めない)。
# 仮説: 年は深層でカテゴリカルに具現化するので、prompt位置ではまだ線形でない。が、モデルが年の
# 1桁目("19"等)を出した「後」の位置なら、残り(下2桁)が線形に読めるようになるのでは?
# 検証: 正しい年をteacher-forcingで流し込み、生成step0/1/2のresidualで「真の年」をプローブしR²比較。
import numpy as np
import torch
from transformer_lens import HookedTransformer

from probe_years import FACTS, FS  # 同一データ・同一プロンプトを流用

PROBE_LAYERS = [4, 12, 20, 24, 27]
STEPS = [0, 1, 2]  # step0=prompt最終位置(実験Sと同じ), step1=年tok0の位置, step2=年tok1の位置


def ridge_r2(Xl, y, n_rep=40):
    rng = np.random.RandomState(0)
    maes, r2s = [], []
    for _ in range(n_rep):
        idx = rng.permutation(len(y))
        n_tr = int(len(y) * 0.8)
        tr, te = idx[:n_tr], idx[n_tr:]
        Xtr, Xte = Xl[tr], Xl[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        A = Xtr.T @ Xtr + 10.0 * np.eye(Xtr.shape[1])
        wv = np.linalg.solve(A, Xtr.T @ (y[tr] - y[tr].mean()))
        pred = Xte @ wv + y[tr].mean()
        maes.append(np.abs(pred - y[te]).mean())
        ss_res = ((pred - y[te]) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot)
    return np.mean(maes), np.mean(r2s), np.std(r2s)


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()

    # (step, layer) -> list of residual vectors。全factで全stepが取れたものだけ採用
    X = {(s, layer): [] for s in STEPS for layer in PROBE_LAYERS}
    y_true = []
    ntok = []
    for stmt, year in FACTS:
        prompt = model.to_tokens(FS.format(stmt))
        ytok = model.to_tokens(" " + str(year), prepend_bos=False)  # 年の継続トークン
        ntok.append(ytok.shape[1])
        full = torch.cat([prompt, ytok], dim=1)
        p0 = prompt.shape[1] - 1  # prompt最終位置
        if p0 + max(STEPS) >= full.shape[1]:
            continue  # 年が短くstep2が取れない場合は除外(整合性のため)
        with torch.no_grad():
            _, cache = model.run_with_cache(full)
        for s in STEPS:
            for layer in PROBE_LAYERS:
                X[(s, layer)].append(cache["resid_post", layer][0, p0 + s].float().cpu().numpy())
        del cache
        y_true.append(year)

    y = np.array(y_true)
    print(f"採用 n={len(y)} 件 / 年トークン数の分布: "
          f"{ {t: ntok.count(t) for t in sorted(set(ntok))} }")
    print("step0=prompt最終(実験Sと同じ) / step1=年1トークン目を出した後 / step2=2トークン目後\n")
    print(f"{'層':>4s}" + "".join(f"   step{s}_R2(±sd)" for s in STEPS) + "   (40分割平均, R²↑=線形に読める)")
    for layer in PROBE_LAYERS:
        line = f"L{layer:>3d}"
        for s in STEPS:
            _, r2, sd = ridge_r2(np.stack(X[(s, layer)]), y)
            line += f"  {r2:5.2f}±{sd:.2f}"
        print(line)

    print("\n--- 各stepの最良層R²(±sd) ---")
    for s in STEPS:
        vals = [ridge_r2(np.stack(X[(s, layer)]), y) for layer in PROBE_LAYERS]
        best = max(vals, key=lambda v: v[1])
        print(f"  step{s}: 最良R²={best[1]:.2f}±{best[2]:.2f}")
    print("\n注: 実験Sは年R²=0.51と報告。分割乱数で0.5〜0.7に振れる(n=60で不安定)。"
          "countingのR²0.94(実験R)と比べ、年は『読めない』というより『中程度に読める』が正確。")


if __name__ == "__main__":
    main()
