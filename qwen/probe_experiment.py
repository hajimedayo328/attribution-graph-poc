# 実験R: 線形プローブ — モデルは内部で「正しい文字数」を知っているか
# Qwen3-0.6B、few-shot counting。各層の最終位置residual streamから文字数を線形回帰で予測。
# プローブ精度 >> 出力精度 なら「内部では知っているが出力で壊れる」。
import re

import numpy as np
import torch
from transformer_lens import HookedTransformer

WORDS = """cat dog sun pen egg ice fox cup hat bed box car map net oil
fish milk door rain tree book lamp ring snow wolf bird cake desk fork gold
bread chair table cloud house water apple lemon tiger horse plant stone sugar dream
garden window yellow forest summer winter bottle candle dragon flower guitar jungle
teacher bicycle evening library kitchen morning picture rainbow station thunder
hospital sunshine notebook airplane mountain elephant computer painting sandwich shoulder
breakfast telephone classroom crocodile adventure butterfly chocolate dangerous""".split()

FS = "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word '{}' has "
PROBE_LAYERS = [4, 8, 12, 16, 20, 24, 27]


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()

    X = {layer: [] for layer in PROBE_LAYERS}
    y_true, y_out = [], []
    for w in WORDS:
        toks = model.to_tokens(FS.format(w))
        with torch.no_grad():
            logits, cache = model.run_with_cache(toks)
        out_tok = model.to_string(logits[0, -1].argmax()).strip()
        m = re.match(r"^(\d)", out_tok)
        y_out.append(int(m.group(1)) if m else -1)
        y_true.append(len(w))
        for layer in PROBE_LAYERS:
            X[layer].append(cache["resid_post", layer][0, -1].float().cpu().numpy())
        del cache

    y_true = np.array(y_true)
    y_out = np.array(y_out)
    out_acc = (y_out == y_true).mean()
    out_close = (np.abs(y_out - y_true) <= 1).mean()
    print(f"n={len(WORDS)} 語(3〜9文字)")
    print(f"モデル出力の正答率: {out_acc:.2f} (±1以内 {out_close:.2f})\n")

    # 8:2分割を5シードで平均(リッジ回帰)
    rng = np.random.RandomState(0)
    print(f"{'層':>4s} {'プローブ正答率':>12s} {'±1以内':>8s} {'R2':>7s}")
    for layer in PROBE_LAYERS:
        Xl = np.stack(X[layer])
        accs, closes, r2s = [], [], []
        for seed in range(5):
            idx = rng.permutation(len(WORDS))
            n_tr = int(len(WORDS) * 0.8)
            tr, te = idx[:n_tr], idx[n_tr:]
            Xtr, Xte = Xl[tr], Xl[te]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
            lam = 10.0
            A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
            wvec = np.linalg.solve(A, Xtr.T @ (y_true[tr] - y_true[tr].mean()))
            pred = Xte @ wvec + y_true[tr].mean()
            accs.append((np.round(pred) == y_true[te]).mean())
            closes.append((np.abs(np.round(pred) - y_true[te]) <= 1).mean())
            ss_res = ((pred - y_true[te]) ** 2).sum()
            ss_tot = ((y_true[te] - y_true[te].mean()) ** 2).sum()
            r2s.append(1 - ss_res / ss_tot)
        print(f"L{layer:>3d} {np.mean(accs):12.2f} {np.mean(closes):8.2f} {np.mean(r2s):7.2f}")


if __name__ == "__main__":
    main()
