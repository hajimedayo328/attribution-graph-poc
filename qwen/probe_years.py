# 実験S: 統一テーマ「読み出しの失敗」の第3タスク検証 — 数値の事実想起(完成年・発生年)
# 内部プローブが出力より正確なら、想起型の数値知識でも「知っているのに言えない」が成立
import re

import numpy as np
import torch
from transformer_lens import HookedTransformer

# (項目, 年) — 有名で一義的な事実のみ
FACTS = [
    ("The Eiffel Tower was completed in", 1889),
    ("The Empire State Building was completed in", 1931),
    ("The Golden Gate Bridge was completed in", 1937),
    ("The Sydney Opera House was completed in", 1973),
    ("The Burj Khalifa was completed in", 2010),
    ("The Titanic sank in", 1912),
    ("World War I began in", 1914),
    ("World War I ended in", 1918),
    ("World War II began in", 1939),
    ("World War II ended in", 1945),
    ("The Berlin Wall fell in", 1989),
    ("The Berlin Wall was built in", 1961),
    ("The first moon landing happened in", 1969),
    ("The French Revolution began in", 1789),
    ("The American Declaration of Independence was signed in", 1776),
    ("The Soviet Union collapsed in", 1991),
    ("The United Nations was founded in", 1945),
    ("NATO was founded in", 1949),
    ("The European Union was established by the Maastricht Treaty in", 1992),
    ("The first iPhone was released in", 2007),
    ("Google was founded in", 1998),
    ("Facebook was founded in", 2004),
    ("Microsoft was founded in", 1975),
    ("Apple was founded in", 1976),
    ("Amazon was founded in", 1994),
    ("YouTube was founded in", 2005),
    ("Twitter was founded in", 2006),
    ("Tesla was founded in", 2003),
    ("The World Wide Web was invented in", 1989),
    ("The first Harry Potter book was published in", 1997),
    ("The Beatles released their first album in", 1963),
    ("Elvis Presley died in", 1977),
    ("John F. Kennedy was assassinated in", 1963),
    ("Abraham Lincoln was assassinated in", 1865),
    ("The American Civil War began in", 1861),
    ("The American Civil War ended in", 1865),
    ("Christopher Columbus reached the Americas in", 1492),
    ("The Great Fire of London happened in", 1666),
    ("The first Olympic Games of the modern era were held in", 1896),
    ("The Chernobyl disaster happened in", 1986),
    ("The Fukushima nuclear disaster happened in", 2011),
    ("The September 11 attacks happened in", 2001),
    ("The first man-made satellite Sputnik was launched in", 1957),
    ("The Wright brothers made their first flight in", 1903),
    ("Albert Einstein published his theory of special relativity in", 1905),
    ("Charles Darwin published On the Origin of Species in", 1859),
    ("Isaac Newton published the Principia in", 1687),
    ("The printing press was invented by Gutenberg around", 1440),
    ("The first COVID-19 cases were reported in", 2019),
    ("Barack Obama became president of the United States in", 2009),
    ("Donald Trump first became president of the United States in", 2017),
    ("Nelson Mandela was released from prison in", 1990),
    ("The Tokyo Olympics (delayed by the pandemic) were held in", 2021),
    ("The Meiji Restoration in Japan began in", 1868),
    ("The atomic bomb was dropped on Hiroshima in", 1945),
    ("The Russian Revolution happened in", 1917),
    ("Queen Elizabeth II died in", 2022),
    ("The first Star Wars movie was released in", 1977),
    ("The Sagrada Familia construction began in", 1882),
    ("Mount Everest was first climbed in", 1953),
]

FS = "Fact: The Statue of Liberty was completed in 1886. Fact: {} "
PROBE_LAYERS = [4, 8, 12, 16, 20, 24, 27]


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()

    X = {layer: [] for layer in PROBE_LAYERS}
    y_true, y_out = [], []
    for stmt, year in FACTS:
        prompt = FS.format(stmt)
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(toks)
            # 4トークン生成して年をパース
            gen = toks
            for _ in range(4):
                nxt = model(gen)[0, -1].argmax()
                gen = torch.cat([gen, nxt.view(1, 1)], dim=1)
            out_text = model.to_string(gen[0, toks.shape[1]:])
        m = re.search(r"(1[0-9]{3}|20[0-2][0-9])", out_text)
        y_out.append(int(m.group(1)) if m else -1)
        y_true.append(year)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks)
        for layer in PROBE_LAYERS:
            X[layer].append(cache["resid_post", layer][0, -1].float().cpu().numpy())
        del cache

    y_true = np.array(y_true)
    y_out = np.array(y_out)
    valid = y_out > 0
    print(f"n={len(FACTS)} 件 (年をパースできた出力: {valid.sum()})")
    print(f"モデル出力: 正答率={(y_out == y_true).mean():.2f}"
          f" ±2年以内={(np.abs(y_out - y_true) <= 2).mean():.2f}"
          f" 平均絶対誤差={np.abs(y_out[valid] - y_true[valid]).mean():.1f}年\n")

    rng = np.random.RandomState(0)
    print(f"{'層':>4s} {'±2年以内':>9s} {'MAE(年)':>8s} {'R2':>7s}")
    for layer in PROBE_LAYERS:
        Xl = np.stack(X[layer])
        closes, maes, r2s = [], [], []
        for seed in range(5):
            idx = rng.permutation(len(FACTS))
            n_tr = int(len(FACTS) * 0.8)
            tr, te = idx[:n_tr], idx[n_tr:]
            Xtr, Xte = Xl[tr], Xl[te]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
            A = Xtr.T @ Xtr + 10.0 * np.eye(Xtr.shape[1])
            wv = np.linalg.solve(A, Xtr.T @ (y_true[tr] - y_true[tr].mean()))
            pred = Xte @ wv + y_true[tr].mean()
            closes.append((np.abs(pred - y_true[te]) <= 2).mean())
            maes.append(np.abs(pred - y_true[te]).mean())
            ss_res = ((pred - y_true[te]) ** 2).sum()
            ss_tot = ((y_true[te] - y_true[te].mean()) ** 2).sum()
            r2s.append(1 - ss_res / ss_tot)
        print(f"L{layer:>3d} {np.mean(closes):9.2f} {np.mean(maes):8.1f} {np.mean(r2s):7.2f}")


if __name__ == "__main__":
    main()
