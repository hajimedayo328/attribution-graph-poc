# 実験T: logit lens — 答えはどの層で「具現化」するか
# counting(計算系)と年想起(記憶系)で、正解トークンが浮上する深さを比較する。
# 副問: countingの誤答では「中間層では正解がtop-1だったのに最終層で失われる」が起きるか
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
FS_COUNT = "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word '{}' has "

FACTS = [
    ("The Eiffel Tower was completed in", 1889), ("The Empire State Building was completed in", 1931),
    ("The Golden Gate Bridge was completed in", 1937), ("The Sydney Opera House was completed in", 1973),
    ("The Titanic sank in", 1912), ("World War I began in", 1914), ("World War II ended in", 1945),
    ("The Berlin Wall fell in", 1989), ("The first moon landing happened in", 1969),
    ("The French Revolution began in", 1789), ("The Soviet Union collapsed in", 1991),
    ("The first iPhone was released in", 2007), ("Google was founded in", 1998),
    ("Facebook was founded in", 2004), ("Microsoft was founded in", 1975),
    ("Apple was founded in", 1976), ("Amazon was founded in", 1994),
    ("John F. Kennedy was assassinated in", 1963), ("Christopher Columbus reached the Americas in", 1492),
    ("The Chernobyl disaster happened in", 1986), ("The September 11 attacks happened in", 2001),
    ("The Wright brothers made their first flight in", 1903),
    ("The atomic bomb was dropped on Hiroshima in", 1945), ("The Russian Revolution happened in", 1917),
    ("Queen Elizabeth II died in", 1977 + 45), ("The first Star Wars movie was released in", 1977),
    ("Mount Everest was first climbed in", 1953), ("Elvis Presley died in", 1977),
    ("The Great Fire of London happened in", 1666), ("The first Olympic Games of the modern era were held in", 1896),
]
FS_YEAR = "Fact: The Statue of Liberty was completed in 1886. Fact: {} "


def lens_run(model, prompt, target_str):
    """各層のlogit lensでtargetトークンのrankを返す。target_str例: '6' / '1889'"""
    toks = model.to_tokens(prompt)
    target_ids = model.to_tokens(target_str, prepend_bos=False)[0]
    tid = target_ids[0].item()
    with torch.no_grad():
        _, cache = model.run_with_cache(toks)
    n_layers = model.cfg.n_layers
    ranks, is_top = [], []
    resid_stack = torch.stack([cache["resid_post", layer][0, -1] for layer in range(n_layers)])
    resid_ln = model.ln_final(resid_stack)
    logits = resid_ln @ model.W_U + model.b_U  # [layer, vocab]
    for layer in range(n_layers):
        rank = (logits[layer] > logits[layer, tid]).sum().item()
        ranks.append(rank)
        is_top.append(rank == 0)
    out_tok = logits[-1].argmax().item()
    del cache
    return ranks, is_top, out_tok == tid


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()
    n_layers = model.cfg.n_layers

    results = {}
    for task, items in [("counting", [(FS_COUNT.format(w), str(len(w))) for w in WORDS]),
                        ("years", [(FS_YEAR.format(s), str(y)) for s, y in FACTS])]:
        emergence, correct_flags, lost_cases = [], [], 0
        all_ranks = []
        lost_ranks = []  # lostケース(中間でtop-1→最終層で失う)限定のrank曲線
        for prompt, target in items:
            ranks, is_top, final_correct = lens_run(model, prompt, target)
            all_ranks.append(ranks)
            correct_flags.append(final_correct)
            # 浮上層 = 正解トークンが初めてtop-1になる層(なければNone)
            em = next((layer for layer, t in enumerate(is_top) if t), None)
            emergence.append(em)
            # 「途中で正解だったのに最終層で失う」
            if not final_correct and any(is_top):
                lost_cases += 1
                lost_ranks.append(ranks)
        acc = np.mean(correct_flags)
        em_ok = [e for e, c in zip(emergence, correct_flags) if c and e is not None]
        print(f"\n===== {task} (n={len(items)}) =====")
        print(f"最終層の正答率: {acc:.2f}")
        if em_ok:
            print(f"正答ケースの浮上層: 中央値 L{int(np.median(em_ok))} (範囲 L{min(em_ok)}-L{max(em_ok)})")
        print(f"誤答のうち「中間層では正解がtop-1だったのに失った」: {lost_cases}件")
        # 平均rank曲線(対数)を粗く表示
        mr = np.median(np.array(all_ranks), axis=0)
        print("正解トークンのrank中央値(層別、全ケース):")
        for layer in range(0, n_layers, 3):
            bar = "#" * max(0, int(12 - np.log2(mr[layer] + 1)))
            print(f"  L{layer:2d}: rank={int(mr[layer]):6d} {bar}")
        # lostケース限定の全層rank中央値(⑦のグラフ用、浮上→消失を正確に示す)
        if lost_ranks:
            lr = np.median(np.array(lost_ranks), axis=0)
            print(f"[lostケース限定 n={len(lost_ranks)}] 正解トークンrank中央値(全{n_layers}層):")
            print("  " + ",".join(f"L{layer}={int(lr[layer])}" for layer in range(n_layers)))
        results[task] = (acc, em_ok)


if __name__ == "__main__":
    main()
