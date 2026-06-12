# 実験Q: attentionの直接観測(Qwen3-0.6B, TransformerLens)
# 仮説: 「答えられる」とき、最終位置が国トークンを強くattendする
import re

import torch
from transformer_lens import HookedTransformer

COUNTRIES = ["France", "Japan", "Germany", "Egypt"]
CAPS = {"France": "paris", "Japan": "tokyo", "Germany": "berlin", "Egypt": "cairo"}
PHRASINGS = {
    "F": "Fact: The capital of {} is",
    "G": "{}'s capital city is",
    "H": "The city that serves as the capital of {} is",
}


def norm(t):
    return re.sub(r"[^a-z]", "", t.lower())


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"layers={n_layers}")

    rows = []
    for p, tmpl in PHRASINGS.items():
        for c in COUNTRIES:
            prompt = tmpl.format(c)
            toks = model.to_tokens(prompt)
            strs = model.to_str_tokens(prompt)
            cpos = [i for i, t in enumerate(strs) if c.lower() in norm(t)]
            last = toks.shape[1] - 1
            with torch.no_grad():
                logits, cache = model.run_with_cache(toks)
            top_tok = model.to_string(logits[0, -1].argmax()).strip().lower()
            answered = top_tok == CAPS[c]
            # 最終位置→国トークンへのattention(全ヘッド平均、層ごと)
            per_layer = []
            for layer in range(n_layers):
                att = cache["pattern", layer][0]  # [head, q, k]
                mass = att[:, last, cpos].sum(dim=-1).mean().item()
                per_layer.append(mass)
            total = sum(per_layer) / n_layers
            rows.append({"c": c, "p": p, "answered": answered, "top": top_tok,
                         "attn": total, "per_layer": per_layer})
            print(f"{c:8s} {p}  top={top_tok!r:12s} answered={answered}  attn(平均)={total:.4f}")
            del cache

    yes = [r["attn"] for r in rows if r["answered"]]
    no = [r["attn"] for r in rows if not r["answered"]]
    print(f"\n最終位置→国トークンattention: 答えた群 mean={sum(yes)/len(yes):.4f} (n={len(yes)})"
          f" / 答えない群 mean={sum(no)/len(no):.4f} (n={len(no)})")

    # 層プロファイル(答えた群と答えない群の差が大きい層)
    n_layers = len(rows[0]["per_layer"])
    print("\n層ごとの差(答えた群mean - 答えない群mean):")
    for layer in range(n_layers):
        ym = sum(r["per_layer"][layer] for r in rows if r["answered"]) / len(yes)
        nm = sum(r["per_layer"][layer] for r in rows if not r["answered"]) / len(no)
        bar = "#" * int(max(0, (ym - nm)) * 200)
        print(f"  L{layer:2d}: {ym - nm:+.4f} {bar}")


if __name__ == "__main__":
    main()
