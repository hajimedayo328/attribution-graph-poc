# 実験U: 「消える正解」の犯人探し
# lost case(中間層で正解がtop-1→最終層で誤答)について、
# (誤答logit - 正解logit) への各層・各部品(attn/MLP)の寄与を分解する
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


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()
    n_layers = model.cfg.n_layers

    lost = []
    contribs = []  # [case, layer, 2(attn/mlp)]
    head_contrib_sum = None  # 最重要層のhead分解用に後で
    for w in WORDS:
        prompt = FS.format(w)
        toks = model.to_tokens(prompt)
        correct_tid = model.to_tokens(str(len(w)), prepend_bos=False)[0, 0].item()
        with torch.no_grad():
            logits, cache = model.run_with_cache(toks)
        out_tid = logits[0, -1].argmax().item()
        if out_tid == correct_tid:
            del cache
            continue
        # 中間層で正解がtop-1だったか(logit lens)
        resid_stack = torch.stack([cache["resid_post", layer][0, -1] for layer in range(n_layers)])
        ll = model.ln_final(resid_stack) @ model.W_U + model.b_U
        was_top = (ll.argmax(dim=-1) == correct_tid).any().item()
        if not was_top:
            del cache
            continue
        # lost case: 各層成分の (wrong - correct) 方向への寄与
        scale = cache["ln_final.hook_scale"][0, -1]  # 最終LNのスケール近似
        d_vec = (model.W_U[:, out_tid] - model.W_U[:, correct_tid]).float()
        per_layer = []
        for layer in range(n_layers):
            a = (cache["attn_out", layer][0, -1] / scale).float()
            m = (cache["mlp_out", layer][0, -1] / scale).float()
            per_layer.append([
                (a @ d_vec).item(),
                (m @ d_vec).item(),
            ])
        contribs.append(per_layer)
        lost.append((w, len(w), model.to_string(torch.tensor([out_tid]))))
        del cache

    contribs = np.array(contribs)  # [case, layer, 2]
    print(f"lost cases: {len(lost)}件")
    print("例:", lost[:8])

    mean_c = contribs.mean(axis=0)  # [layer, 2]
    print(f"\n各層の(誤答-正解)logit差への平均寄与(正=誤答を押す):")
    print(f"{'層':>4s} {'attn':>8s} {'MLP':>8s}")
    for layer in range(n_layers):
        a, m = mean_c[layer]
        mark = " ←犯人候補" if max(a, m) == mean_c.max() else ""
        print(f"L{layer:>3d} {a:+8.3f} {m:+8.3f}{mark}")

    top = np.unravel_index(mean_c.argmax(), mean_c.shape)
    comp = ["attention", "MLP"][top[1]]
    print(f"\n最大の押し上げ: L{top[0]} の {comp} (平均{mean_c[top]:+.3f})")
    total_pos = mean_c.clip(min=0).sum()
    print(f"この部品の寄与シェア: {mean_c[top]/total_pos:.0%}(全正寄与中)")


if __name__ == "__main__":
    main()
