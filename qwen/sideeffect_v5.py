# 実験V-5: アブレーション強度の精密グリッド — Pareto曲線の膝を特定
# V-4でL24-27×0.7が純益プラス・×0.5が大損と判明。×0.6〜0.9を細かく掃いて膝を特定する。
import torch
from transformer_lens import HookedTransformer

LAYERS = [24, 25, 26, 27]
GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

WORDS = """cat dog sun pen fox fish tree book snow wolf bird cake desk fork gold
bread chair table cloud water apple lemon tiger horse plant stone sugar dream
garden window forest summer winter bottle candle dragon flower guitar jungle
teacher bicycle library kitchen morning picture rainbow station thunder hospital
sunshine notebook airplane mountain elephant computer painting sandwich breakfast""".split()
COUNT_FS = "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word '{}' has"
ADD3 = [(347,285),(523,198),(456,367),(689,254),(178,346),(592,239),(415,288),(736,159),
        (264,478),(581,267),(393,448),(627,195),(348,576),(459,283),(712,189),(536,287),
        (273,649),(418,395),(564,278),(329,486)]
ADD3_FS = "347 + 285 = 632. 158 + 247 = 405. {} + {} ="
CAPS = {"Canada":"Ottawa","Australia":"Canberra","Brazil":"Bras","Turkey":"Ankara",
        "Switzerland":"Bern","Norway":"Oslo","Portugal":"Lisbon","Greece":"Athens",
        "Poland":"Warsaw","Sweden":"Stockholm","Austria":"Vienna","Ireland":"Dublin",
        "Finland":"Helsinki","Denmark":"Copenhagen","Hungary":"Budapest","Romania":"Bucharest",
        "Vietnam":"Hanoi","Nigeria":"Abuja","Argentina":"Buenos","Chile":"Santiago",
        "Peru":"Lima","Colombia":"Bogot","Morocco":"Rabat","Kenya":"Nairobi",
        "Iran":"Tehran","Iraq":"Baghdad","Pakistan":"Islamabad","Indonesia":"Jakarta",
        "Philippines":"Manila","Ukraine":"Kyiv","Netherlands":"Amsterdam","Belgium":"Brussels"}
CAP_FS = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of {} is"


def make_hook(scale):
    def h(value, hook):
        return value * scale
    return h


def gen(model, prompt, scale, max_new=6):
    toks = model.to_tokens(prompt)
    fwd = [(f"blocks.{L}.hook_mlp_out", make_hook(scale)) for L in LAYERS] if scale != 1.0 else []
    with torch.no_grad(), model.hooks(fwd_hooks=fwd):
        out = model.generate(toks, max_new_tokens=max_new, do_sample=False, verbose=False)
    return model.to_string(out[0, toks.shape[1]:])


def acc(model, items, scale):
    return sum(exp in gen(model, p, scale) for p, exp in items) / len(items)


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()
    tasks = {
        "counting": [(COUNT_FS.format(w), str(len(w))) for w in WORDS],
        "3桁加算": [(ADD3_FS.format(a, b), str(a + b)) for a, b in ADD3],
        "首都": [(CAP_FS.format(c), a) for c, a in CAPS.items()],
    }
    print(f"L24-27 MLP scale grid (×1.0=無介入)\n")
    print(f"{'scale':>6s} {'counting':>9s} {'3桁加算':>8s} {'首都':>7s} {'Δcount':>8s} {'巻添合計':>9s} {'純益':>8s}")
    print("-" * 60)
    base = {}
    for scale in GRID:
        r = {t: acc(model, items, scale) for t, items in tasks.items()}
        if scale == 1.0:
            base = r
            print(f"{scale:>6.1f} {r['counting']:>9.3f} {r['3桁加算']:>8.3f} {r['首都']:>7.3f}"
                  f" {'—':>8s} {'—':>9s} {'—':>8s}")
            continue
        dc = r['counting'] - base['counting']
        coll = (r['3桁加算'] - base['3桁加算']) + (r['首都'] - base['首都'])  # 負=損
        net = dc + coll
        flag = " ◎" if net > 0 else ""
        print(f"{scale:>6.1f} {r['counting']:>9.3f} {r['3桁加算']:>8.3f} {r['首都']:>7.3f}"
              f" {dc:>+8.3f} {coll:>+9.3f} {net:>+8.3f}{flag}")
    print("\n純益(Δcount+巻添)が正の範囲がPareto的に有用。膝=純益が正→負へ転じる点。")


if __name__ == "__main__":
    main()
