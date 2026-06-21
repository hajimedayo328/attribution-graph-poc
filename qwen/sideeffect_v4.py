# 実験V-4: アブレーション強度のトレードオフ曲線
# V-3で「L24-27 MLP×0.5はcountingを上げるが3桁加算-0.20/首都-0.16を壊す」=トレードオフ判明。
# より弱い介入(L25のみ/×0.7)で副作用を減らしつつcounting改善を保てるか(Pareto点)を探す。
import torch
from transformer_lens import HookedTransformer

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

# (ラベル, 層リスト, スケール)
CONFIGS = [
    ("無介入", [], 1.0),
    ("L25のみ×0.5", [25], 0.5),
    ("L24-27×0.7", [24, 25, 26, 27], 0.7),
    ("L24-27×0.5", [24, 25, 26, 27], 0.5),
]


def make_hook(scale):
    def h(value, hook):
        return value * scale
    return h


def gen_answer(model, prompt, layers, scale, max_new=6):
    toks = model.to_tokens(prompt)
    fwd = [(f"blocks.{L}.hook_mlp_out", make_hook(scale)) for L in layers]
    with torch.no_grad(), model.hooks(fwd_hooks=fwd):
        out = model.generate(toks, max_new_tokens=max_new, do_sample=False, verbose=False)
    return model.to_string(out[0, toks.shape[1]:])


def acc(model, items, layers, scale):
    ok = 0
    for prompt, exp in items:
        if exp in gen_answer(model, prompt, layers, scale):
            ok += 1
    return ok / len(items)


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()
    tasks = {
        "counting↑": [(COUNT_FS.format(w), str(len(w))) for w in WORDS],
        "3桁加算↓": [(ADD3_FS.format(a, b), str(a + b)) for a, b in ADD3],
        "首都↓": [(CAP_FS.format(c), a) for c, a in CAPS.items()],
    }
    print(f"{'介入':<14s} {'counting':>9s} {'3桁加算':>8s} {'首都':>7s}")
    print("-" * 44)
    base = {}
    rows = {}
    for label, layers, scale in CONFIGS:
        r = {t: acc(model, items, layers, scale) for t, items in tasks.items()}
        rows[label] = r
        if label == "無介入":
            base = r
        print(f"{label:<14s} {r['counting↑']:>9.3f} {r['3桁加算↓']:>8.3f} {r['首都↓']:>7.3f}")

    print("\n--- ベースラインからのΔ(counting↑が得・他↓が損) ---")
    print(f"{'介入':<14s} {'Δcounting':>10s} {'Δ3桁加算':>9s} {'Δ首都':>7s} {'純益(count-損)':>14s}")
    for label, layers, scale in CONFIGS:
        if label == "無介入":
            continue
        dc = rows[label]['counting↑'] - base['counting↑']
        da = rows[label]['3桁加算↓'] - base['3桁加算↓']
        dp = rows[label]['首都↓'] - base['首都↓']
        net = dc + da + dp  # 改善 - 副作用合計
        print(f"{label:<14s} {dc:>+10.3f} {da:>+9.3f} {dp:>+7.3f} {net:>+14.3f}")
    print("\n純益が最大の介入がPareto的に最良。負なら副作用が改善を上回る。")


if __name__ == "__main__":
    main()
