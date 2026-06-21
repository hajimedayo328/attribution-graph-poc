# 実験V-3: 真の中域baselineで副作用測定 + 首都劣化の確認
# V-2でQwen0.6Bが2桁算術を満点→天井。3桁加算・2桁×1桁乗算で中域(0.3-0.7)を狙う。
# V-2で見えた「中堅国の首都 -0.167」が本物かnを増やして確認。
import torch
from transformer_lens import HookedTransformer

ABLATE_LAYERS = [24, 25, 26, 27]
SCALE = 0.5

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

MUL21 = [(23,7),(48,6),(37,9),(56,4),(29,8),(64,3),(45,7),(38,6),(72,5),(27,9),
         (53,8),(46,7),(39,4),(68,3),(57,6),(34,8),(49,7),(63,5),(28,9),(76,4)]
MUL21_FS = "23 * 7 = 161. 48 * 6 = 288. {} * {} ="

ADD2 = [(23,45),(17,38),(54,29),(46,37),(62,19),(28,55),(34,48),(71,16)]  # 天井参照
ADD2_FS = "23 + 45 = 68. 17 + 31 = 48. {} + {} ="

CAPS = {"Canada":"Ottawa","Australia":"Canberra","Brazil":"Bras","Turkey":"Ankara",
        "Switzerland":"Bern","Norway":"Oslo","Portugal":"Lisbon","Greece":"Athens",
        "Poland":"Warsaw","Sweden":"Stockholm","Austria":"Vienna","Ireland":"Dublin",
        "Finland":"Helsinki","Denmark":"Copenhagen","Hungary":"Budapest","Romania":"Bucharest",
        "Vietnam":"Hanoi","Nigeria":"Abuja","Argentina":"Buenos","Chile":"Santiago",
        "Peru":"Lima","Colombia":"Bogot","Morocco":"Rabat","Kenya":"Nairobi",
        "Iran":"Tehran","Iraq":"Baghdad","Pakistan":"Islamabad","Indonesia":"Jakarta",
        "Philippines":"Manila","Ukraine":"Kyiv","Netherlands":"Amsterdam","Belgium":"Brussels"}
CAP_FS = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of {} is"


def mlp_hook(value, hook):
    return value * SCALE


def gen_answer(model, prompt, ablate, max_new=6):
    toks = model.to_tokens(prompt)
    fwd = [(f"blocks.{L}.hook_mlp_out", mlp_hook) for L in ABLATE_LAYERS] if ablate else []
    with torch.no_grad(), model.hooks(fwd_hooks=fwd):
        out = model.generate(toks, max_new_tokens=max_new, do_sample=False, verbose=False)
    return model.to_string(out[0, toks.shape[1]:])


def eval_task(model, items):
    base_ok = abl_ok = 0
    drops = []
    for prompt, exp in items:
        b = exp in gen_answer(model, prompt, False)
        a = exp in gen_answer(model, prompt, True)
        base_ok += b
        abl_ok += a
        if b and not a:
            drops.append(prompt.split(".")[-1].strip()[:30])
    n = len(items)
    return base_ok / n, abl_ok / n, n, drops


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()
    print(f"device={model.cfg.device}, ablate=L{ABLATE_LAYERS} MLP×{SCALE}\n")

    tasks = [
        ("counting(正対照)", [(COUNT_FS.format(w), str(len(w))) for w in WORDS]),
        ("2桁加算(天井参照)", [(ADD2_FS.format(a, b), str(a + b)) for a, b in ADD2]),
        ("3桁加算", [(ADD3_FS.format(a, b), str(a + b)) for a, b in ADD3]),
        ("2桁×1桁乗算", [(MUL21_FS.format(a, b), str(a * b)) for a, b in MUL21]),
        ("中堅国の首都", [(CAP_FS.format(c), a) for c, a in CAPS.items()]),
    ]
    print(f"{'タスク':<18s} {'n':>3s} {'baseline':>9s} {'ablated':>8s} {'Δ':>7s} {'中域?':>4s}")
    print("-" * 56)
    res = {}
    for name, items in tasks:
        b, a, n, drops = eval_task(model, items)
        res[name] = (b, a, n, drops)
        mid = "○" if 0.3 <= b <= 0.7 else "×"
        print(f"{name:<18s} {n:>3d} {b:>9.3f} {a:>8.3f} {a-b:>+7.3f} {mid:>4s}")

    print("\n--- 解釈 ---")
    cb, ca, _, _ = res["counting(正対照)"]
    print(f"counting: {cb:.3f}→{ca:.3f} ({ca-cb:+.3f}) {'[OK]改善再現' if ca-cb>=0.05 else '[NG]'}")
    for k, (b, a, n, drops) in res.items():
        if k == "counting(正対照)":
            continue
        tag = "中域" if 0.3 <= b <= 0.7 else "天井/床"
        verdict = "無傷〜改善" if a-b >= -0.05 else ("軽度劣化" if a-b >= -0.15 else "大きく劣化")
        line = f"  {k}({tag}): {b:.3f}→{a:.3f} ({a-b:+.3f}) → {verdict}"
        if drops:
            line += f" | 崩れ:{drops[:4]}"
        print(line)


if __name__ == "__main__":
    main()
