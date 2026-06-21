# 実験V-2: 天井効果をつぶした副作用測定
# 実験V(対照baseline 0.94-1.0で天井効果)の弱点に答える。baselineが中域(0.3-0.7)に
# 収まる難しめの対照タスク(2桁加減算/1桁乗算/中堅国の首都)で L24-27 MLP×0.5 の副作用を測る。
# 2桁の答えに対応するため貪欲生成+文字列照合。countingは正の対照(上がるはず)。
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

ADD = [(23,45),(17,38),(54,29),(46,37),(62,19),(28,55),(34,48),(71,16),
       (39,44),(57,28),(43,39),(66,27),(48,33),(25,58),(37,46),(59,24)]
ADD_FS = "23 + 45 = 68. 17 + 31 = 48. {} + {} ="

SUB = [(52,17),(63,28),(81,46),(74,39),(95,58),(60,23),(47,19),(88,49),
       (71,35),(56,27),(83,46),(92,57),(64,38),(75,29),(50,16),(67,28)]
SUB_FS = "52 - 17 = 35. 80 - 44 = 36. {} - {} ="

MUL = [(3,4),(6,7),(8,9),(4,5),(7,8),(9,6),(5,5),(8,7),
       (6,4),(9,9),(7,6),(4,8),(3,9),(8,5),(6,6),(7,7)]
MUL_FS = "3 * 4 = 12. 6 * 7 = 42. {} * {} ="

CAPS = {"Canada":"Ottawa","Australia":"Canberra","Brazil":"Bras","Turkey":"Ankara",
        "Switzerland":"Bern","Norway":"Oslo","Portugal":"Lisbon","Greece":"Athens",
        "Poland":"Warsaw","Sweden":"Stockholm","Austria":"Vienna","Ireland":"Dublin",
        "Finland":"Helsinki","Denmark":"Copenhagen","Hungary":"Budapest","Romania":"Bucharest",
        "Vietnam":"Hanoi","Nigeria":"Abuja"}
CAP_FS = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of {} is"


def mlp_hook(value, hook):
    return value * SCALE


def gen_answer(model, prompt, ablate, max_new=5):
    toks = model.to_tokens(prompt)
    fwd = [(f"blocks.{L}.hook_mlp_out", mlp_hook) for L in ABLATE_LAYERS] if ablate else []
    with torch.no_grad(), model.hooks(fwd_hooks=fwd):
        out = model.generate(toks, max_new_tokens=max_new, do_sample=False, verbose=False)
    return model.to_string(out[0, toks.shape[1]:])


def eval_task(model, items):
    """items: [(prompt, expected_str)]。baseline/ablated正答率を返す。"""
    base_ok = abl_ok = 0
    for prompt, exp in items:
        b = exp in gen_answer(model, prompt, False)
        a = exp in gen_answer(model, prompt, True)
        base_ok += b
        abl_ok += a
    n = len(items)
    return base_ok / n, abl_ok / n, n


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()
    print(f"device={model.cfg.device}, ablate=L{ABLATE_LAYERS} MLP×{SCALE}\n")

    tasks = [
        ("counting(正の対照)", [(COUNT_FS.format(w), str(len(w))) for w in WORDS]),
        ("2桁加算", [(ADD_FS.format(a, b), str(a + b)) for a, b in ADD]),
        ("2桁減算", [(SUB_FS.format(a, b), str(a - b)) for a, b in SUB]),
        ("1桁乗算", [(MUL_FS.format(a, b), str(a * b)) for a, b in MUL]),
        ("中堅国の首都", [(CAP_FS.format(c), a) for c, a in CAPS.items()]),
    ]

    print(f"{'タスク':<18s} {'n':>3s} {'baseline':>9s} {'ablated':>8s} {'Δ':>7s} {'中域?':>5s}")
    print("-" * 56)
    res = {}
    for name, items in tasks:
        b, a, n = eval_task(model, items)
        res[name] = (b, a, n)
        mid = "○" if 0.3 <= b <= 0.7 else "×(天井/床)"
        print(f"{name:<18s} {n:>3d} {b:>9.3f} {a:>8.3f} {a-b:>+7.3f} {mid:>5s}")

    print("\n--- 解釈(中域baselineタスクのみが有効な副作用テスト) ---")
    cb, ca, _ = res["counting(正の対照)"]
    print(f"counting(正の対照): {cb:.3f}→{ca:.3f} ({ca-cb:+.3f}) {'[OK]改善再現' if ca-cb>=0.05 else '[NG]'}")
    mids = [(k, v) for k, v in res.items() if k != "counting(正の対照)" and 0.3 <= v[0] <= 0.7]
    if not mids:
        print("有効な中域対照タスクが無かった(全部天井/床)。タスク難易度の再調整が必要")
    else:
        for k, (b, a, n) in mids:
            verdict = "無傷〜改善" if a - b >= -0.05 else ("軽度劣化" if a - b >= -0.15 else "大きく劣化")
            print(f"  {k}: {b:.3f}→{a:.3f} ({a-b:+.3f}) → {verdict}")
        worst = min(a - b for _, (b, a, _) in mids)
        print(f"\n中域対照の最悪Δ={worst:+.3f} → "
              + ("countingの改善は局所的(全体崩壊ではない)を中域でも支持"
                 if worst >= -0.05 else
                 "中域では副作用あり。countingの改善は部分的に全体変質を伴う"))


if __name__ == "__main__":
    main()
