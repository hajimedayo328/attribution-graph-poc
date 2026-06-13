# 実験U-2: アブレーションの副作用測定
# 実験Uで「L24-27 MLPを0.5倍にするとcounting正答率0.45→0.57」を得た。
# これが「counting専用に賢くなった」のか「全体崩壊で偶然合った」のかを、
# counting以外の対照タスク(首都想起/対義語/1桁加算)で同じアブレーションをかけて切り分ける。
# まずcountingで+12ptを再現(サニティ)してから対照を測る。
import torch
from transformer_lens import HookedTransformer

ABLATE_LAYERS = [24, 25, 26, 27]
SCALE = 0.5

# ---- counting (実験Uの再現用、culprit_experiment.pyと同一データ) ----
WORDS = """cat dog sun pen egg ice fox cup hat bed box car map net oil
fish milk door rain tree book lamp ring snow wolf bird cake desk fork gold
bread chair table cloud house water apple lemon tiger horse plant stone sugar dream
garden window yellow forest summer winter bottle candle dragon flower guitar jungle
teacher bicycle evening library kitchen morning picture rainbow station thunder
hospital sunshine notebook airplane mountain elephant computer painting sandwich shoulder
breakfast telephone classroom crocodile adventure butterfly chocolate dangerous""".split()
COUNT_FS = "The word 'dog' has 3 letters. The word 'apple' has 5 letters. The word '{}' has "

# ---- 対照タスク1: 首都想起(few-shotのFrance/Japanは除外) ----
CAPITALS = {
    "Italy": "Rome", "Spain": "Madrid", "Russia": "Moscow", "China": "Beijing",
    "Egypt": "Cairo", "Greece": "Athens", "Cuba": "Havana", "Peru": "Lima",
    "Iran": "Tehran", "Iraq": "Baghdad", "Poland": "Warsaw", "Norway": "Oslo",
    "Austria": "Vienna", "Turkey": "Ankara", "Thailand": "Bangkok", "Portugal": "Lisbon",
    "Sweden": "Stockholm", "Kenya": "Nairobi",
}
CAP_FS = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of {} is"

# ---- 対照タスク2: 対義語(hot/big除外) ----
ANTONYMS = {
    "up": "down", "fast": "slow", "happy": "sad", "light": "dark", "high": "low",
    "hard": "soft", "wet": "dry", "full": "empty", "rich": "poor", "young": "old",
    "strong": "weak", "day": "night", "true": "false", "buy": "sell", "push": "pull",
    "left": "right", "open": "close", "good": "bad",
}
ANT_FS = "The opposite of hot is cold. The opposite of big is small. The opposite of {} is"

# ---- 対照タスク3: 1桁加算(結果<=9、例の2+3/6+1は除外) ----
ARITH = [(4, 5), (3, 4), (8, 1), (2, 6), (5, 3), (7, 2), (4, 4), (6, 2),
         (1, 5), (3, 3), (5, 4), (2, 2), (7, 1), (4, 2), (6, 3), (1, 8)]
ARI_FS = "2 + 3 = 5. 6 + 1 = 7. {} + {} ="


def mlp_hook(value, hook):
    return value * SCALE


def predict(model, toks, ablate):
    fwd = [(f"blocks.{L}.hook_mlp_out", mlp_hook) for L in ABLATE_LAYERS] if ablate else []
    with torch.no_grad():
        logits = model.run_with_hooks(toks, fwd_hooks=fwd)
    return logits[0, -1].argmax().item()


def cand_ids(model, answer, with_space=True):
    """正解トークンid集合。with_space=Trueなら'X'と' X'両方を許容(継続形の揺れ対策)。"""
    ids = set()
    forms = [answer, " " + answer] if with_space else [answer]
    for s in forms:
        t = model.to_tokens(s, prepend_bos=False)
        if t.shape[1] > 0:
            ids.add(t[0, 0].item())
    return ids


def eval_task(model, items, with_space=True):
    """items: [(prompt, answer)]。baseline/ablatedの正答率を返す。"""
    base_ok = abl_ok = 0
    flips_down = []  # baselineで正解→ablatedで不正解になった例
    for prompt, ans in items:
        toks = model.to_tokens(prompt)
        cands = cand_ids(model, ans, with_space)
        b = predict(model, toks, False) in cands
        a = predict(model, toks, True) in cands
        base_ok += b
        abl_ok += a
        if b and not a:
            flips_down.append(prompt.split(".")[-1].strip()[:40])
    n = len(items)
    return base_ok / n, abl_ok / n, n, flips_down


def main() -> None:
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    model.eval()
    print(f"device={model.cfg.device}, n_layers={model.cfg.n_layers}, ablate=L{ABLATE_LAYERS} MLP×{SCALE}\n")

    tasks = []
    # counting: 正解は数字(空白なし)で原実験に合わせる
    count_items = [(COUNT_FS.format(w), str(len(w))) for w in WORDS]
    tasks.append(("counting(再現)", count_items, False))
    tasks.append(("首都想起", [(CAP_FS.format(c), a) for c, a in CAPITALS.items()], True))
    tasks.append(("対義語", [(ANT_FS.format(w), a) for w, a in ANTONYMS.items()], True))
    tasks.append(("1桁加算", [(ARI_FS.format(x, y), str(x + y)) for x, y in ARITH], True))

    print(f"{'タスク':<14s} {'n':>3s} {'baseline':>9s} {'ablated':>8s} {'Δ':>7s}")
    print("-" * 48)
    results = {}
    for name, items, ws in tasks:
        b, a, n, flips = eval_task(model, items, with_space=ws)
        results[name] = (b, a, n, flips)
        print(f"{name:<14s} {n:>3d} {b:>9.3f} {a:>8.3f} {a-b:>+7.3f}")

    print("\n--- 解釈 ---")
    cb, ca, _, _ = results["counting(再現)"]
    print(f"counting再現: {cb:.3f}->{ca:.3f} ({ca-cb:+.3f})  "
          f"{'[OK] 実験Uの+0.12を再現' if ca - cb >= 0.08 else '[NG] 再現せず(プロンプト/環境差を疑う)'}")
    control_deltas = [results[k][1] - results[k][0] for k in ("首都想起", "対義語", "1桁加算")]
    worst = min(control_deltas)
    print(f"対照タスクの正答率変化: 首都{control_deltas[0]:+.3f} / 対義語{control_deltas[1]:+.3f} / 算術{control_deltas[2]:+.3f}")
    if worst >= -0.05:
        print("→ 対照タスクはほぼ無傷。「counting専用に賢くなった」を支持(局所的な改善)")
    elif worst >= -0.15:
        print("→ 対照タスクに軽度の劣化。改善はcounting寄りだが完全な無副作用ではない")
    else:
        print("→ 対照タスクが大きく劣化。「全体が壊れてcountingは偶然合っただけ」の疑いが濃い")
    for k in ("首都想起", "対義語", "1桁加算"):
        f = results[k][3]
        if f:
            print(f"  {k}でablation後に崩れた例: {f}")


if __name__ == "__main__":
    main()
