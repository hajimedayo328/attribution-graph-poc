# 実験T-2: logit lensの普遍性 — 「正解が浮上→消失」は別モデルでも出るか
# 実験T(Qwen3-0.6B)で見たcountingの「中間層で正解top-1→最終層で消失(61%)」が、
# Qwen3-1.7Bでも再現するかを検証。RAM安全: 基底モデルのみ(transcoder不要)、1.7B bf16≈3.4GB。
import sys

import numpy as np
import torch
from transformer_lens import HookedTransformer

from logitlens_experiment import WORDS, FS_COUNT, FACTS, FS_YEAR, lens_run
from ram_guard import start_watchdog, assert_safe_to_load

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"


def run_task(model, items, n_layers):
    correct = errors = lost = 0
    emerg, lost_ranks = [], []
    for prompt, target in items:
        ranks, is_top, final_correct = lens_run(model, prompt, target)
        if final_correct:
            correct += 1
            em = next((layer for layer, t in enumerate(is_top) if t), None)
            if em is not None:
                emerg.append(em)
        else:
            errors += 1
            if any(is_top):  # 中間で正解top-1だったのに最終層で失った
                lost += 1
                lost_ranks.append(ranks)
    return correct, errors, lost, emerg, lost_ranks


def main() -> None:
    print(f"=== {MODEL} (logit lens) ===")
    start_watchdog(80)              # 80%超で自己終了(フリーズ前に止める物理ガード)
    assert_safe_to_load(MODEL)     # 推定peak > 空きRAM なら中止(gemma F32等を弾く)
    model = HookedTransformer.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    nl = model.cfg.n_layers
    print(f"n_layers={nl}, device={model.cfg.device}\n")

    for task, items in [("counting", [(FS_COUNT.format(w), str(len(w))) for w in WORDS]),
                        ("years", [(FS_YEAR.format(s), str(y)) for s, y in FACTS])]:
        c, e, lost, emerg, lost_ranks = run_task(model, items, nl)
        acc = c / len(items)
        print(f"[{task}] n={len(items)} 正答率={acc:.2f} 誤答={e}")
        if e:
            print(f"  「中間で正解top-1→最終層で失った」: {lost}件 (誤答の{lost/e*100:.0f}%)")
        if emerg:
            print(f"  正答ケースの浮上層: 中央値 L{int(np.median(emerg))} (範囲 L{min(emerg)}-L{max(emerg)})")
        if lost_ranks:
            lr = np.median(np.array(lost_ranks), axis=0)
            top1_layer = next((layer for layer in range(nl) if lr[layer] == 0), None)
            print(f"  lost中央値プロファイル: L0={int(lr[0])}位 → "
                  f"最小到達={'L%d で1位' % top1_layer if top1_layer is not None else 'top-1未到達'} "
                  f"→ L{nl-1}={int(lr[nl-1])}位")
        print()

    print("--- 0.6Bとの比較 ---")
    print("Qwen3-0.6B: counting誤答の61%(28/46)で浮上→消失、年は0件、正答浮上中央値L23")
    print("→ 上の1.7B結果と比べ、浮上→消失パターンがモデルを跨ぐか判断")


if __name__ == "__main__":
    main()
