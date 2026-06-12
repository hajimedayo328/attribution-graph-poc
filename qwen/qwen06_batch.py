# Qwen3-0.6B で言い換え実験(F/G/H×4カ国)をローカル生成 — universality本検証
# 安全装置: RAMウォッチドッグ(85%で自己終了)。transcoderはVRAM側(実測10.9GB)。
import gc
import os
import threading
import time
from pathlib import Path

import psutil
import torch
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils.create_graph_files import create_graph_files

OUT = Path(__file__).parent / "graphs_qwen06"
OUT.mkdir(exist_ok=True)

COUNTRIES = ["France", "Japan", "Germany", "Egypt"]
PHRASINGS = {
    "F": "Fact: The capital of {} is",
    "G": "{}'s capital city is",
    "H": "The city that serves as the capital of {} is",
}


def watchdog():
    while True:
        if psutil.virtual_memory().percent > 85:
            print(f"WATCHDOG: RAM {psutil.virtual_memory().percent}% -> abort", flush=True)
            os._exit(1)
        time.sleep(1)


def main() -> None:
    threading.Thread(target=watchdog, daemon=True).start()
    print("loading model...", flush=True)
    model = ReplacementModel.from_pretrained(
        "Qwen/Qwen3-0.6B",
        "mwhanna/qwen3-0.6b-transcoders-lowl0",
        dtype=torch.bfloat16,
        lazy_encoder=False, lazy_decoder=True,
    )
    print(f"model loaded. RAM {psutil.virtual_memory().percent}% / VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    order = [(p, c) for c in ["France", "Japan"] for p in PHRASINGS] + \
            [(p, c) for c in ["Germany", "Egypt"] for p in PHRASINGS]
    for phr, country in order:
        slug = f"qwen06-{phr.lower()}-{country.lower()}"
        if (OUT / f"{slug}.json").exists():
            continue
        prompt = PHRASINGS[phr].format(country)
        print(f"attributing: {slug} {prompt!r}", flush=True)
        try:
            graph = attribute(prompt, model,
                              max_n_logits=10, desired_logit_prob=0.95,
                              batch_size=64, max_feature_nodes=8192, verbose=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("CUDA OOM -> batch_size=16で再試行", flush=True)
            graph = attribute(prompt, model,
                              max_n_logits=10, desired_logit_prob=0.95,
                              batch_size=16, max_feature_nodes=8192, verbose=False, offload="cpu")
        create_graph_files(graph, slug, str(OUT))
        del graph
        gc.collect()
        torch.cuda.empty_cache()
        print(f"done: {slug}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
