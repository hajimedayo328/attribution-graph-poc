# Qwen3-1.7B で言い換え実験(F/G/H×4カ国)のattribution graphをローカル生成
# universality検証: gemma-2-2bで見つけた現象がモデルを跨いで再現するか
import gc
import sys
from pathlib import Path

import torch
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils.create_graph_files import create_graph_files

OUT = Path(__file__).parent / "graphs_qwen"
OUT.mkdir(exist_ok=True)

COUNTRIES = ["France", "Japan", "Germany", "Egypt"]
PHRASINGS = {
    "F": "Fact: The capital of {} is",
    "G": "{}'s capital city is",
    "H": "The city that serves as the capital of {} is",
}

def main() -> None:
    print("loading model...", flush=True)
    model = ReplacementModel.from_pretrained(
        "Qwen/Qwen3-1.7B",
        "mwhanna/qwen3-1.7b-transcoders-lowl0",
        dtype=torch.bfloat16,
    )
    print("model loaded", flush=True)

    # 優先順: France/Japan×F/G/H を先に完結させる(時間切れでも6枚でミニ実験成立)
    order = [(p, c) for c in ["France", "Japan"] for p in PHRASINGS] + \
            [(p, c) for c in ["Germany", "Egypt"] for p in PHRASINGS]
    for phr, country in order:
            tmpl = PHRASINGS[phr]
            slug = f"qwen-{phr.lower()}-{country.lower()}"
            if (OUT / f"{slug}.json").exists():
                continue
            prompt = tmpl.format(country)
            print(f"attributing: {slug} {prompt!r}", flush=True)
            graph = attribute(
                prompt, model,
                max_n_logits=10, desired_logit_prob=0.95,
                batch_size=128, max_feature_nodes=8192, verbose=False,
            )
            create_graph_files(graph, slug, str(OUT))
            del graph
            gc.collect()
            torch.cuda.empty_cache()
            print(f"done: {slug}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
