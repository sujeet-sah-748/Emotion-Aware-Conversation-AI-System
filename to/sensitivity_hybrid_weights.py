"""
Sensitivity Analysis C -- Hybrid score weights (w_s/w_r/w_i) vs. top-1
retrieval ranking stability.

Quantifies whether the default 0.7/0.2/0.1 split is a fragile choice
(small perturbations flip the top result) or a robust region (nearby
weightings agree on what's retrieved). This is evidence about the
STABILITY of the choice, not evidence that 0.7/0.2/0.1 is optimal --
no ground-truth relevance judgments exist to optimize against.

MULTI-SEED: memory store contents and queries are regenerated from
N_SEEDS independent random streams; mean +/- std and a normal-
approximation 95% CI are reported for top-1 agreement rather than a
single-store point estimate.
"""
import json
import math
import random
import statistics
import sys
sys.path.insert(0, ".")
from context_engine import MemoryRetriever, RetrievalConfig, HashingEmbedder

N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
WORDS = ["deadline", "client", "invoice", "meeting", "server", "budget",
         "release", "bug", "database", "vacation", "onboarding", "contract"]

WEIGHT_CONFIGS = [
    (0.70, 0.20, 0.10),  # default
    (0.60, 0.20, 0.20),
    (0.80, 0.10, 0.10),
    (0.70, 0.10, 0.20),
    (0.50, 0.30, 0.20),
    (0.90, 0.05, 0.05),
]


def random_sentence(rng, n=15):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def top1_ids_for_weights(base_retriever, embedder, queries, ws, wr, wi):
    cfg = RetrievalConfig(top_k=5, weight_similarity=ws, weight_recency=wr, weight_importance=wi)
    retriever = MemoryRetriever(embedder=embedder, store=base_retriever._store, config=cfg)
    retriever._memories = base_retriever._memories
    out = {}
    for q in queries:
        results = retriever.retrieve(q, top_k=1)
        out[q] = results[0].memory.memory_id if results else None
    return out


def run_single_seed(seed: int):
    rng = random.Random(seed)
    embedder = HashingEmbedder(dimension=256)
    base_retriever = MemoryRetriever(embedder=embedder, config=RetrievalConfig(top_k=5))
    for _ in range(200):
        content = random_sentence(rng, 20)
        importance = rng.random()
        base_retriever.add(content, importance=importance)

    queries = [random_sentence(rng, 6) for _ in range(15)]
    default_top1 = top1_ids_for_weights(base_retriever, embedder, queries, *WEIGHT_CONFIGS[0])

    agreements = {}
    for ws, wr, wi in WEIGHT_CONFIGS:
        top1 = top1_ids_for_weights(base_retriever, embedder, queries, ws, wr, wi)
        agreement = sum(1 for q in queries if top1[q] == default_top1[q]) / len(queries)
        agreements[f"{ws}/{wr}/{wi}"] = agreement
    return agreements


def mean_ci95(values):
    n = len(values)
    m = statistics.mean(values)
    s = statistics.stdev(values) if n > 1 else 0.0
    half_width = 1.96 * s / math.sqrt(n) if n > 1 else 0.0
    return m, s, (m - half_width, m + half_width)


def run():
    per_config_values = {f"{ws}/{wr}/{wi}": [] for ws, wr, wi in WEIGHT_CONFIGS}
    for seed in SEEDS:
        agreements = run_single_seed(seed)
        for key, val in agreements.items():
            per_config_values[key].append(val)

    rows = []
    for ws, wr, wi in WEIGHT_CONFIGS:
        key = f"{ws}/{wr}/{wi}"
        vals = per_config_values[key]
        m, s, ci = mean_ci95(vals)
        rows.append({
            "weights_sim_rec_imp": key,
            "n_seeds": N_SEEDS,
            "top1_agreement_mean": round(m, 4),
            "top1_agreement_std": round(s, 4),
            "top1_agreement_95ci": [round(ci[0], 4), round(ci[1], 4)],
        })

    print(json.dumps(rows, indent=2))
    with open("sensitivity_hybrid_weights.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    run()

