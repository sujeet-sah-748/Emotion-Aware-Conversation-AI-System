"""
Sensitivity Analysis B — Confidence gate (gamma_short) vs. admission rate.

Quantifies the practical tradeoff the 0.45 default encodes: how much of a
realistic confidence distribution gets admitted to short-term state at
different gate values. There is no ground truth to optimize against (that
requires human-labeled data per the paper's stated scope limitation), so
this does not claim an "optimal" gamma -- it makes the tradeoff space
explicit and quantified rather than asserted.

MULTI-SEED: run across N_SEEDS independent random streams and report
mean +/- std (and a normal-approximation 95% CI) rather than a single-seed
point estimate, per the paper's own recommendation that reported deltas
should not be treated as more than descriptive without replication.

Confidence distribution: a fixed synthetic mixture meant to resemble a
plausible classifier output spread -- NOT derived from any real model, and
explicitly labeled as such in the write-up.
"""
import json
import math
import random
import statistics
import sys
sys.path.insert(0, ".")
from emotion_engine import EmotionManager, EmotionManagerConfig, EmotionSignal, VAD

N_SEEDS = 20
SEEDS = list(range(N_SEEDS))


class VariableConfidenceClassifier:
    """Emits a fixed direction with confidence drawn from a synthetic
    mixture: 40% low (noise-like, 0.1-0.4), 35% medium (0.4-0.65),
    25% high (0.65-0.95) -- a rough stand-in for "most turns are mildly
    expressive, some are strongly emotional, a minority are noisy."""
    def __init__(self, rng):
        self.vad = VAD(0.7, 0.4, 0.3)
        self.rng = rng

    def _sample_confidence(self):
        r = self.rng.random()
        if r < 0.40:
            return self.rng.uniform(0.10, 0.40)
        elif r < 0.75:
            return self.rng.uniform(0.40, 0.65)
        else:
            return self.rng.uniform(0.65, 0.95)

    def classify(self, turns):
        c = self._sample_confidence()
        return EmotionSignal(vad=self.vad, confidence=c, label_probs={"joy": c})


class Msg:
    def __init__(self, i):
        self.content = f"turn {i}"
        self.message_id = str(i)


def run_single(gate: float, seed: int, n_turns: int = 500) -> float:
    rng = random.Random(seed)
    clf = VariableConfidenceClassifier(rng)
    cfg = EmotionManagerConfig(stm_min_confidence=gate)
    mgr = EmotionManager(clf, config=cfg)
    stm_events = 0
    for i in range(n_turns):
        m = Msg(i)
        events = mgr.process_turn(m, [m] * mgr.context_turns)
        stm_events += sum(1 for e in events if e.tier.value == "short_term")
    return stm_events / n_turns


def mean_ci95(values):
    n = len(values)
    m = statistics.mean(values)
    s = statistics.stdev(values) if n > 1 else 0.0
    half_width = 1.96 * s / math.sqrt(n) if n > 1 else 0.0
    return m, s, (m - half_width, m + half_width)


def run():
    gate_values = [0.30, 0.45, 0.60, 0.75]
    rows = []

    for gate in gate_values:
        rates = [run_single(gate, seed) for seed in SEEDS]
        m, s, ci = mean_ci95(rates)
        rows.append({
            "gamma_short": gate,
            "n_seeds": N_SEEDS,
            "admission_rate_mean": round(m, 4),
            "admission_rate_std": round(s, 4),
            "admission_rate_95ci": [round(ci[0], 4), round(ci[1], 4)],
            "raw_rates": [round(r, 4) for r in rates],
        })

    print(json.dumps([{k: v for k, v in r.items() if k != "raw_rates"} for r in rows], indent=2))
    with open("sensitivity_confidence_gate.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    run()

