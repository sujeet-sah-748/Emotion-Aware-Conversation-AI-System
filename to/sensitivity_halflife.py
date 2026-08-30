"""
Sensitivity Analysis A — Half-life scaling vs settling dynamics.

Tests whether the system's behavior is smooth and predictable as the three
half-lives are scaled, rather than justifying the specific default values
(300s / 2700s / 259200s) as individually optimal -- no ground truth exists
to optimize against without human-labeled data (see paper Limitations).
What CAN be shown empirically: settling time scales proportionally with
half-life (as exponential decay theory predicts), and the tier ordering
(situational < short-term < long-term) is preserved across a wide range
of multipliers, i.e. the architecture is not fragile to the exact choice.
"""
import json
import sys
sys.path.insert(0, ".")
import emotion_engine as ee
from emotion_engine import EmotionManager, EmotionManagerConfig, EmotionSignal, VAD, project_label

class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.t = start
    def __call__(self):
        return self.t
    def advance(self, s):
        self.t += s

class ConstantClassifier:
    def __init__(self, vad, confidence=0.95):
        self.vad, self.confidence = vad, confidence
    def classify(self, turns):
        return EmotionSignal(vad=self.vad, confidence=self.confidence, label_probs={"joy": self.confidence})

class Msg:
    def __init__(self, i):
        self.content = f"turn {i}"
        self.message_id = str(i)

def settling_turns(config: EmotionManagerConfig, turn_seconds=60.0, max_turns=600):
    clock = FakeClock()
    ee.time.time = clock
    target = VAD(0.85, 0.5, 0.5)
    mgr = EmotionManager(ConstantClassifier(target), config=config)
    target_mag = target.magnitude()
    result = {"situational": None, "short_term": None, "long_term": None}
    for i in range(max_turns):
        m = Msg(i)
        mgr.process_turn(m, [m] * mgr.context_turns)
        state = mgr.affect_state()
        for tier, vad in [("situational", state.situational_vad),
                           ("short_term", state.short_term_vad),
                           ("long_term", state.long_term_vad)]:
            if result[tier] is None and vad.magnitude() / target_mag >= 0.90:
                result[tier] = {"turns": i, "seconds": i * turn_seconds}
        if all(result.values()):
            break
        clock.advance(turn_seconds)
    return result

def run():
    base = EmotionManagerConfig()
    multipliers = [0.5, 1.0, 2.0, 4.0]
    rows = []
    for mult in multipliers:
        cfg = EmotionManagerConfig(
            situational_half_life=base.situational_half_life * mult,
            stm_half_life=base.stm_half_life * mult,
            ltm_half_life=base.ltm_half_life * mult,
        )
        res = settling_turns(cfg)
        row = {"half_life_multiplier": mult}
        for tier in ("situational", "short_term", "long_term"):
            row[f"{tier}_turns_to_90pct"] = res[tier]["turns"] if res[tier] else None
        # ordering check
        vals = [row[f"{t}_turns_to_90pct"] for t in ("situational", "short_term", "long_term")]
        row["ordering_preserved"] = all(v is not None for v in vals) and vals[0] <= vals[1] <= vals[2]
        rows.append(row)
        print(json.dumps(row, indent=2))

    with open("sensitivity_halflife.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows

if __name__ == "__main__":
    run()
