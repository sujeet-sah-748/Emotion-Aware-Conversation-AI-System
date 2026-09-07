"""
test_emotion_consistency.py
===========================

Conversational Emotion Consistency / Trajectory Evaluation

Purpose
-------
Evaluates whether the chatbot maintains a semantically coherent emotional
trajectory across multiple conversational turns.

This is different from ordinary unit testing:
    - Unit tests: "Does this function work?"
    - Adversarial tests: "Does the system resist known failure cases?"
    - This test: "Does the affective state evolve coherently over time?"

The test is designed for research evaluation of a conversational
emotion-aware chatbot.

Requirements
------------
    pip install requests

Run:
    python test_emotion_consistency.py

Requires:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import requests
import time
from dataclasses import dataclass
from typing import Any, Dict, List


# ============================================================
# Configuration
# ============================================================

DEFAULT_API = "http://localhost:8000"

TIMEOUT = 120


# ============================================================
# Data structures
# ============================================================

@dataclass
class TurnResult:
    turn: int
    text: str
    status: int
    dominant: str
    valence: float
    arousal: float
    confidence: float
    latency_ms: float = 0.0


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    turns: List[TurnResult]
    reason: str


# ============================================================
# Helpers
# ============================================================

def normalize(label: str) -> str:
    return label.lower().strip().replace("-", "_").replace(" ", "_")


def contains_any(label: str, candidates: List[str]) -> bool:
    label = normalize(label)

    return any(
        candidate in label or label in candidate
        for candidate in map(normalize, candidates)
    )


def get_affect(response: requests.Response) -> Dict[str, Any]:
    data = response.json()
    return data.get("affect_state", {})


def get_vad(affect: Dict[str, Any], key: str) -> Dict[str, float]:
    vad = affect.get(key, {})

    return {
        "valence": float(vad.get("valence", 0.0)),
        "arousal": float(vad.get("arousal", 0.0)),
        "dominance": float(vad.get("dominance", 0.0)),
    }


def chat(
    base_url: str,
    text: str,
    user_id: str,
    threshold: float = 0.3,
) -> requests.Response:

    return requests.post(
        f"{base_url}/chat",
        json={
            "text": text,
            "user_id": user_id,
            "threshold": threshold,
        },
        timeout=TIMEOUT,
    )


# ============================================================
# Run one conversational scenario
# ============================================================

def run_scenario(
    base_url: str,
    name: str,
    user_id: str,
    conversation: List[str],
) -> ScenarioResult:

    results: List[TurnResult] = []

    for index, message in enumerate(conversation, start=1):

        try:
            _t0 = time.perf_counter()
            response = chat(
                base_url=base_url,
                text=message,
                user_id=user_id,
            )
            _latency_ms = (time.perf_counter() - _t0) * 1000.0

            if response.status_code != 200:
                return ScenarioResult(
                    name=name,
                    passed=False,
                    turns=results,
                    reason=(
                        f"Turn {index} returned HTTP "
                        f"{response.status_code}"
                    ),
                )

            data = response.json()
            affect = data.get("affect_state", {})

            dominant = affect.get("stm_dominant", "")

            vad = get_vad(
                affect,
                "situational_vad",
            )

            confidence = float(
                affect.get("confidence", 0.0)
            )

            results.append(
                TurnResult(
                    turn=index,
                    text=message,
                    status=response.status_code,
                    dominant=dominant,
                    valence=vad["valence"],
                    arousal=vad["arousal"],
                    confidence=confidence,
                    latency_ms=_latency_ms,
                )
            )

        except Exception as exc:

            return ScenarioResult(
                name=name,
                passed=False,
                turns=results,
                reason=f"Turn {index} raised exception: {exc}",
            )

    return ScenarioResult(
        name=name,
        passed=True,
        turns=results,
        reason="All turns completed successfully.",
    )


# ============================================================
# Scenario 1
# Negative → recovery → positive
# ============================================================

def test_recovery_trajectory(base_url: str) -> ScenarioResult:

    conversation = [
        "I got some really bad news today.",
        "Honestly, I feel terrible and disappointed.",
        "I talked to my friend about it and I feel a little better.",
        "I'm starting to feel hopeful again.",
        "Actually, I think things might work out after all.",
    ]

    result = run_scenario(
        base_url,
        "Negative → Recovery → Positive",
        "trajectory-recovery",
        conversation,
    )

    if not result.passed:
        return result

    # We expect the emotional trajectory to become
    # more positive over time.

    first = result.turns[0]
    last = result.turns[-1]

    if last.valence < first.valence:
        result.passed = False
        result.reason = (
            "Expected valence to improve during recovery, "
            f"but changed from {first.valence:.3f} "
            f"to {last.valence:.3f}."
        )

    return result


# ============================================================
# Scenario 2
# Positive → disappointment
# ============================================================

def test_positive_to_negative(base_url: str) -> ScenarioResult:

    conversation = [
        "I am extremely excited about today!",
        "Everything seemed to be going perfectly.",
        "Then my manager cancelled the opportunity.",
        "I'm honestly disappointed and frustrated.",
        "Now I feel pretty discouraged.",
    ]

    result = run_scenario(
        base_url,
        "Positive → Disappointment",
        "trajectory-negative",
        conversation,
    )

    if not result.passed:
        return result

    # Compare peak valence in the positive phase (turns 1-2)
    # against the trough valence in the negative phase (turns 3-5).
    # This is more robust than first-vs-last because the opening
    # sentence sometimes lands with slightly negative initial VAD.
    positive_phase = result.turns[:2]
    negative_phase = result.turns[2:]

    peak_valence = max(t.valence for t in positive_phase)
    trough_valence = min(t.valence for t in negative_phase)

    if trough_valence >= peak_valence:
        result.passed = False
        result.reason = (
            "Expected valence to drop after the disappointment event, "
            f"but peak (turns 1-2) was {peak_valence:.3f} "
            f"and trough (turns 3-5) was {trough_valence:.3f}."
        )
    else:
        result.reason = (
            f"Valence dropped from peak {peak_valence:.3f} "
            f"(turns 1-2) to trough {trough_valence:.3f} (turns 3-5)."
        )

    return result


# ============================================================
# Scenario 3
# Stable neutral conversation
# ============================================================

def test_neutral_stability(base_url: str) -> ScenarioResult:

    conversation = [
        "I need to buy some groceries today.",
        "I also need to pick up some notebooks.",
        "Then I'll probably go home and study.",
        "I have a lot of work to finish tonight.",
    ]

    result = run_scenario(
        base_url,
        "Neutral Stability",
        "trajectory-neutral",
        conversation,
    )

    if not result.passed:
        return result

    # Very large emotional swings in a neutral conversation
    # are suspicious.

    vals = [turn.valence for turn in result.turns]

    if max(vals) - min(vals) > 1.5:
        result.passed = False
        result.reason = (
            "Large unexplained valence swing detected: "
            f"{max(vals) - min(vals):.3f}"
        )

    return result


# ============================================================
# Scenario 4
# Anger → de-escalation
# ============================================================

def test_anger_deescalation(base_url: str) -> ScenarioResult:

    conversation = [
        "My coworker blamed me for something I didn't do.",
        "I'm furious about it.",
        "I wanted to scream at him.",
        "But I took some time to calm down.",
        "Talking about it actually helped me feel calmer.",
    ]

    result = run_scenario(
        base_url,
        "Anger → De-escalation",
        "trajectory-anger",
        conversation,
    )

    if not result.passed:
        return result

    # Anger conversations naturally escalate before de-escalating.
    # The meaningful signal is: peak arousal > final arousal.
    # Comparing first vs last is misleading because the first turn
    # ("My coworker blamed me...") hasn't yet reached peak anger.
    peak_arousal = max(t.arousal for t in result.turns)
    final_arousal = result.turns[-1].arousal

    if final_arousal >= peak_arousal:
        result.passed = False
        result.reason = (
            "Expected arousal to decrease from its peak during "
            f"de-escalation, but peak was {peak_arousal:.3f} "
            f"and final was {final_arousal:.3f}."
        )
    else:
        result.reason = (
            f"Arousal de-escalated from peak {peak_arousal:.3f} "
            f"to {final_arousal:.3f} after calming."
        )

    return result


# ============================================================
# Scenario 5
# Fear → relief
# ============================================================

def test_fear_to_relief(base_url: str) -> ScenarioResult:

    conversation = [
        "I'm really scared about the results.",
        "I've been worrying about this all morning.",
        "They finally called me.",
        "The results were much better than I expected.",
        "I feel so relieved now.",
    ]

    result = run_scenario(
        base_url,
        "Fear → Relief",
        "trajectory-relief",
        conversation,
    )

    if not result.passed:
        return result

    first = result.turns[0]
    last = result.turns[-1]

    if last.valence < first.valence:
        result.passed = False
        result.reason = (
            "Expected valence to increase after relief, "
            f"but changed from {first.valence:.3f} "
            f"to {last.valence:.3f}."
        )

    return result


# ============================================================
# Output
# ============================================================

def print_result(result: ScenarioResult):

    status = "PASS" if result.passed else "FAIL"

    print("\n" + "=" * 75)
    print(f"[{status}] {result.name}")
    print("=" * 75)

    for turn in result.turns:

        print(
            f"Turn {turn.turn}: "
            f"{turn.dominant:<18} "
            f"V={turn.valence:+.3f} "
            f"A={turn.arousal:+.3f} "
            f"C={turn.confidence:.3f} "
            f"[{turn.latency_ms:.0f}ms]"
        )

        print(f"  {turn.text}")

    print(f"\nReason: {result.reason}")


# ============================================================
# Latency summary
# ============================================================

def print_latency_summary(results: List[ScenarioResult]) -> None:

    all_latencies: List[float] = []

    print("\n" + "-" * 75)
    print("LATENCY SUMMARY (per-turn API response times)")
    print("-" * 75)

    for result in results:
        if not result.turns:
            continue

        lats = [t.latency_ms for t in result.turns]
        all_latencies.extend(lats)

        avg = sum(lats) / len(lats)
        p95_idx = max(0, int(len(lats) * 0.95) - 1)
        p95 = sorted(lats)[p95_idx]
        maximum = max(lats)

        print(
            f"  {result.name:<40} "
            f"avg={avg:>7.0f}ms  "
            f"p95={p95:>7.0f}ms  "
            f"max={maximum:>7.0f}ms"
        )

    if all_latencies:
        avg_all = sum(all_latencies) / len(all_latencies)
        p95_idx = max(0, int(len(all_latencies) * 0.95) - 1)
        p95_all = sorted(all_latencies)[p95_idx]
        max_all = max(all_latencies)

        print(
            f"\n  {'GLOBAL':<40} "
            f"avg={avg_all:>7.0f}ms  "
            f"p95={p95_all:>7.0f}ms  "
            f"max={max_all:>7.0f}ms"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Conversational emotional trajectory "
            "evaluation for the emotion chatbot."
        )
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_API,
        help="Chatbot API base URL.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Server check
    # --------------------------------------------------------

    try:
        response = requests.get(
            f"{args.api_url}/",
            timeout=5,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Server returned HTTP {response.status_code}"
            )

    except Exception as exc:

        print(
            "\nERROR: Could not connect to chatbot API."
        )
        print(f"Details: {exc}")
        print(
            "\nStart the server first, for example:"
        )
        print(
            "uvicorn main:app --host 0.0.0.0 --port 8000"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Run scenarios
    # --------------------------------------------------------

    tests = [
        test_recovery_trajectory,
        test_positive_to_negative,
        test_neutral_stability,
        test_anger_deescalation,
        test_fear_to_relief,
    ]

    results: List[ScenarioResult] = []

    for test in tests:

        result = test(args.api_url)

        results.append(result)

        print_result(result)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    passed = sum(
        1 for result in results
        if result.passed
    )

    total = len(results)

    print("\n" + "#" * 75)
    print("CONVERSATIONAL EMOTION CONSISTENCY SUMMARY")
    print("#" * 75)

    print(
        f"Scenarios passed: {passed}/{total} "
        f"({100 * passed / total:.1f}%)"
    )

    for result in results:

        print(
            f"  [{'PASS' if result.passed else 'FAIL'}] "
            f"{result.name}"
        )

    print_latency_summary(results)

    print()

    raise SystemExit(
        0 if passed == total else 1
    )


if __name__ == "__main__":
    main()