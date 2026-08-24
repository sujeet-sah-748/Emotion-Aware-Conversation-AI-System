"""
Turns raw emotion classification output into a supportive text response.

Design principle: VALIDATE first, BALANCE second. Never skip straight to
positivity -- that reads as dismissive. If someone is scared or sad, the
response should sit with that first, and only then (optionally) point toward
a broader perspective, without erasing what they said.

This is NOT a therapy engine and does not attempt to diagnose, counsel, or
give mental-health advice. It generates supportive conversational framing
around an emotion classification. The crisis-detection path below is a
deliberately blunt keyword safety net, not a substitute for a real
moderation/safety pipeline -- treat it as a starting point, not a solution.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# =============================================================================
# EMOTION POLARITY GROUPING
# (based on the grouping used in the GoEmotions paper's taxonomy)
# =============================================================================
NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"
}
POSITIVE_EMOTIONS = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief"
}
AMBIGUOUS_EMOTIONS = {
    "confusion", "curiosity", "realization", "surprise"
}
NEUTRAL_EMOTIONS = {"neutral"}


def polarity_of(label: str) -> str:
    if label in NEGATIVE_EMOTIONS:
        return "negative"
    if label in POSITIVE_EMOTIONS:
        return "positive"
    if label in AMBIGUOUS_EMOTIONS:
        return "ambiguous"
    return "neutral"


# =============================================================================
# CRISIS SAFETY NET (hard override -- bypasses normal response composition)
# =============================================================================
# Deliberately blunt keyword matching. This will have false negatives (misses
# real crisis language it doesn't recognize) and false positives (flags safe
# messages). Treat it as a coarse first filter, not a clinical tool -- pair it
# with a real moderation service before relying on it in production.
_CRISIS_PATTERNS = [
    r"\bkill myself\b", r"\bsuicid", r"\bend my life\b", r"\bwant to die\b",
    r"\bno reason to live\b", r"\bself[\s-]?harm\b", r"\bhurt myself\b",
]
_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)

# Replace with resources appropriate to your users' locale(s) before shipping.
CRISIS_RESPONSE_TEXT = (
    "It sounds like you're going through something really heavy right now. "
    "I want to make sure you get support beyond what I can offer here -- "
    "please consider reaching out to a crisis line or someone you trust. "
    "If you're in the US, you can call or text 988 (Suicide & Crisis Lifeline), "
    "available 24/7."
)


def is_crisis_text(text: str) -> bool:
    return bool(_CRISIS_RE.search(text))


# =============================================================================
# VALIDATION / BALANCE TEMPLATES
# =============================================================================
# Each negative emotion gets a validating phrase (said FIRST, standalone) and
# a bridging phrase (said only if there's something genuine to bridge to --
# either a co-occurring positive/ambiguous emotion, or a general invitation
# to keep talking, never a manufactured silver lining).
VALIDATION = {
    "sadness": "That sounds really heavy, and it makes sense that you'd feel that way.",
    "anger": "That frustration sounds valid -- that's a lot to deal with.",
    "fear": "That sounds genuinely scary. It's okay to feel unsettled by that.",
    "nervousness": "It's understandable to feel on edge about that.",
    "grief": "I'm sorry you're carrying that. Loss like that is hard to sit with.",
    "disappointment": "That's a real letdown, and it's fair to feel disappointed.",
    "disapproval": "It makes sense you'd feel uneasy about that.",
    "embarrassment": "That kind of moment can feel a lot bigger than it really is.",
    "remorse": "It sounds like this is weighing on you.",
    "annoyance": "That does sound genuinely irritating.",
    "disgust": "That reaction makes sense given what you're describing.",
}

# Only used when a positive/ambiguous emotion is ALSO present alongside a
# negative one -- reflects what's actually there, doesn't invent anything.
BRIDGE_WITH_COOCCURRING = (
    "At the same time, it sounds like there's some {positive_label} in there too -- "
    "both things can be true at once."
)

# Used only when NO positive/ambiguous signal is present. Deliberately does NOT
# try to redirect to positivity -- just keeps the door open, since manufacturing
# a silver lining the person didn't express is what makes responses feel hollow.
OPEN_ENDED_FOLLOWUP = "Do you want to talk more about what's going on?"


@dataclass
class ComposedResponse:
    text: str
    is_crisis_override: bool = False
    validated_emotions: List[str] = field(default_factory=list)
    bridged_emotion: Optional[str] = None


def compose_response(user_text: str, emotions: List[Dict]) -> ComposedResponse:
    """
    emotions: list of {"label": str, "score": float}, e.g. the `emotions` field
    already returned by predict_emotions() in the classification API.
    """
    # Hard override: crisis content bypasses everything below.
    if is_crisis_text(user_text):
        return ComposedResponse(text=CRISIS_RESPONSE_TEXT, is_crisis_override=True)

    labels = [e["label"] for e in emotions]
    negatives = [l for l in labels if polarity_of(l) == "negative"]
    positives_or_ambiguous = [l for l in labels if polarity_of(l) in ("positive", "ambiguous")]

    if not negatives:
        # No negative emotion detected -- nothing to validate/balance, just
        # acknowledge plainly. Don't force enthusiasm that wasn't asked for.
        if positives_or_ambiguous:
            return ComposedResponse(
                text="Glad to hear that. Tell me more if you'd like.",
                validated_emotions=[],
            )
        return ComposedResponse(text="Thanks for sharing that.", validated_emotions=[])

    # Validate the strongest negative emotion first, in its own right.
    primary_negative = negatives[0]
    validation_line = VALIDATION.get(
        primary_negative, "That sounds like a lot to sit with."
    )

    parts = [validation_line]
    bridged = None

    if positives_or_ambiguous:
        # Only bridge to something the person actually expressed.
        bridged = positives_or_ambiguous[0]
        parts.append(BRIDGE_WITH_COOCCURRING.format(positive_label=bridged))
    else:
        parts.append(OPEN_ENDED_FOLLOWUP)

    return ComposedResponse(
        text=" ".join(parts),
        validated_emotions=negatives,
        bridged_emotion=bridged,
    )