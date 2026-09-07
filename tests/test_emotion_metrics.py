"""
test_emotion_metrics.py
=======================

Research-grade paragraph-level emotion classification evaluation.

Purpose
-------
Evaluates the emotion classifier itself (not the conversational response
generator) using a fixed, balanced, human-readable dataset of 112
multi-sentence paragraphs.

The dataset contains 4 paragraphs for each of the 28 GoEmotions-style
categories. Each paragraph expresses one PRIMARY emotion, even though
natural language may contain secondary emotional cues.

Metrics
-------
- Accuracy
- Macro Precision / Recall / F1
- Weighted Precision / Recall / F1
- Per-emotion Precision / Recall / F1
- Confusion matrix
- Prediction distribution
- Failed examples
- Average / P50 / P95 API latency

Important:
-----------
This evaluates the TOP predicted emotion against the gold primary emotion.
It is therefore a single-label classification benchmark.

For a multi-label evaluation, create a separate dataset with multiple gold
labels per paragraph.

Requirements
------------
    pip install requests scikit-learn

Run
---
    python test_emotion_metrics.py

Optional:
    python test_emotion_metrics.py --api-url http://localhost:8000
    python test_emotion_metrics.py --threshold 0.30
    python test_emotion_metrics.py --verbose

Expected API
------------
POST /predict
JSON:
    {"text": "...", "threshold": 0.30}

Expected response:
    {
        "emotions": [
            {"label": "sadness", "score": 0.81},
            ...
        ]
    }

If your backend uses different label names, edit LABEL_ALIASES below.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_THRESHOLD = 0.30
TIMEOUT = 60


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

# Canonical labels used by this benchmark.
#
# Your previous outputs contained "anticipation", while the standard
# GoEmotions 28-label vocabulary uses "optimism" rather than "anticipation".
# The script therefore lets you explicitly map backend labels if necessary.
LABEL_ALIASES = {
    "happy": "joy",
    "happiness": "joy",
    "sad": "sadness",
    "angry": "anger",
    "mad": "anger",
    "scared": "fear",
    "afraid": "fear",
    "thankful": "gratitude",
    "sorry": "remorse",
    "excited": "excitement",
    "proud": "pride",
    "loving": "love",
    "disappointed": "disappointment",
    "annoyed": "annoyance",
    "confused": "confusion",
    "disgusted": "disgust",
    "embarrassed": "embarrassment",
    "approving": "approval",
    "disapproving": "disapproval",
    "caring": "caring",
    "curious": "curiosity",
    "desiring": "desire",
    "grateful": "gratitude",
    "relieved": "relief",
    "surprised": "surprise",
    "realizing": "realization",
    "amused": "amusement",
    "nervous": "nervousness",
    "optimistic": "optimism",
}

# Standard 28-category vocabulary.
LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    label = label.replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(label, label)


# ---------------------------------------------------------------------------
# Fixed paragraph-level dataset
# ---------------------------------------------------------------------------
#
# 4 paragraphs per category = 112 fixed evaluation samples.
#
# These are intentionally multi-sentence paragraphs rather than isolated
# trigger-word sentences. The gold label represents the PRIMARY emotion.
# ---------------------------------------------------------------------------

DATASET: List[Tuple[str, str]] = [
    # ========================= ADMIRATION =========================
    ("admiration",
     "I watched my mentor present the project to the entire department. "
     "She explained every difficult part with patience and confidence, and "
     "even the toughest questions did not shake her. I left the meeting "
     "thinking about how much I would like to develop that level of skill."),
    ("admiration",
     "The rescue team worked through the night while everyone else was "
     "waiting for updates. Their coordination was calm, disciplined, and "
     "remarkably brave. Seeing how they handled such a difficult situation "
     "made me deeply respect the people doing that work."),
    ("admiration",
     "My professor returned my research draft with detailed comments that "
     "made every section stronger. He had clearly spent a great deal of time "
     "understanding the argument rather than simply marking mistakes. I was "
     "impressed by the care and expertise behind his feedback."),
    ("admiration",
     "She built the entire application almost from scratch after the original "
     "team left the project. The code was organized, the documentation was "
     "clear, and she even prepared examples for future developers. I could "
     "not help but admire how professionally she handled it."),

    # ========================= AMUSEMENT =========================
    ("amusement",
     "My friend tried to explain the rules of the game and somehow made them "
     "more confusing with every sentence. We both ended up laughing at the "
     "ridiculous explanation. Even after the game ended, we kept joking about "
     "what had happened."),
    ("amusement",
     "During the presentation, the microphone stopped working and my "
     "classmate calmly continued speaking as if nothing unusual had happened. "
     "Then he realized everyone was reading his lips and started laughing. "
     "The whole room found the situation funny."),
    ("amusement",
     "My little cousin put on a serious face and announced that he was going "
     "to become the family's official chef. He then burned the toast within "
     "two minutes and proudly served it anyway. Everyone at the table started "
     "laughing."),
    ("amusement",
     "I opened the old group chat and found a message I had completely "
     "forgotten about. It was such an absurd response to a very ordinary "
     "question that I immediately started laughing. Reading the rest of the "
     "conversation only made the memory funnier."),

    # ========================= ANGER =========================
    ("anger",
     "My coworker took credit for a report that I had spent several days "
     "preparing. When the manager praised him for the work, he did not mention "
     "my contribution at all. I was furious because the situation felt "
     "deliberately unfair."),
    ("anger",
     "The company changed the deadline at the last minute even though I had "
     "already completed everything according to the original schedule. When "
     "I asked why, nobody gave me a clear explanation. I was extremely angry "
     "about being treated that way."),
    ("anger",
     "Someone repeatedly interrupted me while I was trying to explain an "
     "important problem. I asked politely several times, but the behavior "
     "continued. Eventually I became so frustrated and angry that I had to "
     "leave the conversation before saying something I would regret."),
    ("anger",
     "I discovered that a teammate had deleted important files without "
     "checking with anyone first. We lost hours of work because of that "
     "decision. I was furious that such a careless action had created a "
     "problem for the entire team."),

    # ========================= ANNOYANCE =========================
    ("annoyance",
     "The construction outside my room started again before sunrise. I had "
     "already been trying to sleep after a long day, and the constant drilling "
     "made it impossible. I was not furious, but I was extremely irritated by "
     "the noise."),
    ("annoyance",
     "My internet connection kept dropping every few minutes while I was "
     "trying to upload a document. Each time the upload almost finished, the "
     "connection failed again. The repeated interruptions were incredibly "
     "irritating."),
    ("annoyance",
     "I had carefully organized my desk before leaving, but someone moved "
     "everything around without asking. Nothing was seriously damaged, yet "
     "finding my things afterward was unnecessarily difficult. The whole "
     "situation was simply annoying."),
    ("annoyance",
     "The meeting could have been finished in ten minutes, but people kept "
     "repeating points that had already been discussed. I kept checking the "
     "clock while the conversation went in circles. By the end, I was mostly "
     "irritated by how much time had been wasted."),

    # ========================= APPROVAL =========================
    ("approval",
     "I read the final version of the proposal and thought the changes were "
     "well considered. The team had responded to the earlier feedback without "
     "losing the main idea. I agreed with the direction and felt that the "
     "proposal was ready to move forward."),
    ("approval",
     "My friend decided to apologize instead of continuing an argument. "
     "Considering everything that had happened, I thought that was a mature "
     "choice. I told her that I supported the decision and believed she had "
     "handled the situation well."),
    ("approval",
     "The manager introduced a new process that gives everyone more time to "
     "review important decisions. I think the change is practical and fair. "
     "I was pleased to see a solution that actually addressed the team's "
     "concerns."),
    ("approval",
     "After reviewing the student's work, I agreed with the approach he had "
     "taken. The reasoning was clear and the final result matched the evidence "
     "he presented. I told him that I thought his solution was a good one."),

    # ========================= CARING =========================
    ("caring",
     "My friend has been struggling lately, so I checked in even though I "
     "knew she might not want to talk. I reminded her that she did not have "
     "to handle everything alone. I mainly wanted her to know that someone "
     "was there for her."),
    ("caring",
     "When my younger brother came home exhausted, I made him something to "
     "eat and asked how his day had gone. He did not say much, but I stayed "
     "nearby rather than pressuring him to explain. I wanted him to feel "
     "supported."),
    ("caring",
     "The new student looked lost during the first week, so I offered to show "
     "him around the campus. I remembered how difficult it was when I first "
     "arrived somewhere unfamiliar. Helping him settle in felt important to "
     "me."),
    ("caring",
     "My colleague seemed unusually quiet after the meeting. I sent a short "
     "message asking whether everything was okay and told her she could talk "
     "if she needed to. I was genuinely concerned about how she was feeling."),

    # ========================= CONFUSION =========================
    ("confusion",
     "I read the instructions three times, but I still could not understand "
     "what the final requirement meant. One paragraph seemed to contradict "
     "another, and I was unsure which version to follow. I kept staring at "
     "the document wondering what I was missing."),
    ("confusion",
     "The conversation suddenly changed direction and I could not figure out "
     "why. A few minutes earlier everyone had agreed on one plan, but then "
     "people started discussing something completely different. I was left "
     "wondering what had actually been decided."),
    ("confusion",
     "The results looked different from what I expected, and I could not "
     "identify the reason immediately. I checked the inputs and repeated the "
     "calculation, but the same result appeared. I was genuinely puzzled by "
     "what was happening."),
    ("confusion",
     "My friend said she was fine, but her tone and expression suggested the "
     "opposite. I did not know whether she wanted me to ask more questions or "
     "give her space. I felt uncertain about how I should respond."),

    # ========================= CURIOSITY =========================
    ("curiosity",
     "I noticed an unfamiliar device on the laboratory table and immediately "
     "wanted to know what it was designed to do. Nobody had explained it yet, "
     "so I started reading the notes beside it. The more I learned, the more "
     "questions I had about how it worked."),
    ("curiosity",
     "A strange pattern appeared in the data that I had not expected to see. "
     "Instead of ignoring it, I wanted to investigate where it came from. "
     "I started comparing the records because I was genuinely interested in "
     "finding an explanation."),
    ("curiosity",
     "My friend mentioned that she had discovered something unusual during "
     "her trip but refused to explain immediately. I kept wondering what had "
     "happened and why she was being so mysterious. I was eager to hear the "
     "full story."),
    ("curiosity",
     "I came across a topic I had never studied before while reading a paper. "
     "The idea was unfamiliar, but it sounded interesting enough that I wanted "
     "to learn more. I opened several references simply because I wanted to "
     "understand it better."),

    # ========================= DESIRE =========================
    ("desire",
     "I have been thinking about moving to a quieter place for months. I want "
     "a home where I can work without constant traffic and noise outside the "
     "window. The idea of having that kind of space is something I really "
     "want."),
    ("desire",
     "I would love to attend the conference next year because the speakers "
     "are working on exactly the topics I care about. I keep imagining what "
     "it would be like to meet researchers who are doing similar work. I "
     "really hope I can get a chance to go."),
    ("desire",
     "After seeing the new laptop, I started imagining how much easier my "
     "work would be with a faster machine. My current computer still works, "
     "but I strongly want the new one. I have been considering ways to save "
     "enough money for it."),
    ("desire",
     "I miss spending time with my old friends and wish we could all meet "
     "again soon. Everyone has become busy with work and study, so it has "
     "been difficult to arrange anything. I really want us to find a day when "
     "we can all be together."),

    # ========================= DISAPPOINTMENT =========================
    ("disappointment",
     "I had spent weeks preparing for the presentation and expected the final "
     "meeting to go well. Instead, the client cancelled at the last minute "
     "without giving us a clear reason. I felt deeply disappointed after all "
     "that preparation."),
    ("disappointment",
     "I thought my friend would remember an important day, but the day passed "
     "without even a message. I know people get busy, yet I had genuinely "
     "expected something different. The situation left me feeling let down."),
    ("disappointment",
     "The final result was much lower than I had expected after all the work "
     "I put into the project. I had imagined seeing a much stronger outcome "
     "and was confident about it beforehand. Seeing the actual result was "
     "discouraging and disappointing."),
    ("disappointment",
     "We were promised that the new service would solve the problems we had "
     "been reporting for months. After trying it, almost nothing had changed. "
     "I had been hopeful about the improvement, so the lack of progress was "
     "particularly disappointing."),

    # ========================= DISAPPROVAL =========================
    ("disapproval",
     "I watched the manager blame an employee publicly for a problem that "
     "could have been discussed privately. Even if the employee had made a "
     "mistake, humiliating someone in front of the team was unnecessary. I "
     "strongly disagreed with the way the situation was handled."),
    ("disapproval",
     "My teammate ignored the agreed process and submitted the work without "
     "checking the final details. We had specifically discussed why those "
     "checks were necessary. I did not think that was a responsible decision."),
    ("disapproval",
     "The advertisement made claims that were clearly more dramatic than the "
     "evidence supported. It seemed designed to mislead people rather than "
     "inform them. I did not approve of that kind of communication."),
    ("disapproval",
     "Someone kept making jokes about another student's mistake even after it "
     "was obvious that the student was uncomfortable. I thought the behavior "
     "was inappropriate and unnecessary. I did not agree with treating "
     "someone that way."),

    # ========================= DISGUST =========================
    ("disgust",
     "I opened the refrigerator and discovered food that had been left there "
     "for far too long. There was a strong smell and a layer of mold across "
     "the container. I immediately felt sick and wanted to get it out of the "
     "room."),
    ("disgust",
     "The public restroom was in terrible condition, with dirty surfaces and "
     "an unpleasant smell everywhere. I tried not to touch anything while "
     "walking through it. The whole place made me feel physically disgusted."),
    ("disgust",
     "I saw someone deliberately throw garbage into a clean stream despite "
     "there being a bin only a few steps away. Watching the waste float away "
     "made me feel sick and disgusted by the careless behavior."),
    ("disgust",
     "The food had clearly gone bad before it was served, and the smell was "
     "impossible to ignore. I took one bite and immediately stopped eating. "
     "I felt nauseated by the thought of having another bite."),

    # ========================= EMBARRASSMENT =========================
    ("embarrassment",
     "I walked into the wrong classroom and confidently sat down before "
     "realizing that nobody recognized me. Everyone looked at me when I "
     "finally noticed the mistake. I quietly apologized and left, feeling "
     "extremely embarrassed."),
    ("embarrassment",
     "During the presentation I accidentally called my professor by the "
     "wrong name. I corrected myself immediately, but the mistake had already "
     "been noticed by everyone in the room. I could feel my face getting hot."),
    ("embarrassment",
     "I sent a private message to the entire group by accident. It was not "
     "anything terrible, but it was obviously meant for one person rather "
     "than twenty people. I wished I could disappear when I realized what I "
     "had done."),
    ("embarrassment",
     "I tripped while walking onto the stage in front of a large audience. "
     "I was not injured, but every person in the room saw it happen. I smiled "
     "and continued, although I felt incredibly self-conscious afterward."),

    # ========================= EXCITEMENT =========================
    ("excitement",
     "I have been waiting for this concert for months, and the day has finally "
     "arrived. I keep checking the time because I cannot wait for the doors "
     "to open. I feel energetic and thrilled just thinking about seeing the "
     "band live."),
    ("excitement",
     "The research team told us that our project had been selected for the "
     "final presentation. I immediately started thinking about what we could "
     "show and how we would present it. I am genuinely thrilled about the "
     "opportunity."),
    ("excitement",
     "My friends surprised me by telling me that we are going on a trip next "
     "weekend. I started planning what to pack almost immediately. The thought "
     "of getting away together has me feeling incredibly excited."),
    ("excitement",
     "The new product is finally being released tomorrow after months of "
     "development. I have followed every stage of the project and cannot wait "
     "to see people's reactions. There is a real sense of anticipation and "
     "energy around the launch."),

    # ========================= FEAR =========================
    ("fear",
     "I was walking home when I realized that someone had been following me "
     "for several blocks. I could not tell who the person was or what they "
     "wanted. My heart started racing, and I felt genuinely afraid."),
    ("fear",
     "The doctor said that more tests were needed before they could explain "
     "the results. I know that additional testing does not automatically "
     "mean something is wrong, but I could not stop imagining the worst. "
     "The uncertainty made me very anxious and scared."),
    ("fear",
     "I heard a loud crash downstairs in the middle of the night. For a moment "
     "I did not know what had caused it or whether someone had entered the "
     "house. I stayed still, listening carefully and feeling frightened."),
    ("fear",
     "The car suddenly lost control on the wet road and began sliding. I "
     "grabbed the seat and waited for the driver to regain control. Those "
     "few seconds were terrifying, and I was afraid something serious would "
     "happen."),

    # ========================= GRATITUDE =========================
    ("gratitude",
     "My friend stayed with me when I was struggling to finish the project. "
     "She could have gone home, but she stayed late and helped me work through "
     "the difficult parts. I am genuinely grateful that she gave me her time."),
    ("gratitude",
     "The doctor explained everything patiently instead of rushing through "
     "the appointment. I left feeling much more informed and supported than "
     "when I arrived. I was very thankful for the time and attention they gave "
     "me."),
    ("gratitude",
     "My parents supported my decision even though it was different from what "
     "they had originally expected. They listened without judging me and "
     "encouraged me to keep going. I feel deeply thankful for that support."),
    ("gratitude",
     "A colleague noticed that I was struggling and quietly helped me finish "
     "the task before the deadline. I had not asked for help, so the gesture "
     "meant even more to me. I really appreciate what they did."),

    # ========================= GRIEF =========================
    ("grief",
     "My grandfather passed away last week, and the house feels strangely "
     "quiet without him. I keep remembering small conversations we used to "
     "have and then realizing that I will never have another one. The loss "
     "still feels difficult to accept."),
    ("grief",
     "I found an old photograph of my dog who died several years ago. For a "
     "moment it felt as though I was back in those days with him. The memory "
     "was comforting, but it also brought back a deep sense of loss."),
    ("grief",
     "After my friend moved away permanently, I realized how much of my daily "
     "routine had involved seeing her. There are little moments when I still "
     "expect her to be there. I miss her deeply and feel the absence every "
     "day."),
    ("grief",
     "The family gathered after the funeral, but nobody seemed to know what "
     "to say. I kept thinking about the person we had lost and how different "
     "everything would feel now. There was a heavy sadness in the room that "
     "I could not shake."),

    # ========================= JOY =========================
    ("joy",
     "I opened my email this morning and saw that the project had finally "
     "been approved. After months of revisions, seeing that message made me "
     "smile immediately. I felt genuinely happy and wanted to share the news "
     "with everyone who had helped."),
    ("joy",
     "My family surprised me by visiting for the weekend. I had not expected "
     "to see them, so the moment I opened the door I felt a huge wave of "
     "happiness. We spent the evening talking, laughing, and enjoying being "
     "together."),
    ("joy",
     "The results came back and everything was better than I had hoped. I had "
     "spent weeks worrying about the outcome, so the good news felt wonderful. "
     "I could not stop smiling for the rest of the day."),
    ("joy",
     "I finally finished a difficult project that had occupied most of my "
     "free time. Looking at the completed work gave me a strong sense of "
     "happiness and satisfaction. It felt great to know that all the effort "
     "had paid off."),

    # ========================= LOVE =========================
    ("love",
     "I have known my partner for years, and I still feel grateful that we "
     "found each other. Even ordinary evenings feel meaningful when we spend "
     "them together. There is a deep affection and warmth between us that I "
     "value enormously."),
    ("love",
     "Whenever I visit my parents, I notice small things they do to make me "
     "comfortable without expecting anything in return. Those ordinary "
     "gestures remind me how much I care about them. I feel a strong sense "
     "of love for my family."),
    ("love",
     "My old dog always waits by the door when I come home. Seeing him get "
     "excited after a long day makes me smile every time. I feel an enormous "
     "amount of affection for him."),
    ("love",
     "Even after a difficult day, talking to my closest friend makes me feel "
     "understood. We know each other's habits, strengths, and weaknesses, yet "
     "there is still a deep bond between us. I care about that friendship "
     "more than I can easily explain."),

    # ========================= NERVOUSNESS =========================
    ("nervousness",
     "I have to present my research tomorrow, and I keep imagining everything "
     "that could go wrong. I know the material well, but the thought of "
     "standing in front of the panel makes my stomach feel tight. I am "
     "extremely nervous about the presentation."),
    ("nervousness",
     "I am waiting for my first interview with a company I really want to "
     "join. I have prepared several answers, but I keep rehearsing them in "
     "my head anyway. I feel restless because I want to make a good impression."),
    ("nervousness",
     "Tomorrow I have to speak in front of a large audience for the first "
     "time. I know there is no real danger, but my hands become shaky whenever "
     "I think about it. I am anxious about making a mistake in front of everyone."),
    ("nervousness",
     "I submitted the application and now I am waiting for the result. Every "
     "time my phone receives a notification, I wonder whether it is the "
     "response. The uncertainty has left me tense and nervous."),

    # ========================= OPTIMISM =========================
    ("optimism",
     "The project has had several setbacks, but the team is finally making "
     "steady progress. We have learned from the earlier mistakes and now have "
     "a clearer plan. I believe the next phase will go much better."),
    ("optimism",
     "I did not get the first opportunity I applied for, but that experience "
     "helped me improve my application. There are still several other options "
     "available. I feel confident that something good can come from the next "
     "attempt."),
    ("optimism",
     "The situation is difficult right now, but I can already see a few "
     "possible solutions. It will probably take time and patience, yet I do "
     "not think the problem is impossible to solve. I believe things can "
     "improve."),
    ("optimism",
     "My first attempt at the exam did not go as planned, but I now understand "
     "which topics need more work. I have made a study plan and still have "
     "time to prepare. I am hopeful that I can do much better next time."),

    # ========================= PRIDE =========================
    ("pride",
     "I looked at the final version of the project and remembered how much "
     "work went into building it. We started with almost nothing and slowly "
     "turned the idea into something functional. I felt proud of what our team "
     "had accomplished."),
    ("pride",
     "My younger brother completed his first major competition and finished "
     "near the top. I watched how consistently he practiced for months before "
     "the event. Seeing his result made me incredibly proud of him."),
    ("pride",
     "I used to struggle with this subject, but I kept studying until I could "
     "solve the difficult problems on my own. When I finally received a strong "
     "grade, I felt proud because I knew how much effort it represented."),
    ("pride",
     "The presentation went smoothly, and several researchers complimented the "
     "clarity of our work afterward. I had worried that the project was not "
     "good enough, so hearing that feedback meant a lot. I felt proud of what "
     "we had produced."),

    # ========================= REALIZATION =========================
    ("realization",
     "I reread the conversation later and suddenly understood what my friend "
     "had been trying to tell me. The clues had been there earlier, but I "
     "had interpreted them differently at the time. Everything became clear "
     "once I looked at the situation from another perspective."),
    ("realization",
     "While checking the data again, I noticed that one assumption in my "
     "original calculation had been wrong. That explained why the results "
     "looked so strange. I immediately understood where the discrepancy had "
     "come from."),
    ("realization",
     "I had been blaming myself for the disagreement, but then I remembered "
     "that we had never actually discussed the expectation I thought was "
     "obvious. Suddenly the misunderstanding made sense. I realized that both "
     "of us had been working from different assumptions."),
    ("realization",
     "After reading the final paragraph of the article, I understood why the "
     "author had introduced the earlier example. I had initially thought it "
     "was unrelated, but the connection was actually central to the argument. "
     "The whole structure suddenly made sense."),

    # ========================= RELIEF =========================
    ("relief",
     "I had been waiting all day for the test results and was preparing "
     "myself for bad news. When the doctor finally said everything looked "
     "normal, I felt a huge weight disappear. I could finally breathe easily "
     "again."),
    ("relief",
     "The presentation was over, and I realized that none of the things I had "
     "been worrying about had happened. I walked out of the room feeling much "
     "lighter than when I entered. The tension that had been building all "
     "week finally faded."),
    ("relief",
     "I thought I had permanently lost an important file, but then I found a "
     "backup on an old drive. I had been panicking for almost an hour, so "
     "finding the backup made me feel incredibly relieved."),
    ("relief",
     "The storm looked dangerous earlier, but the weather service finally "
     "announced that it was moving away from our area. I had been worried "
     "about the people at home, so the update immediately made me feel safer "
     "and calmer."),

    # ========================= REMORSE =========================
    ("remorse",
     "I said something unnecessarily harsh during an argument yesterday. "
     "At the time I was angry, but afterward I realized that my words had "
     "hurt someone who did not deserve it. I keep thinking about the incident "
     "and wish I had handled it differently."),
    ("remorse",
     "I ignored my friend's message because I was busy, and later I learned "
     "that she had really needed someone to talk to. I cannot change the fact "
     "that I was unavailable when she reached out. I feel genuinely sorry "
     "about not being there."),
    ("remorse",
     "I blamed my teammate for a problem before checking the evidence properly. "
     "Later I discovered that the mistake was actually mine. I apologized, "
     "but I still feel bad that I accused him unfairly."),
    ("remorse",
     "I broke a promise that I had made to someone who trusted me. There was "
     "a reason I failed to keep it, but that does not make the disappointment "
     "I caused disappear. I regret my decision and wish I had acted differently."),

    # ========================= SADNESS =========================
    ("sadness",
     "I came home after a difficult day and realized that I had nobody to talk "
     "to about what had happened. I sat quietly for a long time while the "
     "house stayed completely silent. I felt lonely and deeply sad."),
    ("sadness",
     "The plan I had been looking forward to was cancelled, and everyone else "
     "seemed to move on quickly. I tried to stay busy, but I kept thinking "
     "about how different the day was supposed to be. I felt low and "
     "disheartened."),
    ("sadness",
     "I opened an old message from someone I used to be close to. We had "
     "drifted apart over the years, and reading those words reminded me of "
     "how much had changed. I felt a quiet sadness afterward."),
    ("sadness",
     "I tried to act normal at work, but I could not concentrate on anything. "
     "Everyone around me was talking and laughing while I stared at the "
     "screen. I felt emotionally drained and sad even though I did not want "
     "anyone to notice."),

    # ========================= SURPRISE =========================
    ("surprise",
     "I opened the door expecting a normal delivery, but my friends were "
     "standing outside with a cake. I had no idea they were planning anything "
     "for me. For a few seconds I could not even understand what was happening."),
    ("surprise",
     "The results were completely different from the prediction I had made "
     "before running the experiment. I checked the calculation twice because "
     "the difference was so large. I was genuinely surprised by what the data "
     "showed."),
    ("surprise",
     "My manager called me into the office without explaining why. I expected "
     "to hear about a problem, but instead she told me that I had been selected "
     "for a new opportunity. I was caught completely off guard."),
    ("surprise",
     "I went to the restaurant expecting the usual menu, but they had changed "
     "almost everything since my last visit. Even the layout looked different. "
     "I spent several minutes looking around because I had not expected the "
     "place to have changed so much."),

    # ========================= NEUTRAL =========================
    ("neutral",
     "I went to the store after work and bought vegetables, rice, and some "
     "notebooks. The store was moderately busy, so I waited a few minutes at "
     "the checkout. After paying, I returned home and put everything away."),
    ("neutral",
     "The meeting started at ten and lasted about forty minutes. We discussed "
     "the project schedule, assigned the remaining tasks, and agreed to meet "
     "again next week. I wrote the dates in my calendar before leaving."),
    ("neutral",
     "I opened the laptop, checked my email, and replied to several routine "
     "messages. After that I reviewed the document that was scheduled for "
     "submission. Nothing unusual happened during the morning."),
    ("neutral",
     "The bus arrived at the usual time, and I found an empty seat near the "
     "window. I listened to music during the journey and got off at my normal "
     "stop. I then walked the remaining distance to the office."),
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset() -> None:
    expected_count = len(LABELS) * 4

    if len(DATASET) != expected_count:
        raise RuntimeError(
            f"Dataset should contain {expected_count} samples, "
            f"but contains {len(DATASET)}."
        )

    counts = Counter(label for label, _ in DATASET)

    missing = [label for label in LABELS if counts[label] != 4]

    if missing:
        raise RuntimeError(
            "Dataset is not balanced. Expected exactly 4 samples per "
            f"class. Problem classes: {missing}"
        )

    unknown = [label for label in counts if label not in LABELS]

    if unknown:
        raise RuntimeError(
            f"Unknown dataset labels: {unknown}"
        )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    gold: str
    predicted: str
    score: float
    latency_ms: float
    text: str
    raw_labels: List[Tuple[str, float]]


def predict(
    session: requests.Session,
    api_url: str,
    text: str,
    threshold: float,
) -> Tuple[str, float, List[Tuple[str, float]]]:

    response = session.post(
        f"{api_url}/predict",
        json={
            "text": text,
            "threshold": threshold,
        },
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    emotions = data.get("emotions", [])

    if not emotions:
        return "neutral", 0.0, []

    parsed = []

    for emotion in emotions:
        label = normalize_label(emotion.get("label", ""))
        score = float(emotion.get("score", 0.0))

        parsed.append((label, score))

    # The API normally returns ranked emotions. Sort again so this test
    # does not depend on response ordering.
    parsed.sort(key=lambda x: x[1], reverse=True)

    return parsed[0][0], parsed[0][1], parsed


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Paragraph-level emotion classification metrics."
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    validate_dataset()

    print("=" * 78)
    print("EMOTION CLASSIFICATION — PARAGRAPH METRICS EVALUATION")
    print("=" * 78)
    print(f"API URL:       {args.api_url}")
    print(f"Threshold:     {args.threshold}")
    print(f"Dataset size:  {len(DATASET)} paragraphs")
    print(f"Classes:       {len(LABELS)}")
    print("Samples/class: 4")
    print()

    # --------------------------------------------------------
    # Server check
    # --------------------------------------------------------

    session = requests.Session()

    try:
        r = session.get(
            f"{args.api_url}/",
            timeout=5,
        )

        if r.status_code != 200:
            raise RuntimeError(
                f"Server returned HTTP {r.status_code}"
            )

    except Exception as exc:
        print("ERROR: Could not connect to the API.")
        print(f"Details: {exc}")
        print()
        print(
            "Start the server first, for example:"
        )
        print(
            "uvicorn main:app --host 0.0.0.0 --port 8000"
        )
        raise SystemExit(1)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    gold_labels: List[str] = []
    predicted_labels: List[str] = []
    predictions: List[Prediction] = []

    latencies: List[float] = []

    for index, (gold, text) in enumerate(DATASET, start=1):

        start = time.perf_counter()

        try:
            predicted, score, raw = predict(
                session=session,
                api_url=args.api_url,
                text=text,
                threshold=args.threshold,
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0

        except Exception as exc:

            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0

            print(
                f"[ERROR] Sample {index:03d} "
                f"({gold}): {exc}"
            )

            # Do not silently turn API failures into an emotion prediction.
            # Instead mark the run as incomplete.
            raise SystemExit(
                f"Evaluation stopped because sample {index} failed."
            )

        gold_labels.append(gold)
        predicted_labels.append(predicted)
        latencies.append(latency_ms)

        predictions.append(
            Prediction(
                gold=gold,
                predicted=predicted,
                score=score,
                latency_ms=latency_ms,
                text=text,
                raw_labels=raw,
            )
        )

        if args.verbose:

            status = (
                "OK"
                if predicted == gold
                else "MISS"
            )

            print(
                f"[{status:4s}] {index:03d} "
                f"GOLD={gold:<16} "
                f"PRED={predicted:<16} "
                f"SCORE={score:.3f} "
                f"LAT={latency_ms:.0f}ms"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        gold_labels,
        predicted_labels,
    )

    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            gold_labels,
            predicted_labels,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )

    weighted_p, weighted_r, weighted_f1, _ = (
        precision_recall_fscore_support(
            gold_labels,
            predicted_labels,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )
    )

    per_p, per_r, per_f1, per_support = (
        precision_recall_fscore_support(
            gold_labels,
            predicted_labels,
            labels=LABELS,
            average=None,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        gold_labels,
        predicted_labels,
        labels=LABELS,
    )

    # --------------------------------------------------------
    # Print headline metrics
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("OVERALL METRICS")
    print("=" * 78)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {macro_p:.4f}")
    print(f"Macro Recall:      {macro_r:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted Precision:{weighted_p:.4f}")
    print(f"Weighted Recall:   {weighted_r:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("PER-EMOTION METRICS")
    print("=" * 78)

    print(
        f"{'Emotion':<18}"
        f"{'Precision':>11}"
        f"{'Recall':>11}"
        f"{'F1':>11}"
        f"{'Support':>10}"
    )

    print("-" * 61)

    for label, p, r, f1, support in zip(
        LABELS,
        per_p,
        per_r,
        per_f1,
        per_support,
    ):
        print(
            f"{label:<18}"
            f"{p:>11.3f}"
            f"{r:>11.3f}"
            f"{f1:>11.3f}"
            f"{support:>10}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CONFUSION MATRIX")
    print("=" * 78)

    # Compact CSV-style output is easier to paste into Excel/Python.
    print(
        "gold/pred," +
        ",".join(LABELS)
    )

    for i, label in enumerate(LABELS):

        print(
            label + "," +
            ",".join(str(x) for x in cm[i])
        )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("PREDICTION DISTRIBUTION")
    print("=" * 78)

    predicted_counts = Counter(predicted_labels)

    for label in LABELS:
        print(
            f"{label:<18} "
            f"{predicted_counts[label]:>4}"
        )

    # --------------------------------------------------------
    # Error analysis
    # --------------------------------------------------------

    failures = [
        p for p in predictions
        if p.gold != p.predicted
    ]

    print()
    print("=" * 78)
    print(
        f"ERROR ANALYSIS — {len(failures)} "
        f"incorrect / {len(predictions)} total"
    )
    print("=" * 78)

    if not failures:
        print("No classification errors.")
    else:

        for i, failure in enumerate(failures, start=1):

            print()
            print(
                f"[{i}] GOLD={failure.gold} | "
                f"PRED={failure.predicted} | "
                f"SCORE={failure.score:.3f}"
            )

            print(
                "Text:"
            )

            print(
                failure.text
            )

            if failure.raw_labels:
                print(
                    "Top predictions: " +
                    ", ".join(
                        f"{label}={score:.3f}"
                        for label, score
                        in failure.raw_labels[:5]
                    )
                )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("API LATENCY")
    print("=" * 78)

    sorted_latencies = sorted(latencies)

    def percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0

        index = int(
            round((len(values) - 1) * q)
        )

        return values[index]

    print(
        f"Mean:  {statistics.mean(latencies):.2f} ms"
    )

    print(
        f"Median/P50: "
        f"{percentile(sorted_latencies, 0.50):.2f} ms"
    )

    print(
        f"P95:   "
        f"{percentile(sorted_latencies, 0.95):.2f} ms"
    )

    print(
        f"Max:   "
        f"{max(latencies):.2f} ms"
    )

    # --------------------------------------------------------
    # Research interpretation
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("RESEARCH INTERPRETATION")
    print("=" * 78)

    print(
        "This benchmark measures paragraph-level PRIMARY emotion "
        "classification."
    )

    print(
        "Accuracy measures overall correctness, while Macro-F1 gives "
        "equal importance to every emotion class."
    )

    print(
        "The confusion matrix identifies which emotions are systematically "
        "confused with one another."
    )

    print(
        "Do NOT interpret this score as conversational quality, VAD "
        "tracking quality, or response-generation quality."
    )

    print()
    print(
        "For a research paper, report the dataset composition, label "
        "definition, threshold, model version, and these metrics together."
    )

    print()
    print("=" * 78)
    print("END OF EVALUATION")
    print("=" * 78)


if __name__ == "__main__":
    main()
