"""BolInterview engine: roles, question bank, session state, scoring.

Sessions live in an in-memory dict keyed by uuid4, like Knowlix's
_sessions. All voice/LLM calls go through the ``voice`` and ``llm``
modules so tests can monkeypatch them — no live calls in tests.

Question bank: original questions written for this app, for a 3rd-year
B.Tech ECE student interviewing for an internship. Four per role:
1 HR/introduction + 3 technical. Hindi (Devanagari) and English versions.
"""

import base64
import uuid

from . import llm, voice

MAX_VOICE_AUDIO_BYTES = 5 * 1024 * 1024
AUDIO_MIME = "audio/mpeg"
QUESTIONS_PER_SESSION = 4

ROLES = [
    {"id": "ai-ml", "name": "AI/ML Intern"},
    {"id": "sde", "name": "Software Developer Intern"},
    {"id": "data-analyst", "name": "Data Analyst Intern"},
]

LANGUAGES = [
    {"id": "hi-IN", "name": "Hindi"},
    {"id": "en-IN", "name": "English"},
]

# role id -> language id -> [4 questions]
QUESTIONS = {
    "ai-ml": {
        "hi-IN": [
            "Apne baare mein batayein — aap kaun hain aur aapko AI aur machine learning mein dilchaspi kyun hai?",
            "Supervised learning aur unsupervised learning mein kya antar hai? Ek-ek example ke saath samjhayein.",
            "Overfitting kya hota hai? Model overfit ho raha ho to aap kya karenge?",
            "Apne kisi project ke baare mein batayein jisme aapne machine learning use kiya ho — problem kya thi aur aapne use kaise solve kiya?",
        ],
        "en-IN": [
            "Tell me about yourself — who are you, and why are you interested in AI and machine learning?",
            "What is the difference between supervised and unsupervised learning? Explain with one example of each.",
            "What is overfitting? What would you do if your model was overfitting?",
            "Tell me about a project where you used machine learning — what was the problem and how did you solve it?",
        ],
    },
    "sde": {
        "hi-IN": [
            "Apne baare mein batayein — aap software development mein kyun aana chahte hain?",
            "API kya hoti hai? REST API kaise kaam karti hai, apne shabdon mein samjhayein.",
            "Ek list mein duplicate values hatane ke liye aap kya approach use karenge? Time complexity bhi batayein.",
            "Git mein merge conflict kya hota hai aur aap use kaise solve karte hain?",
        ],
        "en-IN": [
            "Tell me about yourself — why do you want to get into software development?",
            "What is an API? Explain how a REST API works, in your own words.",
            "How would you remove duplicate values from a list? Also tell me the time complexity.",
            "What is a merge conflict in Git, and how do you resolve it?",
        ],
    },
    "data-analyst": {
        "hi-IN": [
            "Apne baare mein batayein — data analysis mein aapki dilchaspi kaise shuru hui?",
            "SQL mein JOIN kya hota hai? INNER JOIN aur LEFT JOIN mein kya antar hai?",
            "Ek dataset mein missing values hon to aap kya karenge? Apna approach batayein.",
            "Aap kisi company ke sales data se kya insights nikalenge? Apna process step by step batayein.",
        ],
        "en-IN": [
            "Tell me about yourself — how did you get interested in data analysis?",
            "What is a JOIN in SQL? What is the difference between an INNER JOIN and a LEFT JOIN?",
            "What would you do if a dataset had missing values? Walk me through your approach.",
            "What insights would you draw from a company's sales data? Explain your process step by step.",
        ],
    },
}

_sessions: dict = {}


class InterviewError(Exception):
    """An error with an HTTP status, raised by the engine for main.py to map."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _audio_b64(data: bytes):
    """Base64-encode mp3 bytes for JSON, or None when there is no audio."""
    if not data:
        return None
    return base64.b64encode(data).decode("ascii")


def _role_name(role_id: str) -> str:
    for role in ROLES:
        if role["id"] == role_id:
            return role["name"]
    return role_id


def _language_name(language: str) -> str:
    for lang in LANGUAGES:
        if lang["id"] == language:
            return lang["name"]
    return language


def start_session(role: str, language: str) -> dict:
    """Create a session and speak the first question.

    Raises InterviewError: 400 unknown role/language, 503 voice not configured.
    """
    if role not in QUESTIONS:
        raise InterviewError(400, "Unknown role. Pick one of: ai-ml, sde, data-analyst.")
    if language not in ("hi-IN", "en-IN"):
        raise InterviewError(400, "Unknown language. Pick hi-IN or en-IN.")
    if not voice.voice_available():
        raise InterviewError(
            503, "Voice is not configured — set GNANI_API_KEY on the backend."
        )
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "role": role,
        "language": language,
        "index": 0,  # next question to answer
        "results": [],
        "done": False,
        "scorecard": None,
    }
    question = QUESTIONS[role][language][0]
    audio = voice.synthesize_speech(question, language=language)
    return {
        "session_id": session_id,
        "role": role,
        "language": language,
        "question_index": 0,
        "question_text": question,
        "question_audio_base64": _audio_b64(audio),
        "audio_mime": AUDIO_MIME,
    }


def _build_scorecard(session: dict) -> dict:
    """Overall score + strengths/improvements for a finished session."""
    results = session["results"]
    scores = [r["score"] for r in results if r.get("score") is not None]
    overall = round(sum(scores) / len(scores), 1) if scores else None
    summary = llm.summarize_session(results, session["language"])
    return {
        "role": _role_name(session["role"]),
        "language": _language_name(session["language"]),
        "per_question": [
            {
                "question": r["question"],
                "transcript": r["transcript"],
                "score": r["score"],
                "feedback": r["feedback"],
            }
            for r in results
        ],
        "overall_score": overall,
        "strengths": summary.get("strengths") or [],
        "improvements": summary.get("improvements") or [],
    }


def _summary_text(session: dict, scorecard: dict) -> str:
    """Short spoken closing line for the scorecard, in the session language."""
    overall = scorecard["overall_score"]
    if session["language"] == "hi-IN":
        if overall is None:
            return "Interview poora hua. Aapke jawab record kar liye gaye hain — scorecard screen par dekh sakte hain."
        return (f"Interview poora hua. Aapka overall score {overall} hai, 10 mein se. "
                "Scorecard screen par dekh sakte hain — aur jin sawalon mein score kam raha, "
                "unhe bol-bol kar practise karein.")
    if overall is None:
        return "Interview complete. Your answers have been recorded — you can see the scorecard on screen."
    return (f"Interview complete. Your overall score is {overall} out of 10. "
            "You can see the scorecard on screen — and do practise out loud "
            "the questions you scored lowest on.")


def answer_question(session_id: str, audio_bytes: bytes, filename: str) -> dict:
    """Transcribe one spoken answer, evaluate it, and move the interview on.

    Raises InterviewError: 404 unknown session, 400 empty/oversized audio or
    empty transcript, 503 voice not configured.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise InterviewError(404, "Unknown session_id.")
    if session["done"]:
        raise InterviewError(400, "This interview is already finished.")
    if not audio_bytes:
        raise InterviewError(400, "No audio received — please record an answer.")
    if len(audio_bytes) > MAX_VOICE_AUDIO_BYTES:
        raise InterviewError(400, "That recording is too large (max 5 MB, about 30 seconds of speech).")
    if not voice.voice_available():
        raise InterviewError(
            503, "Voice is not configured — set GNANI_API_KEY on the backend."
        )

    language = session["language"]
    transcript = voice.transcribe_audio(audio_bytes, filename=filename, language_code=language)
    if not transcript or not transcript.strip():
        raise InterviewError(400, "No speech detected — please try again.")

    index = session["index"]
    question = QUESTIONS[session["role"]][language][index]
    evaluation = llm.evaluate_answer(question, transcript, language)
    result = {
        "question": question,
        "transcript": transcript,
        "score": evaluation.get("score"),
        "feedback": evaluation.get("feedback") or "",
    }
    session["results"].append(result)
    feedback_audio = voice.synthesize_speech(result["feedback"], language=language)

    base = {
        "session_id": session_id,
        "question_index": index,
        "transcript": transcript,
        "score": result["score"],
        "feedback_text": result["feedback"],
        "feedback_audio_base64": _audio_b64(feedback_audio),
        "audio_mime": AUDIO_MIME,
    }

    if index + 1 >= QUESTIONS_PER_SESSION:
        session["index"] = QUESTIONS_PER_SESSION
        session["done"] = True
        scorecard = _build_scorecard(session)
        session["scorecard"] = scorecard
        summary_text = _summary_text(session, scorecard)
        summary_audio = voice.synthesize_speech(summary_text, language=language)
        base.update({
            "done": True,
            "scorecard": scorecard,
            "summary_text": summary_text,
            "summary_audio_base64": _audio_b64(summary_audio),
        })
        return base

    session["index"] = index + 1
    next_question = QUESTIONS[session["role"]][language][index + 1]
    next_audio = voice.synthesize_speech(next_question, language=language)
    base.update({
        "done": False,
        "next_question_index": index + 1,
        "next_question_text": next_question,
        "next_question_audio_base64": _audio_b64(next_audio),
    })
    return base


def get_scorecard(session_id: str) -> dict:
    """Return the stored scorecard for a finished session."""
    session = _sessions.get(session_id)
    if session is None:
        raise InterviewError(404, "Unknown session_id.")
    if not session["done"] or session["scorecard"] is None:
        raise InterviewError(400, "This interview is not finished yet.")
    return session["scorecard"]


def clear_sessions() -> None:
    """Test helper: drop all sessions."""
    _sessions.clear()
