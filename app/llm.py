"""Groq generation backend for BolInterview.

Follows the Knowlix ``llm.py`` pattern: the API key comes ONLY from the
``GROQ_API_KEY`` environment variable — it is never hardcoded, logged, or
shipped. When no key is set, or the Groq call fails for any reason, every
function returns "" (or None for scores) and callers degrade gracefully —
a score is never invented.

No live Groq call is ever made in tests — monkeypatch ``_get_groq_session``
(or ``requests``) with a fake session.
"""

import json
import logging
import os
import re

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_groq_session = None


def _get_groq_api_key() -> str:
    """Read the Groq API key from the environment (never hardcoded)."""
    return os.getenv("GROQ_API_KEY", "").strip()


def llm_available() -> bool:
    """True when a Groq API key is configured."""
    return bool(_get_groq_api_key())


def _get_groq_session():
    """Return a reusable requests.Session for Groq calls."""
    global _groq_session
    if _groq_session is None and requests is not None:
        _groq_session = requests.Session()
        _groq_session.headers.update({"Content-Type": "application/json"})
    return _groq_session


def _chat_completion(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """One Groq chat-completions call. Returns the reply text or "" on any failure."""
    api_key = _get_groq_api_key()
    if not api_key:
        return ""
    session = _get_groq_session()
    if session is None:
        return ""

    try:
        response = session.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                # gpt-oss-120b is a reasoning model: without this, its internal
                # reasoning can eat the whole max_tokens budget and the reply
                # comes back with empty content (verified live 8 Nov 2026).
                "reasoning_effort": "low",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()
    except Exception as exc:
        # Never raise from generation — callers degrade gracefully.
        logger.warning("Groq call failed (%s).", exc)
        return ""


def _parse_json_object(raw: str):
    """Parse a JSON object out of model output (tolerant of surrounding text)."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def evaluate_answer(question: str, transcript: str, language: str) -> dict:
    """Score one interview answer 0-10 and give short feedback.

    Returns {"score": int 0-10 or None, "feedback": str}. The score is
    None (never invented) when Groq is unavailable or the reply is not
    parseable. Feedback is in the session language (Hindi or English).
    """
    lang_name = "Hindi (Devanagari script)" if language == "hi-IN" else "English"
    no_answer = (not transcript or not transcript.strip()
                 or transcript.strip().lower() in ("i don't know", "i dont know",
                                                   "don't know", "pata nahi",
                                                   "nahi pata", "mujhe nahi pata"))
    system_prompt = (
        "You are a friendly but honest internship interviewer. Score the "
        "candidate's spoken answer from 0 to 10 on relevance, clarity and "
        "correctness combined. Reply with ONLY a JSON object like "
        '{"score": 7, "feedback": "..."}. The feedback is one or two short '
        f"sentences in {lang_name}. Empty or 'I don't know' answers score 0-2 "
        "with encouraging feedback."
    )
    user_prompt = f"Question: {question}\nAnswer: {transcript or '(no answer)'}\nScore it."
    raw = _chat_completion(system_prompt, user_prompt, max_tokens=400)
    parsed = _parse_json_object(raw)
    if parsed is None:
        return {
            "score": None,
            "feedback": ("Evaluation unavailable — answer recorded."
                         if language != "hi-IN"
                         else "Moolyankan uplabdh nahi hai — jawab record kar liya gaya hai."),
        }
    try:
        score = int(parsed.get("score"))
    except (TypeError, ValueError):
        score = None
    if score is not None:
        score = max(0, min(10, score))
    feedback = (parsed.get("feedback") or "").strip()
    if not feedback:
        feedback = ("Evaluation unavailable — answer recorded."
                    if language != "hi-IN"
                    else "Moolyankan uplabdh nahi hai — jawab record kar liya gaya hai.")
    if no_answer and score is not None:
        score = min(score, 2)
    return {"score": score, "feedback": feedback}


def summarize_session(results: list, language: str) -> dict:
    """Turn per-question results into strengths + improvements.

    Returns {"strengths": [1-2 lines], "improvements": [2-3 lines]} in the
    session language. Falls back to honest generic lines (not invented
    judgements) when Groq is unavailable.
    """
    lang_name = "Hindi (Devanagari script)" if language == "hi-IN" else "English"
    if language == "hi-IN":
        fallback = {
            "strengths": ["Aapne chaaro sawalon ke jawab diye."],
            "improvements": ["Jin sawalon mein score kam raha, unke jawab bol-bol kar practise karein."],
        }
    else:
        fallback = {
            "strengths": ["You answered all four questions."],
            "improvements": ["Practise out loud the questions you scored lowest on."],
        }
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"Q{i}: {r.get('question', '')}\n"
            f"Answer: {r.get('transcript') or '(no answer)'}\n"
            f"Score: {r.get('score') if r.get('score') is not None else 'not scored'}"
        )
    system_prompt = (
        "You are a friendly internship interview coach. From the interview "
        "below, write 1-2 short strengths and 2-3 short, specific improvement "
        f"tips in {lang_name}. Reply with ONLY a JSON object like "
        '{"strengths": ["..."], "improvements": ["...", "..."]}.'
    )
    raw = _chat_completion(system_prompt, "\n\n".join(lines), max_tokens=600)
    parsed = _parse_json_object(raw)
    if parsed is None:
        return fallback
    strengths = [s for s in (parsed.get("strengths") or []) if isinstance(s, str) and s.strip()][:2]
    improvements = [s for s in (parsed.get("improvements") or []) if isinstance(s, str) and s.strip()][:3]
    if not strengths or not improvements:
        return fallback
    return {"strengths": strengths, "improvements": improvements}
