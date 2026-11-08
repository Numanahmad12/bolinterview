"""BolInterview tests — all Gnani + Groq calls mocked (zero live calls).

Covers the start/answer flow through all four questions to the scorecard,
the evaluation-failure path (no invented score), validation errors, and
language selection (Hindi vs English params passed to the voice layer).
"""

import base64

import pytest
from fastapi.testclient import TestClient

from app import interview, llm, voice
from app.main import app

client = TestClient(app)

FAKE_MP3 = b"FAKE-MP3-BYTES"
FAKE_AUDIO = b"FAKE-RECORDED-AUDIO"


@pytest.fixture(autouse=True)
def _clean_sessions():
    interview.clear_sessions()
    yield
    interview.clear_sessions()


@pytest.fixture()
def voice_on(monkeypatch):
    """Voice configured; STT/TTS replaced with fakes that record their args."""
    calls = {"transcribe": [], "synthesize": []}

    def fake_transcribe(audio_bytes, filename="clip.m4a", language_code="hi-IN"):
        calls["transcribe"].append(
            {"bytes": audio_bytes, "filename": filename, "language_code": language_code}
        )
        return "Yeh mera jawab hai."

    def fake_synthesize(text, language="hi-IN"):
        calls["synthesize"].append({"text": text, "language": language})
        return FAKE_MP3

    monkeypatch.setattr(voice, "voice_available", lambda: True)
    monkeypatch.setattr(voice, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(voice, "synthesize_speech", fake_synthesize)
    return calls


@pytest.fixture()
def evaluator(monkeypatch):
    """Groq evaluation replaced with a fake that scores 8, 6, 9, 7 in order."""
    scores = iter([8, 6, 9, 7])

    def fake_evaluate(question, transcript, language):
        return {"score": next(scores), "feedback": "Good answer, keep it up."}

    def fake_summarize(results, language):
        return {
            "strengths": ["Clear, structured answers."],
            "improvements": ["Add one concrete example per answer.", "Slow down on technical terms."],
        }

    monkeypatch.setattr(llm, "evaluate_answer", fake_evaluate)
    monkeypatch.setattr(llm, "summarize_session", fake_summarize)


def _start(role="ai-ml", language="hi-IN"):
    res = client.post("/api/interview/start", json={"role": role, "language": language})
    assert res.status_code == 200, res.text
    return res.json()


def _answer(session_id, audio=FAKE_AUDIO, filename="clip.m4a"):
    return client.post(
        "/api/interview/answer",
        params={"session_id": session_id},
        files={"audio": (filename, audio, "audio/mp4")},
    )


def test_health_and_roles():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    res = client.get("/api/roles")
    assert res.status_code == 200
    body = res.json()
    assert [r["id"] for r in body["roles"]] == ["ai-ml", "sde", "data-analyst"]
    assert [lang["id"] for lang in body["languages"]] == ["hi-IN", "en-IN"]


def test_start_returns_first_question(voice_on):
    body = _start("ai-ml", "hi-IN")
    assert body["session_id"]
    assert body["question_index"] == 0
    assert body["question_text"] == interview.QUESTIONS["ai-ml"]["hi-IN"][0]
    assert base64.b64decode(body["question_audio_base64"]) == FAKE_MP3
    assert body["audio_mime"] == "audio/mpeg"
    assert voice_on["synthesize"][0]["language"] == "hi-IN"


def test_start_invalid_role_and_language(voice_on):
    assert client.post("/api/interview/start", json={"role": "nope", "language": "hi-IN"}).status_code == 400
    assert client.post("/api/interview/start", json={"role": "ai-ml", "language": "fr-FR"}).status_code == 400


def test_start_missing_key_503(monkeypatch):
    monkeypatch.setattr(voice, "voice_available", lambda: False)
    res = client.post("/api/interview/start", json={"role": "ai-ml", "language": "hi-IN"})
    assert res.status_code == 503
    assert "GNANI_API_KEY" in res.json()["detail"]


def test_full_flow_four_questions_to_scorecard(voice_on, evaluator):
    body = _start("ai-ml", "hi-IN")
    session_id = body["session_id"]
    for i in range(4):
        res = _answer(session_id)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["transcript"] == "Yeh mera jawab hai."
        assert data["question_index"] == i
        assert base64.b64decode(data["feedback_audio_base64"]) == FAKE_MP3
        if i < 3:
            assert data["done"] is False
            assert data["next_question_index"] == i + 1
            assert data["next_question_text"] == interview.QUESTIONS["ai-ml"]["hi-IN"][i + 1]
            assert base64.b64decode(data["next_question_audio_base64"]) == FAKE_MP3
        else:
            assert data["done"] is True
            card = data["scorecard"]
            assert len(card["per_question"]) == 4
            assert [q["score"] for q in card["per_question"]] == [8, 6, 9, 7]
            assert card["overall_score"] == 7.5  # (8+6+9+7)/4
            assert card["strengths"] == ["Clear, structured answers."]
            assert len(card["improvements"]) == 2
            assert data["summary_text"]
            assert base64.b64decode(data["summary_audio_base64"]) == FAKE_MP3
    # Scorecard endpoint returns the stored card.
    res = client.get(f"/api/interview/scorecard/{session_id}")
    assert res.status_code == 200
    assert res.json()["overall_score"] == 7.5


def test_evaluation_failure_no_invented_score(voice_on, monkeypatch):
    monkeypatch.setattr(
        llm, "evaluate_answer",
        lambda q, t, lang: {"score": None, "feedback": "Evaluation unavailable — answer recorded."},
    )
    monkeypatch.setattr(
        llm, "summarize_session",
        lambda results, lang: {
            "strengths": ["You answered all four questions."],
            "improvements": ["Practise out loud the questions you scored lowest on."],
        },
    )
    body = _start("sde", "en-IN")
    session_id = body["session_id"]
    last = None
    for _ in range(4):
        res = _answer(session_id)
        assert res.status_code == 200, res.text
        last = res.json()
        assert last["score"] is None
        assert "unavailable" in last["feedback_text"].lower()
    assert last["done"] is True
    assert last["scorecard"]["overall_score"] is None
    assert last["scorecard"]["strengths"]
    assert last["scorecard"]["improvements"]


def test_answer_unknown_session_404(voice_on):
    res = _answer("no-such-session")
    assert res.status_code == 404


def test_answer_empty_audio_400(voice_on):
    body = _start()
    res = _answer(body["session_id"], audio=b"")
    assert res.status_code == 400


def test_answer_oversized_audio_400(voice_on):
    body = _start()
    res = _answer(body["session_id"], audio=b"x" * (5 * 1024 * 1024 + 1))
    assert res.status_code == 400


def test_answer_empty_transcript_400(voice_on, monkeypatch):
    monkeypatch.setattr(voice, "transcribe_audio", lambda *a, **k: "")
    body = _start()
    res = _answer(body["session_id"])
    assert res.status_code == 400
    assert "No speech detected" in res.json()["detail"]


def test_answer_after_done_400(voice_on, evaluator):
    body = _start()
    session_id = body["session_id"]
    for _ in range(4):
        assert _answer(session_id).status_code == 200
    assert _answer(session_id).status_code == 400


def test_scorecard_before_finish_400(voice_on):
    body = _start()
    res = client.get(f"/api/interview/scorecard/{body['session_id']}")
    assert res.status_code == 400
    assert client.get("/api/interview/scorecard/nope").status_code == 404


def test_language_selection_changes_voice_params(voice_on):
    body = _start("data-analyst", "en-IN")
    assert voice_on["synthesize"][0]["language"] == "en-IN"
    assert body["question_text"] == interview.QUESTIONS["data-analyst"]["en-IN"][0]
    res = _answer(body["session_id"])
    assert res.status_code == 200
    assert voice_on["transcribe"][0]["language_code"] == "en-IN"
    assert voice_on["transcribe"][0]["filename"] == "clip.m4a"


def test_voice_for_language_mapping():
    assert voice.voice_for_language("hi-IN") == "Nalini"
    assert voice.voice_for_language("en-IN") == "Kaveri"
    assert voice.voice_for_language("xx-YY") == "Nalini"  # default


def test_question_bank_shape():
    for role in ("ai-ml", "sde", "data-analyst"):
        for lang in ("hi-IN", "en-IN"):
            questions = interview.QUESTIONS[role][lang]
            assert len(questions) == 4
            assert all(isinstance(q, str) and q.strip() for q in questions)
