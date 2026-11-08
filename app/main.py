"""BolInterview — a voice mock-interviewer for internship candidates.

FastAPI backend + a single static frontend served by the same app.

Endpoints:
    GET    /                          the frontend (static/index.html)
    GET    /api/health                liveness + which keys are configured
    GET    /api/roles                 roles + languages
    POST   /api/interview/start       {role, language} -> first spoken question
    POST   /api/interview/answer     multipart audio + session_id -> transcript,
                                     score, spoken feedback, next question or scorecard
    GET    /api/interview/scorecard/{session_id}

Voice: Gnani Prisma v2.5 STT + Timbre v2.5 TTS (key from GNANI_API_KEY only).
Evaluation: Groq openai/gpt-oss-120b (key from GROQ_API_KEY only).
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import interview, llm, voice

APP_VERSION = "1.0.1"

app = FastAPI(title="BolInterview", version=APP_VERSION)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class StartRequest(BaseModel):
    role: str
    language: str = "hi-IN"


def _map_error(exc: interview.InterviewError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "gnani_configured": voice.voice_available(),
        "groq_configured": llm.llm_available(),
    }


@app.get("/api/roles")
def roles() -> dict:
    return {"roles": interview.ROLES, "languages": interview.LANGUAGES}


@app.post("/api/interview/start")
def start(req: StartRequest) -> dict:
    try:
        return interview.start_session(req.role, req.language)
    except interview.InterviewError as exc:
        raise _map_error(exc)


@app.post("/api/interview/answer")
async def answer(session_id: str, audio: UploadFile = File(...)) -> dict:
    data = await audio.read()
    try:
        return interview.answer_question(
            session_id, data, audio.filename or "clip.m4a"
        )
    except interview.InterviewError as exc:
        raise _map_error(exc)


@app.get("/api/interview/scorecard/{session_id}")
def scorecard(session_id: str) -> dict:
    try:
        return interview.get_scorecard(session_id)
    except interview.InterviewError as exc:
        raise _map_error(exc)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
