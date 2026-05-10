from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.engine import InterviewEngine
from app.llm_client import LLMClient
from app.schemas import AnswerRequest, AnswerResponse, RestoreRequest, StartRequest, StartResponse
from app.session_store import store


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

settings = get_settings()
llm_client = LLMClient(settings)
engine = InterviewEngine(llm_client)

app = FastAPI(title="问脉 VivaScope", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "provider": "dashscope",
        "model_configured": bool(settings.qwen_model),
        "api_key_configured": bool(settings.dashscope_api_key),
        "mock_allowed": settings.allow_mock,
    }


@app.post("/api/sessions", response_model=StartResponse)
async def create_session(request: StartRequest) -> StartResponse:
    if request.mode in {"project", "mixed"} and len(request.project.strip()) < 30:
        raise HTTPException(status_code=422, detail="项目追问或综合模拟需要至少填写一段项目经历。")
    session = await engine.start(request)
    store.save(session)
    return StartResponse(session=session)


@app.get("/api/sessions/{session_id}", response_model=StartResponse)
async def get_session(session_id: str) -> StartResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或服务器已重启。")
    return StartResponse(session=session)


@app.post("/api/sessions/{session_id}/answer", response_model=AnswerResponse)
async def answer_question(session_id: str, request: AnswerRequest) -> AnswerResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或服务器已重启。")
    if session.status == "finished":
        raise HTTPException(status_code=409, detail="本轮训练已结束，请重新开始。")
    session, turn, next_question = await engine.answer(session, request.answer)
    store.save(session)
    return AnswerResponse(
        session=session,
        turn=turn,
        next_question=next_question,
        final_report=session.final_report,
    )


@app.post("/api/sessions/restore", response_model=StartResponse)
async def restore_session(request: RestoreRequest) -> StartResponse:
    session = store.restore(request.session)
    return StartResponse(session=session)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    store.delete(session_id)
    return {"status": "deleted"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)

