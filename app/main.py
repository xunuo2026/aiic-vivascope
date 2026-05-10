from __future__ import annotations

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.asr_client import DashScopeASRClient
from app.config import get_settings
from app.engine import InterviewEngine
from app.llm_client import LLMClient
from app.resume_parser import extract_pdf_text, heuristic_resume_prefill, merge_prefill, resume_prefill_messages
from app.schemas import AnswerRequest, AnswerResponse, RestoreRequest, ResumeParseResponse, StartRequest, StartResponse
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
        "asr_model_configured": bool(settings.dashscope_asr_model),
        "asr_configured": settings.has_asr,
        "mock_allowed": settings.allow_mock,
    }


@app.websocket("/api/speech/stream")
async def speech_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    if not settings.has_asr:
        await websocket.send_json({"type": "error", "message": "语音识别未配置，请检查 DASHSCOPE_API_KEY 和 DASHSCOPE_ASR_MODEL。"})
        await websocket.close()
        return

    async def safe_send(payload: dict) -> None:
        try:
            await websocket.send_json(payload)
        except RuntimeError:
            pass

    try:
        async with DashScopeASRClient(settings) as asr:
            await safe_send({"type": "ready"})

            async def receive_audio() -> None:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect
                    audio = message.get("bytes")
                    if audio:
                        await asr.append_audio(audio)
                        continue
                    text = message.get("text")
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "stop":
                        await asr.finish()
                        return

            async def forward_events() -> None:
                async for event in asr.events():
                    await safe_send(event)
                    if event.get("type") in {"done", "error"}:
                        return

            receive_task = asyncio.create_task(receive_audio())
            forward_task = asyncio.create_task(forward_events())
            done, pending = await asyncio.wait({receive_task, forward_task}, return_when=asyncio.FIRST_COMPLETED)

            if receive_task in done and not forward_task.done():
                try:
                    await asyncio.wait_for(forward_task, timeout=15)
                except TimeoutError:
                    await safe_send({"type": "done"})

            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await safe_send({"type": "error", "message": f"语音识别连接失败：{exc}"})
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.post("/api/sessions", response_model=StartResponse)
async def create_session(request: StartRequest) -> StartResponse:
    if request.mode in {"project", "mixed"} and len(request.project.strip()) < 30:
        raise HTTPException(status_code=422, detail="项目追问或综合模拟需要至少填写一段项目经历。")
    session = await engine.start(request)
    store.save(session)
    return StartResponse(session=session)


@app.post("/api/resume/parse", response_model=ResumeParseResponse)
async def parse_resume(file: UploadFile = File(...)) -> ResumeParseResponse:
    filename = file.filename or ""
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="请上传 PDF 格式的简历。")
    content = await file.read()
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF 文件过大，请控制在 8MB 以内。")
    try:
        text = extract_pdf_text(content)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PDF 解析失败：{exc}") from exc
    if len(text) < 40:
        raise HTTPException(status_code=422, detail="未能从 PDF 中读取到足够文本，请改用手动填写。")

    fallback = heuristic_resume_prefill(text)
    result = await llm_client.complete_json(resume_prefill_messages(text), temperature=0.2, timeout_seconds=45)
    prefill = merge_prefill(fallback, result)
    source = "dashscope" if result else "local_fallback"
    warning = "" if result else llm_client.last_error
    return ResumeParseResponse(prefill=prefill, source=source, warning=warning)


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
