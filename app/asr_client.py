from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from uuid import uuid4

import websockets

from app.config import Settings


class DashScopeASRClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ws = None

    async def __aenter__(self) -> "DashScopeASRClient":
        if not self.settings.has_asr:
            raise RuntimeError("语音识别未配置，请检查 DASHSCOPE_API_KEY 和 DASHSCOPE_ASR_MODEL。")
        url = f"{self.settings.dashscope_asr_ws_url}?model={self.settings.dashscope_asr_model}"
        self.ws = await websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {self.settings.dashscope_api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        await self._send(
            {
                "event_id": uuid4().hex,
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm16",
                    "sample_rate": 16000,
                    "input_audio_transcription": {
                        "language": "zh",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                    },
                },
            }
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    async def append_audio(self, chunk: bytes) -> None:
        if not chunk:
            return
        await self._send(
            {
                "event_id": uuid4().hex,
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
        )

    async def finish(self) -> None:
        await self._send({"event_id": uuid4().hex, "type": "session.finish"})

    async def events(self) -> AsyncIterator[dict]:
        if self.ws is None:
            return
        async for message in self.ws:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            translated = self._translate_event(payload)
            if translated is not None:
                yield translated

    async def _send(self, payload: dict) -> None:
        if self.ws is None:
            raise RuntimeError("语音识别连接尚未建立。")
        await self.ws.send(json.dumps(payload, ensure_ascii=False))

    def _translate_event(self, event: dict) -> dict | None:
        event_type = str(event.get("type", ""))
        if event_type == "error":
            error = event.get("error") or {}
            message = error.get("message") or event.get("message") or "语音识别服务返回错误。"
            return {"type": "error", "message": message}

        text = (
            event.get("text")
            or event.get("transcript")
            or event.get("stash")
            or event.get("delta")
            or event.get("output", {}).get("text")
        )

        if (
            event_type == "conversation.item.input_audio_transcription.text"
            or event_type.endswith(".delta")
            or "delta" in event_type
            or event_type.endswith(".speech_delta")
        ):
            return {"type": "partial", "text": text or ""}

        if (
            event_type == "conversation.item.input_audio_transcription.completed"
            or event_type.endswith(".completed")
            or event_type.endswith(".done")
            or "completed" in event_type
        ):
            if text:
                return {"type": "final", "text": text}

        if event_type in {"session.created", "session.updated"}:
            return {"type": "ready"}

        if event_type in {"session.finished", "response.done"}:
            if text:
                return {"type": "final", "text": text}
            return {"type": "done"}

        return None
