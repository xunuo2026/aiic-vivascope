from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_error = ""

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.45,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        if not self.settings.has_llm:
            self.last_error = "DASHSCOPE_API_KEY 或 QWEN_MODEL 未配置，已使用本地演示策略。"
            return None

        payload = {
            "model": self.settings.qwen_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.85,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds or self.settings.request_timeout_seconds) as client:
                response = await client.post(self.settings.dashscope_chat_url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:400] if exc.response is not None else ""
            self.last_error = f"DashScope HTTP {exc.response.status_code}: {body}"
            return None
        except httpx.HTTPError as exc:
            self.last_error = f"DashScope 请求失败：{exc}"
            return None

        try:
            raw = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self.last_error = f"DashScope 响应格式异常：{exc}"
            return None

        if isinstance(raw, list):
            raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw)
        parsed = self._parse_json_object(str(raw))
        if parsed is None:
            self.last_error = "模型未返回合法 JSON，已使用本地演示策略。"
        return parsed

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

