from functools import lru_cache
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()


class Settings(BaseModel):
    dashscope_api_key: str = getenv("DASHSCOPE_API_KEY", "")
    qwen_model: str = getenv("QWEN_MODEL", "")
    dashscope_base_url: str = getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    dashscope_asr_model: str = getenv("DASHSCOPE_ASR_MODEL", "qwen3-asr-flash-realtime")
    dashscope_asr_ws_url: str = getenv(
        "DASHSCOPE_ASR_WS_URL",
        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    )
    app_host: str = getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(getenv("APP_PORT", "8000"))
    request_timeout_seconds: float = float(getenv("DASHSCOPE_TIMEOUT_SECONDS", "60"))
    allow_mock: bool = getenv("VIVASCOPE_ALLOW_MOCK", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    @property
    def dashscope_chat_url(self) -> str:
        return f"{self.dashscope_base_url.rstrip('/')}/chat/completions"

    @property
    def has_llm(self) -> bool:
        return bool(self.dashscope_api_key and self.qwen_model)

    @property
    def has_asr(self) -> bool:
        return bool(self.dashscope_api_key and self.dashscope_asr_model and self.dashscope_asr_ws_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
