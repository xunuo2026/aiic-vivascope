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


@lru_cache
def get_settings() -> Settings:
    return Settings()

