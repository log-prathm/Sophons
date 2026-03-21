from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BETELGEUSE_", extra="ignore")

    app_name: str = "Betelgeuse"
    api_prefix: str = "/api"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    conversation_window: int = 12
    cpu_threads: int = 8
    llm_max_new_tokens: int = 192
    llm_temperature: float = 0.8
    llm_top_p: float = 0.9
    llm_repetition_penalty: float = 1.05
    tts_default_format: str = "wav"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def stt_dir(self) -> Path:
        return self.models_dir / "STTs"

    @property
    def llm_dir(self) -> Path:
        return self.models_dir / "LLMs"

    @property
    def tts_dir(self) -> Path:
        return self.models_dir / "TTSs"

    @property
    def runtime_dir(self) -> Path:
        return self.project_root / "runtime"

    @property
    def audio_dir(self) -> Path:
        return self.runtime_dir / "audio"

    @property
    def sessions_dir(self) -> Path:
        return self.runtime_dir / "sessions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    return settings

