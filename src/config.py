"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str = ""

    # Storage paths
    upload_dir: str = "./data/uploads"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/legal_assistant"

    # Models
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"

    # OCR
    tesseract_cmd: str = "/usr/bin/tesseract"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS – comma-separated list of allowed origins.
    # Use "*" only in development; for production set explicit origins, e.g.:
    # CORS_ALLOWED_ORIGINS=https://myapp.example.com,https://admin.example.com
    cors_allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    def ensure_dirs(self) -> None:
        """Create required storage directories if they don't exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
