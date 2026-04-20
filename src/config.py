"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str = ""

    # Storage paths
    chroma_persist_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"
    database_url: str = "sqlite:///./data/legal_assistant.db"

    # Models
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"

    # OCR
    tesseract_cmd: str = "/usr/bin/tesseract"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    def ensure_dirs(self) -> None:
        """Create required storage directories if they don't exist."""
        for directory in [self.chroma_persist_dir, self.upload_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        # Ensure the directory for SQLite exists
        db_path = self.database_url.replace("sqlite:///", "")
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
