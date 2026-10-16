"""
Application configuration management.
Reads from environment variables and .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    # Server
    server_address: str = field(default_factory=lambda: os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost"))
    server_port: int = field(default_factory=lambda: int(os.getenv("STREAMLIT_SERVER_PORT", "8501")))

    # Storage
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", "storage")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "storage/cache")))
    learning_db_path: Path = field(
        default_factory=lambda: Path(os.getenv("LEARNING_DB_PATH", "storage/learning.db"))
    )
    exports_dir: Path = field(default_factory=lambda: Path(os.getenv("EXPORTS_DIR", "storage/exports")))

    # Processing
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "200")))
    sample_rows_preview: int = field(default_factory=lambda: int(os.getenv("SAMPLE_ROWS_PREVIEW", "200")))
    profiling_sample_limit: int = field(
        default_factory=lambda: int(os.getenv("PROFILING_SAMPLE_LIMIT", "50000"))
    )

    # Optional AI
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", ""))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))

    def __post_init__(self) -> None:
        self.storage_dir = Path(self.storage_dir)
        self.cache_dir = Path(self.cache_dir)
        self.learning_db_path = Path(self.learning_db_path)
        self.exports_dir = Path(self.exports_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.cache_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def ollama_enabled(self) -> bool:
        return bool(self.ollama_base_url)


# Singleton
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
