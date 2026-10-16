"""
Local cache management for DataFrames and profiling results.
Uses Parquet for efficient storage. Keyed by content fingerprint.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _df_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.meta.pkl"

    def cache_dataframe(self, key: str, df: pd.DataFrame) -> None:
        """Cache a DataFrame as Parquet."""
        try:
            df.to_parquet(self._df_path(key), index=False)
        except Exception as exc:
            logger.warning("Failed to cache DataFrame [%s]: %s", key, exc)

    def load_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Load a cached DataFrame. Returns None if not found."""
        path = self._df_path(key)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Failed to load cached DataFrame [%s]: %s", key, exc)
            path.unlink(missing_ok=True)
            return None

    def cache_object(self, key: str, obj: Any) -> None:
        """Cache any serialisable object."""
        try:
            with open(self._meta_path(key), "wb") as fh:
                pickle.dump(obj, fh)
        except Exception as exc:
            logger.warning("Failed to cache object [%s]: %s", key, exc)

    def load_object(self, key: str) -> Optional[Any]:
        path = self._meta_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)  # noqa: S301
        except Exception as exc:
            logger.warning("Failed to load cached object [%s]: %s", key, exc)
            path.unlink(missing_ok=True)
            return None

    def invalidate(self, key: str) -> None:
        self._df_path(key).unlink(missing_ok=True)
        self._meta_path(key).unlink(missing_ok=True)

    def clear_all(self) -> int:
        """Delete all cache files. Returns count of files deleted."""
        count = 0
        for f in self.cache_dir.glob("*"):
            f.unlink(missing_ok=True)
            count += 1
        return count

    def cache_size_mb(self) -> float:
        total = sum(f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file())
        return round(total / 1024 / 1024, 2)

    @staticmethod
    def make_key(source_name: str, sheet: Optional[str], config_hash: str = "") -> str:
        token = f"{source_name}|{sheet or ''}|{config_hash}"
        return hashlib.md5(token.encode()).hexdigest()[:16]  # noqa: S324


# Module-level singleton
_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    global _cache
    if _cache is None:
        from utils.configuration import get_config
        _cache = CacheManager(get_config().cache_dir)
    return _cache
