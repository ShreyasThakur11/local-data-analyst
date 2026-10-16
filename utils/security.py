"""Security utilities."""
from __future__ import annotations

import re
import hashlib
from pathlib import Path

# SQL keywords that should not appear in read-only analytical queries
_FORBIDDEN_SQL_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE|PRAGMA|ATTACH|DETACH)\b",
    re.IGNORECASE,
)

_FILESYSTEM_PATTERNS = re.compile(
    r"(read_csv|read_parquet|read_json|glob|copy|export)\s*\(",
    re.IGNORECASE,
)


def is_read_only_sql(query: str) -> tuple[bool, str]:
    """
    Verify that a SQL query is read-only.
    Returns (ok, reason).
    """
    q = query.strip()

    if _FORBIDDEN_SQL_PATTERNS.search(q):
        match = _FORBIDDEN_SQL_PATTERNS.search(q)
        return False, f"Forbidden SQL keyword: {match.group()}"

    if _FILESYSTEM_PATTERNS.search(q):
        return False, "Direct filesystem access functions are not permitted."

    if not q.upper().startswith("SELECT"):
        return False, "Only SELECT statements are permitted."

    return True, "OK"


def compute_file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Return SHA-256 hex digest of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_dataframe_hash(df) -> str:  # noqa: ANN001
    """Return a quick fingerprint of a pandas DataFrame."""
    import pandas as pd
    token = f"{list(df.columns)}|{df.shape}|{df.dtypes.to_dict()}"
    return hashlib.md5(token.encode()).hexdigest()  # noqa: S324 – not cryptographic
