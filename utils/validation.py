"""Input validation helpers."""
from __future__ import annotations

import re
from pathlib import Path

ALLOWED_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".parquet"}


def validate_file_path(path: str | Path) -> tuple[bool, str]:
    """
    Validate a user-supplied local file path.
    Returns (ok, message).
    Does NOT follow symlinks to protected system paths.
    """
    p = Path(path).resolve()

    # Must exist
    if not p.exists():
        return False, f"File not found: {p}"

    # Must be a regular file
    if not p.is_file():
        return False, f"Path is not a regular file: {p}"

    # Extension check
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{p.suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Basic size check — warn if > 500 MB but do not block
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > 500:
        return True, f"Warning: file is {size_mb:.0f} MB. Loading may be slow."

    return True, "OK"


def sanitise_column_name(name: str) -> str:
    """Produce a safe, normalised column name."""
    s = str(name).strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = s.lower()
    if not s or s[0].isdigit():
        s = "col_" + s
    return s or "unnamed"


def is_safe_sql_identifier(name: str) -> bool:
    """Check that a name is safe to use as a SQL identifier."""
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))
