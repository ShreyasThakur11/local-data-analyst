"""
Data ingestion module.
Handles Excel workbooks, CSV, TSV, and Parquet files.
All reading is non-destructive: source files are never modified.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class SheetInfo:
    name: str
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    has_merged_cells: bool = False
    has_formulas: bool = False
    blank_rows_detected: int = 0
    blank_cols_detected: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class WorkbookInspection:
    path: str
    sheet_names: List[str]
    sheets: Dict[str, SheetInfo]
    format: str
    file_size_mb: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    ok: bool
    df: Optional[pd.DataFrame] = None
    source_name: str = ""
    sheet_name: Optional[str] = None
    header_row: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ── Workbook inspection ────────────────────────────────────────────────────────

def inspect_workbook(path: Path | str) -> WorkbookInspection:
    """
    Inspect an Excel workbook and return metadata without loading full data.
    Never modifies or executes anything in the workbook.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0

    inspection = WorkbookInspection(
        path=str(path),
        sheet_names=[],
        sheets={},
        format=suffix,
        file_size_mb=round(size_mb, 2),
    )

    if suffix in {".xlsx", ".xlsm"}:
        _inspect_openpyxl(path, inspection)
    elif suffix == ".xls":
        _inspect_xlrd(path, inspection)
    else:
        inspection.errors.append(f"Unsupported workbook format: {suffix}")

    return inspection


def _inspect_openpyxl(path: Path, inspection: WorkbookInspection) -> None:
    try:
        import openpyxl
        # read_only + data_only to avoid executing macros or formulas
        wb = openpyxl.load_workbook(
            str(path), read_only=True, data_only=True, keep_links=False
        )
        inspection.sheet_names = wb.sheetnames

        for name in wb.sheetnames:
            ws = wb[name]
            info = SheetInfo(name=name)

            try:
                # openpyxl read_only sheets expose max_row / max_column
                info.n_rows = ws.max_row
                info.n_cols = ws.max_column
            except Exception:
                pass

            # Merged cells are not accessible in read_only mode — note this
            info.notes.append("Merged-cell detection requires non-read-only mode.")

            inspection.sheets[name] = info

        wb.close()
    except Exception as exc:
        inspection.errors.append(f"openpyxl error: {exc}")


def _inspect_xlrd(path: Path, inspection: WorkbookInspection) -> None:
    try:
        import xlrd
        wb = xlrd.open_workbook(str(path))
        inspection.sheet_names = wb.sheet_names()

        for name in wb.sheet_names():
            ws = wb.sheet_by_name(name)
            info = SheetInfo(name=name, n_rows=ws.nrows, n_cols=ws.ncols)
            inspection.sheets[name] = info

        inspection.warnings.append(
            ".xls files use the legacy xlrd parser. "
            "Formula values may not be current if the workbook was not recalculated before saving."
        )
    except ImportError:
        inspection.errors.append(
            "xlrd is required to read .xls files. Install it with: pip install xlrd>=2.0"
        )
    except Exception as exc:
        inspection.errors.append(f"xlrd error: {exc}")


# ── Sheet loading ──────────────────────────────────────────────────────────────

def load_excel_sheet(
    path: Path | str,
    sheet_name: str,
    header_row: int = 0,
    nrows: Optional[int] = None,
) -> IngestionResult:
    """
    Load a single worksheet into a DataFrame.
    header_row: 0-indexed row to use as column headers.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    warnings: List[str] = []

    if suffix in {".xlsx", ".xlsm"}:
        engine = "openpyxl"
    elif suffix == ".xls":
        engine = "xlrd"
        warnings.append(
            "Legacy .xls format: formula-derived values are read from cached values only."
        )
    else:
        return IngestionResult(
            ok=False,
            source_name=path.name,
            errors=[f"Unsupported workbook format: {suffix}"],
        )

    try:
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=header_row,
            nrows=nrows,
            engine=engine,
        )
    except Exception as exc:
        logger.exception("Excel load error")
        return IngestionResult(
            ok=False,
            source_name=path.name,
            errors=[f"Failed to load sheet '{sheet_name}': {exc}"],
        )

    df, post_warnings = _post_process_df(df, source=f"{path.name}[{sheet_name}]")
    warnings.extend(post_warnings)

    return IngestionResult(
        ok=True,
        df=df,
        source_name=path.name,
        sheet_name=sheet_name,
        header_row=header_row,
        warnings=warnings,
        meta={
            "file_path": str(path),
            "sheet": sheet_name,
            "header_row": header_row,
            "shape": df.shape,
        },
    )


def load_excel_from_bytes(
    data: bytes,
    filename: str,
    sheet_name: str,
    header_row: int = 0,
) -> IngestionResult:
    """Load Excel sheet from an in-memory byte buffer (file upload)."""
    suffix = Path(filename).suffix.lower()
    warnings: List[str] = []

    if suffix in {".xlsx", ".xlsm"}:
        engine = "openpyxl"
    elif suffix == ".xls":
        engine = "xlrd"
        warnings.append("Legacy .xls format: cached formula values only.")
    else:
        return IngestionResult(
            ok=False,
            source_name=filename,
            errors=[f"Unsupported format: {suffix}"],
        )

    buf = io.BytesIO(data)
    try:
        df = pd.read_excel(buf, sheet_name=sheet_name, header=header_row, engine=engine)
    except Exception as exc:
        return IngestionResult(
            ok=False,
            source_name=filename,
            errors=[f"Failed to load '{sheet_name}': {exc}"],
        )

    df, post_warnings = _post_process_df(df, source=f"{filename}[{sheet_name}]")
    warnings.extend(post_warnings)

    return IngestionResult(
        ok=True,
        df=df,
        source_name=filename,
        sheet_name=sheet_name,
        header_row=header_row,
        warnings=warnings,
        meta={"filename": filename, "sheet": sheet_name, "shape": df.shape},
    )


def get_excel_sheet_names(data: bytes, filename: str) -> Tuple[List[str], List[str]]:
    """Return (sheet_names, warnings) from an in-memory workbook."""
    suffix = Path(filename).suffix.lower()
    buf = io.BytesIO(data)
    warnings = []

    try:
        if suffix in {".xlsx", ".xlsm"}:
            import openpyxl
            wb = openpyxl.load_workbook(buf, read_only=True, data_only=True, keep_links=False)
            names = wb.sheetnames
            wb.close()
        elif suffix == ".xls":
            import xlrd
            wb = xlrd.open_workbook(file_contents=buf.read())
            names = wb.sheet_names()
            warnings.append("Legacy .xls: formula values are cached values only.")
        else:
            return [], [f"Unsupported format: {suffix}"]
    except Exception as exc:
        return [], [str(exc)]

    return names, warnings


# ── CSV / TSV ─────────────────────────────────────────────────────────────────

def load_csv(
    path: Path | str | None = None,
    data: bytes | None = None,
    filename: str = "upload.csv",
    sep: Optional[str] = None,
    encoding: str = "utf-8-sig",
    nrows: Optional[int] = None,
) -> IngestionResult:
    """Load a CSV or TSV file."""
    warnings: List[str] = []
    name = filename if data else Path(path).name

    # Auto-detect separator
    if sep is None:
        suffix = Path(filename if data else path).suffix.lower()
        sep = "\t" if suffix == ".tsv" else ","

    try:
        if data:
            source = io.BytesIO(data)
        else:
            source = path

        df = pd.read_csv(source, sep=sep, encoding=encoding, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        try:
            if data:
                source = io.BytesIO(data)
            else:
                source = path
            df = pd.read_csv(source, sep=sep, encoding="latin-1", nrows=nrows, low_memory=False)
            warnings.append("File decoded with latin-1 fallback encoding.")
        except Exception as exc:
            return IngestionResult(ok=False, source_name=name, errors=[str(exc)])
    except Exception as exc:
        return IngestionResult(ok=False, source_name=name, errors=[str(exc)])

    df, post_warnings = _post_process_df(df, source=name)
    warnings.extend(post_warnings)

    return IngestionResult(
        ok=True,
        df=df,
        source_name=name,
        warnings=warnings,
        meta={"shape": df.shape, "sep": sep},
    )


# ── Parquet ───────────────────────────────────────────────────────────────────

def load_parquet(
    path: Path | str | None = None,
    data: bytes | None = None,
    filename: str = "upload.parquet",
    columns: Optional[List[str]] = None,
) -> IngestionResult:
    """Load a Parquet file."""
    name = filename if data else Path(path).name
    try:
        if data:
            source = io.BytesIO(data)
        else:
            source = path

        df = pd.read_parquet(source, columns=columns)
    except Exception as exc:
        return IngestionResult(ok=False, source_name=name, errors=[str(exc)])

    df, warnings = _post_process_df(df, source=name)
    return IngestionResult(ok=True, df=df, source_name=name, warnings=warnings, meta={"shape": df.shape})


# ── Post-processing ───────────────────────────────────────────────────────────

def _post_process_df(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalise a freshly loaded DataFrame:
    - Handle blank/None column names
    - Detect and suffix duplicate column names
    - Strip leading/trailing whitespace from string columns
    - Drop entirely blank rows and columns
    - Issue warnings for detected problems
    """
    warnings: List[str] = []

    # Rename None / NaN column headers
    new_cols: List[str] = []
    seen: Dict[str, int] = {}
    unnamed_count = 0

    for col in df.columns:
        if pd.isna(col) or str(col).strip() == "":
            unnamed_count += 1
            col = f"Unnamed_{unnamed_count}"
        else:
            col = str(col).strip()

        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
            warnings.append(f"Duplicate column renamed to '{col}'.")
        else:
            seen[col] = 0

        new_cols.append(col)

    df.columns = new_cols

    if unnamed_count:
        warnings.append(f"{unnamed_count} column(s) had blank or missing headers and were renamed.")

    # Drop fully blank rows
    blank_rows_before = len(df)
    df = df.dropna(how="all")
    dropped = blank_rows_before - len(df)
    if dropped:
        warnings.append(f"{dropped} fully blank row(s) removed.")

    # Drop fully blank columns
    blank_cols = [c for c in df.columns if df[c].isna().all()]
    if blank_cols:
        df = df.drop(columns=blank_cols)
        warnings.append(f"Removed {len(blank_cols)} entirely blank column(s): {blank_cols[:5]}")

    # Strip string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        try:
            df[col] = df[col].astype(str).str.strip().replace("nan", pd.NA)
        except Exception:
            pass

    return df, warnings
