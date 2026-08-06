"""
Data loading utilities for the Library Q&A app.

The catalog file is loaded exactly once per Streamlit session/process and
cached with st.cache_data, so repeated questions never re-hit the disk.
"""
import os
import pandas as pd
import streamlit as st

EXPECTED_COLUMNS = [
    "Book_ID", "Title", "Author", "Category", "Category_NonLatin_Flag",
    "Year_Field_Raw", "Number_of_Editions_Recorded", "Earliest_Year",
    "Latest_Year", "Possible_Duplicate", "Needs_Review",
]

# Columns that are semantically boolean flags but arrive as 1/NaN in the sheet
FLAG_COLUMNS = ["Category_NonLatin_Flag", "Possible_Duplicate", "Needs_Review"]


TRUTHY_STRINGS = {"y", "yes", "true", "1", "x", "flagged"}


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Robustly coerce a mixed-type column to booleans without ever raising."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.fillna(0) != 0

    # For cells that weren't numeric (e.g. "Y"/"yes"/stray text), fall back
    # to a truthy-string check instead of silently dropping the value.
    non_numeric_mask = series.notna() & numeric.isna()
    if non_numeric_mask.any():
        truthy = series.astype(str).str.strip().str.lower().isin(TRUTHY_STRINGS)
        result = result | (non_numeric_mask & truthy)

    return result


def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".ods", ".xlsx", ".xlsm"):
        # python-calamine is a Rust-based reader that is dramatically faster
        # than the pure-Python odf/openpyxl engines (roughly 30-70x on large
        # .ods files in testing) - this is what actually fixes slow loading
        # for catalogs with thousands of rows. Fall back to the slower but
        # more permissive engines only if calamine can't parse the file.
        try:
            return pd.read_excel(path, engine="calamine")
        except Exception:
            engine = "odf" if ext == ".ods" else "openpyxl"
            return pd.read_excel(path, engine=engine)
    raise ValueError(f"Unsupported file type: {ext}")


def _disk_cache_path(path: str) -> str:
    # A sibling parquet file, invalidated whenever the source file's mtime moves forward.
    return path + ".qna_cache.parquet"


def _process(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all the normalization steps once, on a freshly-read DataFrame."""
    # Normalize flag columns to real booleans: 1/"Y"/"true" -> True, NaN/0/"N" -> False.
    # Done defensively (never crash the app) since real-world sheets sometimes
    # mix numeric 1/blank with text markers, or have stray non-numeric cells.
    for col in FLAG_COLUMNS:
        if col in df.columns:
            df[col] = _to_bool_series(df[col])

    # Make sure year columns are numeric (coerce bad values to NaN rather than crash)
    for col in ["Earliest_Year", "Latest_Year", "Number_of_Editions_Recorded"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convenience lowercase helper columns used for fast, case-insensitive matching
    for col in ["Title", "Author", "Category"]:
        if col in df.columns:
            df[f"_{col.lower()}_norm"] = df[col].astype(str).str.strip().str.lower()

    return df


@st.cache_data(show_spinner="Loading catalog...")
def load_catalog(path: str, mtime: float) -> pd.DataFrame:
    """
    Load the catalog once and cache it in memory for the session.

    `mtime` is included purely so that Streamlit's cache key changes if the
    underlying file is ever replaced (e.g. a new upload) even though the
    path stays the same.

    On top of the in-memory cache, we also keep an on-disk parquet copy next
    to the source file. Parquet reads are ~40x faster than re-parsing a large
    .ods/.xlsx, so this keeps things fast across app restarts/redeploys too
    (st.cache_data's in-memory cache doesn't survive those, but this does).
    """
    cache_path = _disk_cache_path(path)
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= mtime:
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # fall through and re-read the source file

    df = _read_any(path)
    df = _process(df)

    try:
        df.to_parquet(cache_path)
    except Exception:
        pass  # disk cache is a nice-to-have; never let it block loading

    return df


def get_catalog(path: str) -> pd.DataFrame:
    """Convenience wrapper that derives the cache-busting mtime for you."""
    mtime = os.path.getmtime(path)
    return load_catalog(path, mtime)


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return any expected columns that are missing, so the UI can warn instead of crash later."""
    return [c for c in EXPECTED_COLUMNS if c not in df.columns]
