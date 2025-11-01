from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd

def _camelot_available() -> bool:
    try:
        import camelot  # noqa: F401
        return True
    except Exception:
        return False

def _tabula_available() -> bool:
    try:
        import tabula  # noqa: F401
        return True
    except Exception:
        return False

def extract_tables(pdf_path: str, flavor: str = "stream") -> List[pd.DataFrame]:
    """Try multiple backends to extract tables. Returns list of DataFrames."""
    tables: List[pd.DataFrame] = []

    # Try Camelot first (stream, then lattice)
    if _camelot_available():
        import camelot
        try:
            cam_tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
            for t in cam_tables:
                df = t.df
                # Drop all-empty rows/cols
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if not df.empty:
                    tables.append(df)
        except Exception:
            pass
        # try lattice if stream was used
        if flavor == "stream":
            try:
                cam_tables2 = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
                for t in cam_tables2:
                    df = t.df.dropna(how="all").dropna(axis=1, how="all")
                    if not df.empty:
                        tables.append(df)
            except Exception:
                pass

    # Fallback to tabula
    if not tables and _tabula_available():
        import tabula
        try:
            dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
            for df in dfs:
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if not df.empty:
                    tables.append(df)
        except Exception:
            pass

    # Final fallback: pdfplumber's table extraction
    if not tables:
        import pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables():
                        df = pd.DataFrame(table)
                        df = df.dropna(how="all").dropna(axis=1, how="all")
                        if not df.empty:
                            tables.append(df)
        except Exception:
            pass

    # Simple header normalization heuristic: promote first row to header if header-like
    norm_tables: List[pd.DataFrame] = []
    for df in tables:
        if df.shape[0] > 1:
            # If first row has unique non-null values, treat as header
            maybe_header = df.iloc[0]
            if maybe_header.notna().sum() >= min(2, df.shape[1]):
                df2 = df.copy()
                df2.columns = [str(c).strip() for c in maybe_header]
                df2 = df2.iloc[1:].reset_index(drop=True)
                norm_tables.append(df2)
                continue
        norm_tables.append(df.reset_index(drop=True))
    return norm_tables
