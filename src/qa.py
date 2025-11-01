from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Callable
import re
import pandas as pd
import numpy as np
import os

from .build_index import TextIndex, TextChunk

AGG_FUNCS = {
    "sum": np.sum,
    "avg": np.mean,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "count": lambda x: len(x),
}

SAFE_DTYPES = (np.number,)

def answer_text_query(index: TextIndex, question: str, top_k: int = 5) -> Dict[str, Any]:
    hits = index.query(question, top_k=top_k)
    return {
        "matches": [
            {"page": h[0].page_num, "score": round(h[1], 4), "text": h[0].text[:1000]}
            for h in hits
        ]
    }

def _find_candidate_tables(tables: List[pd.DataFrame], question: str) -> List[int]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", question)]
    scores = []
    for i, df in enumerate(tables):
        cols = [str(c).lower() for c in df.columns]
        score = sum(t in " ".join(cols) for t in tokens)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i,_ in scores[:3]]

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def answer_table_query(tables: List[pd.DataFrame], question: str) -> Dict[str, Any]:
    if not tables:
        return {"result": "No tables available."}

    cand_idxs = _find_candidate_tables(tables, question)
    for idx in cand_idxs:
        df = _coerce_numeric(tables[idx])

        # Try simple patterns: "sum of <col>", "avg of <col> where <col2> == X"
        m = re.search(r"(sum|avg|mean|min|max|count) of ([A-Za-z0-9_\- ]+)(?: where ([^=<>]+?)\s*(==|=|>|<|>=|<=)\s*([\w\-\.]+))?", question, re.I)
        if m:
            agg, col, cond_col, op, cond_val = m.groups()
            col = col.strip()
            if col in df.columns:
                data = df
                if cond_col and op:
                    cond_col = cond_col.strip()
                    if cond_col in df.columns:
                        # Cast condition value
                        try:
                            cond_val_cast = pd.to_numeric(cond_val, errors="raise")
                        except Exception:
                            cond_val_cast = cond_val
                        ops = {
                            "==": lambda a,b: a == b,
                            "=": lambda a,b: a == b,
                            ">": lambda a,b: a > b,
                            "<": lambda a,b: a < b,
                            ">=": lambda a,b: a >= b,
                            "<=": lambda a,b: a <= b,
                        }
                        data = df[ops[op](df[cond_col], cond_val_cast)]
                series = data[col]
                if np.issubdtype(series.dtype, np.number):
                    val = AGG_FUNCS[agg.lower()](series.dropna().to_numpy())
                    return {"table_index": idx, "operation": agg.lower(), "column": col, "value": float(val)}
                else:
                    if agg.lower() == "count":
                        return {"table_index": idx, "operation": "count", "column": col, "value": int(series.notna().sum())}

        # If pattern not matched, try to show a preview of relevant columns
        cols = [c for c in df.columns if any(tok in str(c).lower() for tok in re.findall(r"[A-Za-z0-9_]+", question.lower()))]
        preview = df[cols].head(5).to_dict(orient="records") if cols else df.head(5).to_dict(orient="records")
        return {"table_index": idx, "preview": preview}

    return {"result": "Could not understand the question for available tables."}
