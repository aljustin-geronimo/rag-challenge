from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

from .build_index import TextIndex

# -----------------------------
# Global setup
# -----------------------------
AGG_FUNCS = {
    "sum": np.sum,
    "avg": np.mean,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "count": lambda x: len(x),
}

# Load embedding model once globally
_semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Semantic fuzzy column matcher
# -----------------------------
def find_similar_column(query, df_columns, threshold: float = 0.6):
    """Finds the most semantically similar column name to the query."""
    if not df_columns:
        return None
    query_emb = _semantic_model.encode(query, convert_to_tensor=True)
    cols_emb = _semantic_model.encode(df_columns, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, cols_emb)[0]
    best_idx = sims.argmax().item()
    if sims[best_idx] >= threshold:
        return df_columns[best_idx]
    return None


# -----------------------------
# Text query helper
# -----------------------------
def answer_text_query(index: TextIndex, question: str, top_k: int = 5) -> Dict[str, Any]:
    hits = index.query(question, top_k=top_k)
    return {
        "matches": [
            {"page": h[0].page_num, "score": round(h[1], 4), "text": h[0].text[:1000]}
            for h in hits
        ]
    }


# -----------------------------
# Helper: candidate tables & numeric coercion
# -----------------------------
def _find_candidate_tables(tables: List[pd.DataFrame], question: str) -> List[int]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", question)]
    scores = []
    for i, df in enumerate(tables):
        cols = [str(c).lower() for c in df.columns]
        score = sum(t in " ".join(cols) for t in tokens)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:3]]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        col = out[c]
        if hasattr(col, "dtype") and not isinstance(col, pd.DataFrame):
            try:
                out[c] = pd.to_numeric(col, errors="coerce")
            except Exception:
                pass
    return out


# -----------------------------
# Table query engine (with fuzzy matching)
# -----------------------------
def answer_table_query(tables: List[pd.DataFrame], question: str) -> Dict[str, Any]:
    if not tables:
        return {"result": "No tables available."}

    cand_idxs = _find_candidate_tables(tables, question)
    for idx in cand_idxs:
        df = _coerce_numeric(tables[idx])

        # Try pattern: "sum of <col> where <col2> == X"
        m = re.search(
            r"(sum|avg|mean|min|max|count) of ([A-Za-z0-9_\- ]+)"
            r"(?: where ([^=<>]+?)\s*(==|=|>|<|>=|<=)\s*([\w\-\.]+))?",
            question,
            re.I,
        )
        if m:
            agg, col, cond_col, op, cond_val = m.groups()
            col = col.strip()

            # --- Fuzzy match for target column ---
            if col not in df.columns:
                similar = find_similar_column(col, list(df.columns))
                if similar:
                    print(f"🔍 Using fuzzy match: '{col}' → '{similar}'")
                    col = similar
                else:
                    print(f"⚠️ No similar column found for '{col}'")

            if col in df.columns:
                data = df

                # --- Conditional filtering ---
                if cond_col and op:
                    cond_col = cond_col.strip()
                    if cond_col not in df.columns:
                        similar_cond = find_similar_column(cond_col, list(df.columns))
                        if similar_cond:
                            print(
                                f"🔍 Using fuzzy match for condition: '{cond_col}' → '{similar_cond}'"
                            )
                            cond_col = similar_cond

                    if cond_col in df.columns:
                        try:
                            cond_val_cast = pd.to_numeric(cond_val, errors="raise")
                        except Exception:
                            cond_val_cast = cond_val

                        ops = {
                            "==": lambda a, b: a == b,
                            "=": lambda a, b: a == b,
                            ">": lambda a, b: a > b,
                            "<": lambda a, b: a < b,
                            ">=": lambda a, b: a >= b,
                            "<=": lambda a, b: a <= b,
                        }
                        data = df[ops[op](df[cond_col], cond_val_cast)]

                # --- Perform aggregation ---
                series = data[col]
                if np.issubdtype(series.dtype, np.number):
                    val = AGG_FUNCS[agg.lower()](series.dropna().to_numpy())
                    return {
                        "table_index": idx,
                        "operation": agg.lower(),
                        "column": col,
                        "value": float(val),
                    }
                elif agg.lower() == "count":
                    return {
                        "table_index": idx,
                        "operation": "count",
                        "column": col,
                        "value": int(series.notna().sum()),
                    }

        # --- No aggregation matched: return preview ---
        cols = [
            c
            for c in df.columns
            if any(tok in str(c).lower() for tok in re.findall(r"[A-Za-z0-9_]+", question.lower()))
        ]
        preview = (
            df[cols].head(5).to_dict(orient="records")
            if cols
            else df.head(5).to_dict(orient="records")
        )
        return {"table_index": idx, "preview": preview}

    return {"result": "Could not understand the question for available tables."}
