import pandas as pd
import numpy as np
import logging
from .config import DEPT_COL, SCORE_COL

logger = logging.getLogger(__name__)

def dept_counts(df: pd.DataFrame) -> pd.Series:
    if DEPT_COL not in df.columns:
        logger.warning("部署カラムが存在しません: '%s'", DEPT_COL)
        return pd.Series(dtype=int)
    return df[DEPT_COL].value_counts(dropna=False)

def score_hist(df: pd.DataFrame, bins: int = 5):
    """Return histogram arrays (counts, bin_edges). If score column missing/empty -> empty arrays."""
    if SCORE_COL not in df.columns:
        logger.info("スコアカラムが存在しないため、ヒストグラムをスキップします: '%s'", SCORE_COL)
        return np.array([]), np.array([])

    s = pd.to_numeric(df[SCORE_COL], errors="coerce").dropna()
    if s.empty:
        logger.info("スコアが空のため、ヒストグラムをスキップします")
        return np.array([]), np.array([])

    counts, edges = np.histogram(s, bins=bins)
    return counts, edges

def pivot_dept_sentiment(df: pd.DataFrame, label_col: str, text_col: str) -> pd.DataFrame:
    """Pivot table department x sentiment label counts."""
    missing = [c for c in [DEPT_COL, label_col, text_col] if c not in df.columns]
    if missing:
        logger.warning("pivot作成に必要なカラムが不足しています: %s", missing)
        return pd.DataFrame()

    return (
        df.pivot_table(
            index=DEPT_COL,
            columns=label_col,
            values=text_col,
            aggfunc="count",
            fill_value=0
        ).sort_index()
    )
