import pandas as pd
import logging
from .config import TEXT_COL, DEPT_COL, SCORE_COL

logger = logging.getLogger(__name__)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- answer_text ---
    out[TEXT_COL] = out[TEXT_COL].astype(str).fillna("").str.strip()
    empty_text_count = (out[TEXT_COL] == "").sum()

    # --- department ---
    out[DEPT_COL] = out[DEPT_COL].astype(str).fillna("Unknown").str.strip()

    # --- satisfaction_score ---
    out[SCORE_COL] = pd.to_numeric(out[SCORE_COL], errors="coerce")
    score_nan_count = out[SCORE_COL].isna().sum()

    # --- aggregate log ---
    logger.info(
        "前処理完了: 行数=%d, 空テキスト件数=%d, satisfaction_score NaN件数=%d",
        len(out),
        empty_text_count,
        score_nan_count,
    )

    return out
