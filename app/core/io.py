import pandas as pd
import logging
from .config import TEXT_COL, DEPT_COL, SCORE_COL

logger = logging.getLogger(__name__)

def load_csv(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    cols = set(df.columns)

    # 必須カラムチェック
    if TEXT_COL not in cols:
        logger.error(
            "CSVスキーマエラー: 必須カラム「%s」が存在しません。現在のカラム一覧=%s",
            TEXT_COL,
            list(cols),
        )
        raise ValueError(
            f"必須カラム「{TEXT_COL}」が見つかりません。"
            f"CSVには少なくとも「{TEXT_COL}」カラムを含めてください。"
        )

    # オプションカラムの有無
    has_dept = DEPT_COL in cols
    has_score = SCORE_COL in cols

    if not has_dept:
        df[DEPT_COL] = "Unknown"

    if not has_score:
        df[SCORE_COL] = pd.NA

    # 集約ログ（正常系）
    logger.info(
        "CSV読み込み完了: 行数=%d, カラム数=%d, カラム状態={%s}",
        len(df),
        len(cols),
        ", ".join([
            f"{TEXT_COL}:OK",
            f"{DEPT_COL}:{'OK' if has_dept else '未存在'}",
            f"{SCORE_COL}:{'OK' if has_score else '未存在'}",
        ])
    )

    # フォールバックが発生した場合のみ WARNING
    if not has_dept or not has_score:
        logger.warning(
            "CSVオプションカラムが不足しています。フォールバック処理を適用しました: "
            "department=%s, satisfaction_score=%s",
            "存在" if has_dept else "未存在",
            "存在" if has_score else "未存在",
        )

    return df
