# app/core/config.py

# =========================
# CSV / Schema
# =========================

TEXT_COL = "answer_text"
DEPT_COL = "department"
SCORE_COL = "satisfaction_score"


# =========================
# Sentiment labels (JP)
# =========================
LABEL_POS = "ポジ"
LABEL_NEG = "ネガ"
LABEL_NEU = "ニュートラル"


# =========================
# Model config
# =========================
# Japanese sentiment model (HuggingFace)
MODEL_ID = "koheiduck/bert-japanese-finetuned-sentiment"

# Confidence thresholds
TH_LOW = 0.77
TH_HIGH = 0.98

# =========================
# Rule markers 
# =========================
DEV_DEPT = "開発"
EIGYOU_DEPT = "営業"

NEU_MARKERS_GLOBAL = ["問題なく", "ものの", "概ね", "一定"]
POS_MARKERS_GLOBAL = ["おかげ", "満足"]

NEU_MARKERS_DEV = [
    "予定通り", "計画通り", "一部", "スムーズ",
    "進行しています", "進捗しています", "完了できました", "期待します"
]

NEU_MARKERS_EIGYOU = [
    "一部"
]
