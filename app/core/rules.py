from .config import (
    LABEL_NEU, LABEL_POS,
    TH_LOW, TH_HIGH,
    DEV_DEPT, EIGYOU_DEPT,
    NEU_MARKERS_GLOBAL, POS_MARKERS_GLOBAL,
    NEU_MARKERS_DEV, NEU_MARKERS_EIGYOU
)

def override_to_neutral_global(text: str, pred_label: str) -> str:
    t = (text or "").strip()
    if not t:
        return LABEL_NEU
    if pred_label == LABEL_NEU:
        return pred_label
    if any(m in t for m in NEU_MARKERS_GLOBAL):
        return LABEL_NEU
    return pred_label

def override_dev_only(text: str, pred_label: str, department: str, score: float) -> str:
    if str(department) != DEV_DEPT:
        return pred_label
    if not (TH_LOW <= score < TH_HIGH):
        return pred_label
    if pred_label != LABEL_POS:
        return pred_label
    t = (text or "").strip()
    if any(m in t for m in NEU_MARKERS_DEV):
        return LABEL_NEU
    return pred_label

def override_eigyou_only(text: str, pred_label: str, department: str, score: float) -> str:
    if str(department) != EIGYOU_DEPT:
        return pred_label
    if not (TH_LOW <= score < TH_HIGH):
        return pred_label
    if pred_label != LABEL_POS:
        return pred_label
    t = (text or "").strip()
    if any(m in t for m in NEU_MARKERS_EIGYOU):
        return LABEL_NEU
    return pred_label

def override_to_positive(text: str, pred_label: str) -> str:
    t = (text or "").strip()
    if not t:
        return pred_label
    if any(m in t for m in POS_MARKERS_GLOBAL):
        return LABEL_POS
    return pred_label
