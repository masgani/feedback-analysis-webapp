import logging
import os
import re
from typing import List, Optional

from wordcloud import WordCloud

logger = logging.getLogger(__name__)

# fugashi (Japanese tokenizer)
try:
    from fugashi import Tagger
    _tagger = Tagger()
    _FUGASHI_OK = True
except Exception as e:
    _tagger = None
    _FUGASHI_OK = False
    logger.warning("fugashiの初期化に失敗しました。ワードクラウドの分かち書きが無効になります: %s", e)

# Font candidates (Docker-friendly)
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

def resolve_font_path(custom_font_path: Optional[str] = None) -> Optional[str]:
    """Find usable font path for Japanese wordcloud."""
    if custom_font_path and os.path.exists(custom_font_path):
        return custom_font_path
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    logger.warning("日本語フォントが見つかりません。文字化けする可能性があります。")
    return None

def tokenize_ja(text: str) -> List[str]:
    """Tokenize Japanese text using fugashi. If unavailable, fallback to simple split."""
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)

    if not t:
        return []

    # If fugashi is available
    if _FUGASHI_OK and _tagger is not None:
        words = []
        for w in _tagger(t):
            pos = getattr(w.feature, "pos1", None) or w.feature.pos1
            if pos in ("名詞", "形容詞", "動詞"):
                surface = w.surface
                if len(surface) >= 2:
                    words.append(surface)
        return words

    # Fallback (very rough): split by spaces
    # Note: Japanese text often has no spaces, so this is mostly for safety.
    return [x for x in t.split(" ") if len(x) >= 2]

def make_wc_text(series) -> str:
    tokens: List[str] = []
    for t in series.fillna("").astype(str).tolist():
        tokens.extend(tokenize_ja(t))
    return " ".join(tokens)

def build_wordcloud(
    wc_text: str,
    font_path: Optional[str] = None,
    width: int = 1000,
    height: int = 600,
    background_color: str = "white"
) -> Optional[WordCloud]:
    """Build WordCloud. Return None if text is empty."""
    wc_text = (wc_text or "").strip()
    if not wc_text:
        logger.info("ワードクラウド生成をスキップします（トークンが空です）")
        return None

    fp = resolve_font_path(font_path)
    wc = WordCloud(
        font_path=fp,
        width=width,
        height=height,
        background_color=background_color,
        collocations=False
    ).generate(wc_text)

    logger.info("ワードクラウド生成完了: chars=%d, font=%s", len(wc_text), fp or "None")
    return wc
