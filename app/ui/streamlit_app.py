import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging
import requests

from app.core.logging_config import setup_logging
setup_logging()

from app.core.io import load_csv
from app.core.preprocess import clean_df
from app.core.config import DEPT_COL, TEXT_COL, SCORE_COL
from app.core.analytics import dept_counts, pivot_dept_sentiment, score_hist
from app.core.wordclouds import make_wc_text, build_wordcloud

API_URL = os.environ.get("API_URL", "http://localhost:8000")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

if FONT_PATH:
    fm.fontManager.addfont(FONT_PATH)
    font_name = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
else:
    FONT_PATH = None

logging.info(f"[DEBUG] FONT_PATH={FONT_PATH}")
logging.info(f"[DEBUG] matplotlib font.family={plt.rcParams.get('font.family')}")
logging.info(f"[DEBUG] matplotlib font.sans-serif={plt.rcParams.get('font.sans-serif')}")

st.set_page_config(page_title="AIフィードバック分析", layout="wide")
st.title("AIフィードバック分析Webアプリ")

uploaded = st.file_uploader("📄 CSVファイルをアップロードしてください（feedback.csv）", type=["csv"])
use_sample = st.checkbox("サンプルデータを使用（data/feedback.csv）", value=False)

df = None
original_cols = set()

if uploaded is not None:
    header_df = pd.read_csv(uploaded, nrows=0)
    original_cols = set(header_df.columns)
    uploaded.seek(0)
    df = load_csv(uploaded)

elif use_sample:
    if os.path.exists("data/feedback.csv"):
        header_df = pd.read_csv("data/feedback.csv", nrows=0)
        original_cols = set(header_df.columns)
        df = load_csv("data/feedback.csv")
    else:
        st.warning("data/feedback.csv が見つかりません。")

else:
    st.info("左上のアップローダーからCSVをアップロードするか、サンプルデータを選択してください。")
    st.stop()

use_dept_rules = (DEPT_COL in original_cols)

df = clean_df(df)

# FastAPIで推論
texts = df[TEXT_COL].fillna("").astype(str).tolist()
depts = df[DEPT_COL].fillna("").astype(str).tolist()

payload = {
    "texts": texts,
    "depts": depts,
    "use_dept_rules": use_dept_rules,
}

try:
    with st.spinner("感情分析を実行中…（API推論）"):
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=300)
        resp.raise_for_status()
        out = resp.json()
except requests.RequestException as e:
    st.error(f"推論APIへの接続に失敗しました。API_URL={API_URL}\n\n詳細: {e}")
    st.stop()

labels = out.get("labels", [])
scores = out.get("scores", [])

df["sentiment_pred"] = labels
df["sentiment_score_pred"] = scores

st.subheader("プレビュー")
st.dataframe(df.head(50))

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("部署別件数")
    counts = dept_counts(df)
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("department")
    ax.set_ylabel("count")
    st.pyplot(fig)

with col2:
    st.subheader("スコア分布")
    counts_h, edges = score_hist(df, bins=5)
    if len(counts_h) == 0:
        st.info("satisfaction_score がないため、スコア分布は表示しません。")
    else:
        fig, ax = plt.subplots()
        ax.bar(edges[:-1], counts_h, width=(edges[1] - edges[0]), align="edge")
        ax.set_xlabel(SCORE_COL)
        ax.set_ylabel("count")
        st.pyplot(fig)

st.subheader("感情分類（全体）")
sent_counts = df["sentiment_pred"].value_counts()

c3, c4 = st.columns(2)
with c3:
    fig, ax = plt.subplots()
    sent_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("sentiment")
    ax.set_ylabel("count")
    st.pyplot(fig)

with c4:
    fig, ax = plt.subplots()
    sent_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

st.subheader("感情分類（部署別：積み上げ）")
pv = pivot_dept_sentiment(df, label_col="sentiment_pred", text_col=TEXT_COL)
if pv.empty:
    st.info("部署別集計を作成できません（必要なカラムが不足しています）。")
else:
    fig, ax = plt.subplots()
    pv.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("department")
    ax.set_ylabel("count")
    st.pyplot(fig)

st.subheader("ワードクラウド（全体）")
wc_text = make_wc_text(df[TEXT_COL])
wc = build_wordcloud(wc_text, font_path=FONT_PATH)
if wc is None:
    st.info("ワードクラウドを生成できませんでした（有効な単語がありません）。")
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

# Download result CSV
st.subheader("ダウンロード")
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "分析結果CSVをダウンロード",
    data=csv_bytes,
    file_name="feedback_with_sentiment.csv",
    mime="text/csv",
)
