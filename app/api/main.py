from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.sentiment import SentimentService

# 感情分析（推論）専用のFastAPIアプリケーション
app = FastAPI(
    title="フィードバック感情分析 推論API",
    version="0.1.0",
    description="アンケートテキストに対して感情分析（ポジ／ネガ／ニュートラル）を行うAPI"
)

# アプリ起動時にモデルを一度だけロード
# （各リクエストごとにロードしない）
svc = SentimentService.create()


class PredictRequest(BaseModel):
    """
    感情分析リクエストの入力形式
    """
    texts: List[str]                         # 分析対象テキスト一覧
    depts: Optional[List[str]] = None        # 部署名（任意）
    use_dept_rules: bool = False             # 部署別ルールを使用するか


class PredictResponse(BaseModel):
    """
    感情分析レスポンス形式
    """
    labels: List[str]                        # 感情ラベル（pos / neg / neutral）
    scores: List[float]                      # 感情スコア


@app.get("/health")
def health():
    """
    ヘルスチェック用エンドポイント
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    テキスト感情分析を実行する推論エンドポイント
    """
    texts = req.texts
    depts = req.depts if req.depts is not None else [""] * len(texts)

    # texts と depts の件数が一致しない場合はエラー
    if len(depts) != len(texts):
        raise HTTPException(
            status_code=400,
            detail="texts と depts の要素数が一致していません。"
        )

    # 感情分析を実行
    labels, scores = svc.predict_batch(
        texts,
        depts,
        use_dept_rules=req.use_dept_rules,
    )

    return {
        "labels": labels,
        "scores": scores,
    }
