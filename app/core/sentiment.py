from dataclasses import dataclass
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import logging
import time
from collections import Counter

from .config import MODEL_ID, TH_LOW, LABEL_NEU, LABEL_POS, LABEL_NEG
from .rules import (
    override_to_neutral_global,
    override_dev_only,
    override_eigyou_only,
    override_to_positive
)


logger = logging.getLogger(__name__)

@dataclass
class SentimentService:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: torch.device

    @staticmethod
    def create() -> "SentimentService":
        t0 = time.perf_counter()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("モデル初期化開始: model_id=%s, device=%s", MODEL_ID, device)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()

        dt = time.perf_counter() - t0
        logger.info("モデル初期化完了: sec=%.2f, device=%s", dt, device)

        return SentimentService(tokenizer=tokenizer, model=model, device=device)

    def _id_to_jp(self, label_id: int) -> str:
        lbl = str(self.model.config.id2label[int(label_id)]).upper()
        if lbl == "POSITIVE":
            return LABEL_POS
        if lbl == "NEGATIVE":
            return LABEL_NEG
        return LABEL_NEU

    def predict_batch(self, texts: List[str], depts: List[str], use_dept_rules: bool = True, batch_size: int = 32, max_length: int = 256) -> Tuple[List[str], List[float]]:
        t0 = time.perf_counter()

        n = len(texts)
        if n == 0:
            logger.warning("推論対象が0件です。処理をスキップします。")
            return [], []
        
        if len(depts) != n:
            logger.error("入力長不一致: texts=%d, depts=%d", n, len(depts))
            raise ValueError(f"Length mismatch: texts={n}, depts={len(depts)}")

        logger.info(
            "推論開始: n=%d, batch_size=%d, max_length=%d, device=%s, dept_rules=%s",
            n, batch_size, max_length, self.device, "有効" if use_dept_rules else "無効"
        )

        labels: List[str] = []
        scores: List[float] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_depts = depts[i:i+batch_size]

            empty_mask = [len((t or "").strip()) == 0 for t in batch_texts]
            safe_texts = [t if (t or "").strip() else " " for t in batch_texts]

            inputs = self.tokenizer(
                safe_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).tolist()
            pred_sc = probs.max(dim=-1).values.tolist()

            for j, (is_empty, pid, sc) in enumerate(zip(empty_mask, pred_ids, pred_sc)):
                sc = float(sc)
                if is_empty:
                    labels.append(LABEL_NEU)
                    scores.append(0.0)
                    continue

                raw = self._id_to_jp(pid)

                # 1) low confidence => neutral
                if sc < TH_LOW:
                    adj = LABEL_NEU
                else:
                    # 2) global neutral markers
                    adj = override_to_neutral_global(batch_texts[j], raw)

                # 3) dept-specific downgrades
                if use_dept_rules:
                    adj = override_dev_only(batch_texts[j], adj, batch_depts[j], sc)
                    adj = override_eigyou_only(batch_texts[j], adj, batch_depts[j], sc)

                # 4) global positive override last
                adj = override_to_positive(batch_texts[j], adj)

                labels.append(adj)
                scores.append(sc)

        dt = time.perf_counter() - t0
        label_counts = Counter(labels)

        empty_count = sum(1 for t in texts if len((t or "").strip()) == 0)
        empty_ratio = empty_count / n if n else 0.0

        logger.info(
            "推論完了: sec=%.2f, ラベル分布=%s, 空テキスト=%d(%.1f%%)",
            dt, dict(label_counts), empty_count, empty_ratio * 100
        )

        if empty_ratio >= 0.3:
            logger.warning(
                "空テキストの割合が高いです: %d/%d (%.1f%%)。結果がニュートラルに偏る可能性があります。",
                empty_count, n, empty_ratio * 100
            )
        return labels, scores
