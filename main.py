# app.py
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "clf_model.joblib")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
MODEL_VERSION = os.getenv("MODEL_VERSION", "unversioned")

state = {"pipeline": None, "model_ok": False}


# --------------------------------------------------------------------------
# Startup / shutdown
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["pipeline"] = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")

        # Fail fast: run a dummy prediction so schema/column mismatches
        # surface at deploy time, not on the first real request.
        dummy = pd.DataFrame([{
            "text_clean": "test",
            "amount": 1.0,
            "is_debit": 1,
            "is_credit": 0,
        }])
        state["pipeline"].predict(dummy)
        state["model_ok"] = True
        logger.info("Startup sanity check passed.")
    except Exception as e:
        logger.error(f"Model failed to load or sanity-check: {e}")
        state["pipeline"] = None
        state["model_ok"] = False
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ORIGINS != ["*"],  # can't combine "*" with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Text cleaning (must match training preprocessing exactly)
# --------------------------------------------------------------------------
def clean_text(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^а-яa-zё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class Transaction(BaseModel):
    naznachenie: str
    debit_summa: Optional[float] = 0
    kredit_summa: Optional[float] = 0

    @model_validator(mode="after")
    def check_at_least_one_summa(self):
        if not self.debit_summa and not self.kredit_summa:
            raise ValueError("At least one of debit_summa or kredit_summa must be provided and non-zero.")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "naznachenie": "Оплата за товар Wildberries",
                "debit_summa": 150000,
                "kredit_summa": 0,
            }
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def build_features(item: Transaction) -> pd.DataFrame:
    is_debit = 1 if item.debit_summa else 0
    is_credit = 1 if item.kredit_summa else 0
    amount = (item.debit_summa or 0) + (item.kredit_summa or 0)
    return pd.DataFrame([{
        "text_clean": clean_text(item.naznachenie),
        "amount": amount,
        "is_debit": is_debit,
        "is_credit": is_credit,
    }])


def predict_with_confidence(pipeline, X: pd.DataFrame) -> tuple[str, Optional[float]]:
    pred = pipeline.predict(X)[0]
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)[0]
            confidence = float(max(proba))
        except Exception:
            pass  # some pipelines/estimators don't support predict_proba cleanly
    return pred, confidence


def require_model():
    if not state["model_ok"] or state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    return state["pipeline"]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok" if state["model_ok"] else "degraded", "model_loaded": state["model_ok"]}


@app.get("/model-info")
async def model_info():
    require_model()
    return {"model_version": MODEL_VERSION, "model_path": MODEL_PATH}


@app.post("/predict")
async def predict(item: Transaction):
    pipeline = require_model()

    try:
        X = build_features(item)
        cat, conf = predict_with_confidence(pipeline, X)
        return {
            "category": cat,
            "confidence": conf,
            "is_debit": int(X.iloc[0]["is_debit"]),
            "is_credit": int(X.iloc[0]["is_credit"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


class BatchRequest(BaseModel):
    transactions: list[Transaction]


@app.post("/predict/batch")
async def predict_batch(body: BatchRequest):
    """
    Bulk categorization for 1C export rows — one model call for the whole batch
    instead of one HTTP round-trip per row (much faster from Sheets/Apps Script).
    """
    pipeline = require_model()
    items = body.transactions

    if not items:
        raise HTTPException(status_code=400, detail="No transactions provided.")

    X = pd.concat([build_features(item) for item in items], ignore_index=True)

    try:
        preds = pipeline.predict(X)
        confidences = [None] * len(preds)
        if hasattr(pipeline, "predict_proba"):
            try:
                probas = pipeline.predict_proba(X)
                confidences = [float(max(p)) for p in probas]
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    results = [
        {"category": pred, "confidence": conf}
        for pred, conf in zip(preds, confidences)
    ]
    return {"results": results, "count": len(results)}