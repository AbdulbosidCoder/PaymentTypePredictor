# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = joblib.load('kategoriya_model_full.pkl')





def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^а-яa-zё0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class Transaction(BaseModel):
    naznachenie: str
    bank_schet: str
    turi: str
    oy: str
    mfo: str
    summa: float

@app.post("/predict")
def predict(item: Transaction):
    text_clean = clean_text(item.naznachenie)
    X = pd.DataFrame([{
        'text_clean': text_clean,
        'Банк счет': item.bank_schet,
        'turi': item.turi,
        'oy': item.oy,
        'МФО': item.mfo,
        'summa': item.summa
    }])
    pred = pipeline.predict(X)[0]
    return {"category": pred}
