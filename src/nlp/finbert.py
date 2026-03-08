from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import functools

@dataclass
class FinBertResult:
    negative: float
    neutral: float
    positive: float

@functools.lru_cache(maxsize=1)
def _load_finbert(model_name: str):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        raise RuntimeError("transformers 미설치: pip install -r requirements_nlp.txt") from e
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch 미설치: pip install -r requirements_nlp.txt") from e

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl

def score_texts(texts: List[str], model_name: str = "ProsusAI/finbert") -> List[FinBertResult]:
    tok, mdl = _load_finbert(model_name)
    import torch

    out: List[FinBertResult] = []
    for text in texts:
        x = tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = mdl(**x).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
        # common order: negative, neutral, positive
        out.append(FinBertResult(negative=float(probs[0]), neutral=float(probs[1]), positive=float(probs[2])))
    return out
