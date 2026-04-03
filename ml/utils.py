# ml/utils.py
import joblib
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from typing import Dict

def save_pipeline(pipeline, save_dir: str, filename: str = "best_model_pipeline.joblib", metadata: dict = None):
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = {
        "pipeline": pipeline,
        "metadata": metadata or {}
    }
    path = p / filename
    joblib.dump(out, path)
    return path

def load_pipeline(path: str):
    data = joblib.load(path)
    # Support both forms: saved dict or raw pipeline
    if isinstance(data, dict) and 'pipeline' in data:
        return data['pipeline'], data.get('metadata', {})
    else:
        return data, {}

def run_cv(pipeline, X, y, cv=10, scoring=None, n_jobs=-1):
    if scoring is None:
        scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
    cv_res = cross_validate(pipeline, X, y,
                            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                            scoring=scoring, return_train_score=False, n_jobs=n_jobs)
    # aggregate means
    summary = {metric: float(cv_res[f'test_{metric}'].mean()) for metric in scoring}
    # also return full cv_res
    return summary, cv_res

def save_json(obj, path):
    """Save an object to JSON, handling non-serializable ML objects gracefully."""
    def default_serializer(o):
        if hasattr(o, "__class__") and "SMOTE" in str(o.__class__):
            return "SMOTE"
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.int64, np.int32, np.float64, np.float32)):
            return o.item()
        return str(o)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2, default=default_serializer)

def load_csv(path, drop_cols=None):
    df = pd.read_csv(path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df
def sanitize_for_json(obj):
    """Recursively convert non-serializable objects (SMOTE, NumPy arrays) for JSON."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif hasattr(obj, "__class__") and "SMOTE" in str(obj.__class__):
        return "SMOTE"
    elif obj is None:
        return None
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)
