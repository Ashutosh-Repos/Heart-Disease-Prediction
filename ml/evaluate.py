# ml/evaluate.py
"""
Evaluate a saved pipeline on a holdout split and create evaluation artifacts.
Usage:
    python -m ml.evaluate --data-path ../data/Cardiovascular_Disease_Dataset.csv --model-path ../models/best_model_pipeline.joblib
"""

import argparse
from pathlib import Path
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from ml.visualize_metrics import plot_confusion_matrix, plot_roc, plot_pr

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def compute_feature_importances(pipeline, X, y, output_dir: Path, feature_names):
    clf = pipeline.named_steps.get('clf', None)
    pre = pipeline.named_steps.get('preprocessor', None)

    try:
        # Try model internal importances
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            coef = clf.coef_
            if coef.ndim == 1:
                importances = np.abs(coef)
            else:
                importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = None
    except Exception:
        importances = None

    if importances is not None:
        # try to build feature names after preprocessing
        try:
            # use preprocessor to transform a single row to get output length
            Xt = pre.transform(X.head(1))
            n_feats = Xt.shape[1]
            if hasattr(pre, 'get_feature_names_out'):
                names = list(pre.get_feature_names_out())
            elif len(feature_names) == n_feats:
                names = feature_names
            else:
                names = [f"f_{i}" for i in range(n_feats)]
        except Exception:
            names = feature_names or [f"f_{i}" for i in range(len(importances))]
        df_fi = pd.DataFrame(sorted(zip(names, importances), key=lambda x: x[1], reverse=True),
                             columns=['feature', 'importance'])
        df_fi.to_csv(output_dir / "feature_importances.csv", index=False)
        return df_fi

    # fallback to permutation importance
    print("Falling back to permutation importance (slower).")
    result = permutation_importance(pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = result.importances_mean
    try:
        pre = pipeline.named_steps['preprocessor']
        Xt = pre.transform(X.head(1))
        n_feats = Xt.shape[1]
        if hasattr(pre, 'get_feature_names_out'):
            names = list(pre.get_feature_names_out())
        elif len(feature_names) == n_feats:
            names = feature_names
        else:
            names = [f"f_{i}" for i in range(n_feats)]
    except Exception:
        names = feature_names or [f"f_{i}" for i in range(len(perm_importances))]
    df_fi = pd.DataFrame(sorted(zip(names, perm_importances), key=lambda x: x[1], reverse=True),
                         columns=['feature', 'importance'])
    df_fi.to_csv(output_dir / "feature_importances_permutation.csv", index=False)
    return df_fi

def main(args):
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(data_path)
    if args.id_column and args.id_column in df.columns:
        df = df.drop(columns=[args.id_column])
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in data columns: {list(df.columns)}")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)

    # load pipeline (support both dict and raw)
    data = joblib.load(model_path)
    if isinstance(data, dict) and 'pipeline' in data:
        pipeline = data['pipeline']
        metadata = data.get('metadata', {})
    else:
        pipeline = data
        metadata = {}

    # [Rigor] Pure Inference Protocol: Do NOT refit a pre-trained pipeline.
    # This preserves the weights and patterns learned from the training source (e.g. Cardiovascular dataset).
    # pipeline.fit(X_train, y_train) 

    preds = pipeline.predict(X_test)
    if hasattr(pipeline, 'predict_proba'):
        probs = pipeline.predict_proba(X_test)[:, 1]
    else:
        # fallback to decision_function
        if hasattr(pipeline, 'decision_function'):
            scores = pipeline.decision_function(X_test)
            mn, mx = scores.min(), scores.max()
            probs = (scores - mn) / (mx - mn) if mx > mn else np.zeros_like(scores)
        else:
            probs = np.zeros_like(preds, dtype=float)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else None,
        "n_test": int(len(y_test))
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(classification_report(y_test, preds, output_dict=True, zero_division=0), f, indent=2)

    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm, labels=['0', '1'], outpath=out_dir / "confusion_matrix.png")
    plot_roc(y_test, probs, outpath=out_dir / "roc_curve.png")
    plot_pr(y_test, probs, outpath=out_dir / "pr_curve.png")

    # compute feature importances (uses injected metadata preferentially)
    feature_names = metadata.get('features', list(X.columns))
    importances_feats = metadata.get('transformed_feature_names', feature_names)
    try:
        fi_df = compute_feature_importances(pipeline, X_test, y_test, out_dir, importances_feats)
    except Exception as e:
        print("Failed to compute feature importances:", e)
        fi_df = None

    def sanitize_json(data):
        if isinstance(data, dict):
            return {k: sanitize_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [sanitize_json(v) for v in data]
        elif isinstance(data, float) and pd.isna(data):
            return None
        return data

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(sanitize_json({
            "metrics_file": str(out_dir / "metrics.json"),
            "classification_report_file": str(out_dir / "classification_report.json"),
            "confusion_matrix_file": str(out_dir / "confusion_matrix.png"),
            "roc_curve_file": str(out_dir / "roc_curve.png"),
            "pr_curve_file": str(out_dir / "pr_curve.png"),
            "feature_importances_file": str(out_dir / ("feature_importances.csv" if fi_df is not None else "feature_importances_permutation.csv")),
            "metadata": metadata
        }), f, indent=2)

    print("Evaluation done. Artifacts saved to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/Cardiovascular_Disease_Dataset.csv")
    parser.add_argument("--model-path", type=str, default="models/best_model_pipeline.joblib")
    parser.add_argument("--output-dir", type=str, default="models/eval_results")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--id-column", type=str, default="patientid")
    parser.add_argument("--target", type=str, default="target")
    args = parser.parse_args()
    main(args)
