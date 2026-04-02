# ml/train.py
"""
Train multiple models, run CV, pick best by ROC AUC, and save best pipeline.
Usage:
    python -m ml.train --data-path data/Cardiovascular_Disease_Dataset.csv --target target
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from ml.preprocessing import build_preprocessor, infer_feature_groups
from ml.model_config import get_models
from ml.utils import run_cv, save_pipeline, save_json
from ml.visualize_metrics import generate_all_plots, generate_model_detail_plots

import warnings
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

def main(args):
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not in dataset columns: {list(df.columns)}")

    # drop identifier if exists
    if args.id_column and args.id_column in df.columns:
        df = df.drop(columns=[args.id_column])

    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Log missing data stats
    missing_stats = X.isnull().mean() * 100
    print("\n[Data Quality] Missing Value Percentages:")
    for col, val in missing_stats.items():
        if val > 0:
            print(f"  - {col}: {val:.2f}%")
    print("-" * 30)

    numeric_feats, categorical_feats = infer_feature_groups(list(X.columns), numeric_hints=args.numeric_hints)
    # If infer messed up and some truly numeric columns are in categorical, you can override by CLI hints.

    print("Numeric features:", numeric_feats)
    print("Categorical features:", categorical_feats)

    preprocessor = build_preprocessor(numeric_feats, categorical_feats)
    models = get_models(random_state=args.random_state)

    # 80/20 split for generating detailed plots (CM, ROC, PR) for all models
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_state
    )

    cv_results = {}
    fold_metrics = {}
    pipelines = {}

    report_dir = Path(args.output_dir) / "comparison_plots"

    for name, clf in models.items():
        print(f"\nRunning CV for: {name}")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])
        scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
        summary, full_cv = run_cv(pipeline, X, y, cv=args.cv, scoring=scoring, n_jobs=args.n_jobs)
        cv_results[name] = summary
        
        # Capture per-fold results
        # convert numpy arrays to lists for JSON serialization
        serializable_cv = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in full_cv.items()}
        fold_metrics[name] = serializable_cv
        
        # Generate detailed plots (CM, ROC, PR) for THIS model
        print(f"Generating detailed evaluation plots for {name}...")
        pipeline.fit(X_train_eval, y_train_eval)
        generate_model_detail_plots(pipeline, X_test_eval, y_test_eval, name, report_dir)

        pipelines[name] = pipeline
        print(f"CV mean scores for {name}: {summary}")

    # pick best model by roc_auc
    best_name = max(cv_results.keys(), key=lambda n: cv_results[n]['roc_auc'])
    print(f"\nBest model by ROC AUC: {best_name} -> {cv_results[best_name]}")

    # Fit best pipeline on full data and save
    best_pipeline = pipelines[best_name]
    print("Fitting best pipeline on full data...")
    best_pipeline.fit(X, y)

    metadata = {
        "features": list(X.columns),
        "target": args.target,
        "model_name": best_name,
        "cv_results": cv_results
    }

    out_path = save_pipeline(best_pipeline, args.output_dir, filename=args.output_filename, metadata=metadata)
    print("Saved best pipeline to:", out_path)

    # Save CV reports
    output_dir = Path(args.output_dir)
    cv_report_path = output_dir / "cv_report.json"
    fold_metrics_path = output_dir / "fold_metrics.json"
    
    save_json(cv_results, cv_report_path)
    save_json(fold_metrics, fold_metrics_path)
    print(f"Saved cv_report.json and fold_metrics.json")

    # [NEW] Generate visualizations automatically
    try:
        print("\nGenerating model performance visualizations...")
        generate_all_plots(
            str(cv_report_path), 
            str(Path(args.output_dir) / "hyperopt"), 
            str(Path(args.output_dir) / "comparison_plots"),
            fold_json=str(fold_metrics_path)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Warning: Visualization failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../data/Cardiovascular_Disease_Dataset.csv")
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--id-column", type=str, default="patientid", help="Identifier column to drop if present")
    parser.add_argument("--numeric-hints", type=list, default=None, help="Optional list of numeric column name hints")
    parser.add_argument("--output-dir", type=str, default="../models")
    parser.add_argument("--output-filename", type=str, default="best_model_pipeline.joblib")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    main(args)
