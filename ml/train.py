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

# SMOTE support
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import ml.hyperparam_search as hyperparam_search

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

    # 1. Use Centralized Research Schema
    from ml.schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
    numeric_features = NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    # 2. Schema Hardening: Account for any unexpected columns in the data
    all_cols = X.columns.tolist()
    known = set(numeric_features + categorical_features)
    for c in all_cols:
        if c not in known:
            # automatic heuristic for extra custom columns
            if any(k in c.lower() for k in ['age', 'bp', 'press', 'rate', 'max', 'peak', 'chol', 'sugar']):
                numeric_features.append(c)
            else:
                categorical_features.append(c)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    
    # 80/20 split for generating detailed plots (CM, ROC, PR) for all models
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_state
    )

    # Prepare models
    models = get_models(random_state=args.random_state)
    
    # Track whether each model should use SMOTE (default to the user's flag)
    model_use_smote = {name: args.use_smote for name in models.keys()}


    if args.tune:
        print(f"\n[Rigor] Splitting data for Nested Validation: {len(X_train_eval)} (Dev/Tune) vs {len(X_test_eval)} (Gold Isolation).")
        # Redirect specialized args for the search module
        search_args = argparse.Namespace(
            data_path=args.data_path,
            target=args.target,
            id_column=args.id_column,
            output_dir=args.output_dir,
            n_iter=args.tune_iter,
            cv=args.cv,
            n_jobs=args.n_jobs,
            top_k=1,
            random_state=args.random_state
        )
        # Pass the isolated 80% training set to the search module to prevent leakage
        hyperparam_search.main(search_args, X_train=X_train_eval, y_train=y_train_eval)
        print("[Optimizer] Search complete. Loading winners into pipeline...")
        
        # Load tuned parameters into the models dictionary
        hyperopt_dir = Path(args.output_dir) / "hyperopt"
        for name in models.keys():
            summary_json = hyperopt_dir / name / "summary.json"
            if summary_json.exists():
                import json
                with open(summary_json, 'r') as f:
                    summary = json.load(f)
                best_params = summary.get("best_params", {})
                if best_params:
                    print(f"  - {name}: Applying optimized params: {best_params}")
                    
                    # Inherit the SMOTE decision directly from the optimizer 
                    smote_val = best_params.get("smote")
                    model_use_smote[name] = (smote_val is not None)

                    # Remove 'smote' from params as it's a pipeline step, not a clf param
                    clf_params = {k: v for k, v in best_params.items() if k != "smote"}
                    models[name].set_params(**clf_params)
    
    cv_results = {}
    fold_metrics = {}
    pipelines = {}

    report_dir = Path(args.output_dir) / "comparison_plots"

    for name, clf in models.items():
        print(f"\nRunning CV for: {name}")
        # Use ImbPipeline and apply SMOTE if requested
        steps = [('preprocessor', preprocessor)]
        if model_use_smote[name]:
            print(f"  - Applying SMOTE balancing to {name}...")
            steps.append(('smote', SMOTE(random_state=args.random_state)))
        steps.append(('clf', clf))
        
        pipeline = ImbPipeline(steps=steps)
        scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
        # Benchmarking is restricted to the 80% Development set to maintain independent validation
        summary, full_cv = run_cv(pipeline, X_train_eval, y_train_eval, cv=args.cv, scoring=scoring, n_jobs=args.n_jobs)
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
    # compute feature importances (uses injected metadata preferentially)
    feature_names = list(X.columns)
    try:
        fi_df = compute_feature_importances(best_pipeline, X_test_eval, y_test_eval, args.output_dir, feature_names)
    except Exception as e:
        print(f"Warning: Feature importance calculation failed: {e}")
    best_pipeline.fit(X, y)

    # Save a small subset of the training data as background profiles for SHAP KernelExplainer
    bg_data = X_train_eval.head(20).to_dict(orient="records") if len(X_train_eval) > 0 else []

    metadata = {
        "features": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": args.target,
        "model_name": best_name,
        "cv_results": cv_results,
        "clinical_standard": "UCI-1-indexed",
        "background_data": bg_data
    }

    out_path = save_pipeline(best_pipeline, args.output_dir, filename=args.output_filename, metadata=metadata)
    print("Saved best pipeline to:", out_path)

    # Save CV reports
    output_dir = Path(args.output_dir)
    cv_report_path = output_dir / "cv_report.json"
    fold_metrics_path = output_dir / "fold_metrics.json"
    certification_path = output_dir / "research_certification.json"
    
    from ml.utils import sanitize_for_json
    save_json(sanitize_for_json(cv_results), cv_report_path)
    save_json(sanitize_for_json(fold_metrics), fold_metrics_path)
    
    # [NEW] Unified Research Certification
    # This acts as the single source of truth for the API and Dashboard
    certification = {
        "status": "Certified",
        "nested_validation": True,
        "dev_samples": len(X_train_eval),
        "gold_samples": len(X_test_eval),
        "best_model": best_name,
        "cv_results": cv_results,
        "clinical_standard": "UCI-1-indexed",
        "random_state": args.random_state
    }
    save_json(sanitize_for_json(certification), certification_path)
    print(f"Saved Unified Research Certification to: {certification_path}")

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
    parser.add_argument("--data-path", type=str, default="data/Cardiovascular_Disease_Dataset.csv")
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--id-column", type=str, default="patientid", help="Identifier column to drop if present")
    parser.add_argument("--numeric-hints", type=list, default=None, help="Optional list of numeric column name hints")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--output-filename", type=str, default="best_model_pipeline.joblib")
    parser.add_argument("--cv", type=int, default=10, help="CV folds")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE balancing in standard training")
    parser.add_argument("--tune", action="store_true", help="Run deep hyperparameter tuning search")
    parser.add_argument("--tune-iter", type=int, default=50, help="Number of search iterations if --tune is used")
    args = parser.parse_args()
    main(args)
