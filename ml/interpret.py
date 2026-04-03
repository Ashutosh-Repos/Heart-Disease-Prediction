# ml/interpret.py
"""
Generate SHAP explanations for a saved pipeline (preprocessor + model).

Saves:
  - models/explain/summary_plot.png        (summary beeswarm)
  - models/explain/bar_plot.png            (mean |SHAP| bar)
  - models/explain/shap_values.npy         (raw shap values for dataset or subset)
  - models/explain/shap_summary.csv       (per-feature mean abs shap)
  - models/explain/sample_<i>_waterfall.png (per-sample waterfall)
  - models/explain/sample_<i>_force.html    (per-sample force plot HTML)

Usage (from project root):
  python -m ml.interpret \
      --model-path models/best_model_pipeline.joblib \
      --data-path data/Cardiovascular_Disease_Dataset.csv \
      --target target \
      --output-dir models/explain \
      --max-background 200 \
      --n-samples 50

Notes:
 - Recommended to run this after training. For CatBoost / LightGBM / XGBoost / RandomForest, TreeExplainer is used (fast).
 - If model is not tree-based, KernelExplainer is used and is slow; reduce --max-background and --n-samples.
 - Requires: shap, matplotlib, pandas, numpy
    pip install shap matplotlib pandas numpy
"""

import argparse
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
except Exception as e:
    raise RuntimeError("This script requires the 'shap' package. Install with: pip install shap") from e

# ---- Helper: derive post-transform feature names ----
def get_transformed_feature_names(preprocessor, original_feature_names):
    """
    Try to reconstruct output feature names after ColumnTransformer.
    Returns a list of names (length = number of output features).
    If it fails, returns fallback generic names.
    """
    # If not a ColumnTransformer, return original features
    from sklearn.compose import ColumnTransformer
    if preprocessor is None:
        return original_feature_names

    try:
        if isinstance(preprocessor, ColumnTransformer):
            feature_names_out = []
            # iterate through transformers_
            for name, trans, cols in preprocessor.transformers_:
                if name == 'remainder':
                    # if remainder is 'drop' skip
                    if trans == 'drop':
                        continue
                    # if passthrough, try to extend
                    if trans == 'passthrough':
                        if isinstance(cols, (list, tuple)):
                            feature_names_out.extend(list(cols))
                        else:
                            feature_names_out.append(cols)
                        continue
                # handle pipeline wrappers
                transformer = trans
                if hasattr(trans, "named_steps"):
                    # pipeline -> get last transformer
                    transformer = trans.named_steps[list(trans.named_steps.keys())[-1]]
                if hasattr(transformer, "get_feature_names_out"):
                    try:
                        # transformer.get_feature_names_out(cols) may fail for some older versions
                        names = transformer.get_feature_names_out(cols)
                    except Exception:
                        try:
                            names = transformer.get_feature_names_out()
                        except Exception:
                            names = None
                    if names is not None:
                        feature_names_out.extend(list(names))
                        continue
                # fallback: cols may be a list of column names
                if isinstance(cols, (list, tuple)):
                    feature_names_out.extend(list(cols))
                else:
                    feature_names_out.append(cols)
            # final sanity check
            return feature_names_out
        else:
            # preprocessor isn't ColumnTransformer
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    return list(preprocessor.get_feature_names_out(original_feature_names))
                except Exception:
                    return original_feature_names
            return original_feature_names
    except Exception:
        # fallback
        return original_feature_names

# ---- Main ----
def main(args):
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("Loading dataset:", data_path)
    df = pd.read_csv(data_path)
    if args.id_column and args.id_column in df.columns:
        df = df.drop(columns=[args.id_column])
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset")
    X = df.drop(columns=[args.target])
    y = df[args.target]

    print("Loading pipeline:", model_path)
    saved = joblib.load(model_path)
    if isinstance(saved, dict) and "pipeline" in saved:
        pipeline = saved["pipeline"]
        metadata = saved.get("metadata", {})
    else:
        pipeline = saved
        metadata = {}

    # Extract preprocessor and classifier from pipeline if possible
    preprocessor = None
    model = None
    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessor", None)
        model = pipeline.named_steps.get("clf", None)
    else:
        # pipeline may be a raw estimator (no preprocessor)
        model = pipeline

    # Determine feature names after preprocessing
    original_feature_names = list(X.columns)
    transformed_feature_names = get_transformed_feature_names(preprocessor, original_feature_names)

    # Prepare background (for KernelExplainer) and X for explanation
    # We'll create X_trans (the preprocessor output) only when needed or for matching shapes
    # For TreeExplainer we can pass model and raw X (explainer will handle pipeline if we wrap appropriately).
    # However our pipeline is preprocessor -> model; shap's TreeExplainer expects the model and transformed data.
    # We'll transform X to array using preprocessor if present.
    if preprocessor is not None:
        try:
            X_trans = preprocessor.transform(X)
        except Exception as e:
            # fallback: try fit_transform (rare)
            X_trans = preprocessor.fit_transform(X)
    else:
        X_trans = X.values

    # Convert to numpy 2D
    X_trans = np.asarray(X_trans)

    # Decide explainer
    is_tree_model = False
    tree_types = ("CatBoostClassifier", "LGBMClassifier", "XGBClassifier", "XGBRegressor", "LGBMRegressor", "RandomForestClassifier", "DecisionTreeClassifier", "GradientBoostingClassifier")
    if model is not None:
        mname = type(model).__name__
        if any(t in mname for t in ["CatBoost", "LGBM", "XGB", "RandomForest", "DecisionTree", "GradientBoosting"]):
            is_tree_model = True

    print("Model type:", type(model).__name__, "| Tree-like:", is_tree_model)
    print("Original features:", len(original_feature_names), "Transformed features:", X_trans.shape[1])

    # Build explainer
    explainer = None
    if is_tree_model:
        print("Using TreeExplainer (fast) ...")
        try:
            # For tree models, pass the fitted model (the inner estimator) and feature_perturbation is 'interventional' by default in newer shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            # shap_values format may be list (for multiclass). For binary classifier, shap_values might be [neg, pos] or a single array.
            # Normalize to 2D array of SHAP values for the positive class if necessary
            if isinstance(shap_values, list):
                # try to pick the class index 1 if available
                if len(shap_values) > 1:
                    # many tree models return list-of-arrays (classes)
                    shap_vals_for_pos = np.array(shap_values[1])
                else:
                    shap_vals_for_pos = np.array(shap_values[0])
                shap_vals = shap_vals_for_pos
            else:
                shap_vals = np.array(shap_values)
        except Exception as e:
            print("TreeExplainer failed, falling back to KernelExplainer:", e)
            is_tree_model = False

    if not is_tree_model:
        # KernelExplainer fallback (slow) - use a small background dataset
        print("Using KernelExplainer (slow) ...")
        bg_size = int(min(args.max_background, X_trans.shape[0]))
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(X_trans.shape[0], size=bg_size, replace=False)
        background = X_trans[idx]
        # KernelExplainer expects a prediction function that returns model probabilities for the positive class
        def predict_fn(x):
            # x will be in transformed space; we need to inverse-transform? Easier: re-run pipeline on raw input if x is not transformed.
            # BUT RandomizedSearch uses preprocessor separately; KernelExplainer will be given transformed arrays (we created background from X_trans).
            # To be safe, we will wrap a function that accepts transformed arrays and calls model.predict_proba.
            return model.predict_proba(x)[:, 1]
        # If model.predict_proba expects original features, we need to pass inverse; but in our design model was trained on transformed arrays (pipeline).
        # For safety, we'll check if model is pipeline-like; but here model is the inner estimator so it expects transformed arrays.
        explainer = shap.KernelExplainer(predict_fn, background)
        # select sample subset to explain if dataset is large
        explain_idx = np.arange(min(args.n_samples, X_trans.shape[0]))
        shap_vals = explainer.shap_values(X_trans[explain_idx], nsamples=args.nsamples_per_call)  # may be slow
        # shap_vals will be shape (n_samples, n_features) or list
        # We'll expand later below
        # For KernelExplainer we limit to subset
        X_trans = X_trans[explain_idx]
        y = y.iloc[explain_idx]

    # At this point shap_vals should be a 2D array (n_samples, n_features)
    shap_vals = np.array(shap_vals)
    # If shap_vals is 3D (e.g., (n_classes, n_samples, n_features)), try to pick positive class
    if shap_vals.ndim == 3:
        # pick class 1 if exists
        if shap_vals.shape[0] > 1:
            shap_vals = shap_vals[1]
        else:
            shap_vals = shap_vals[0]

    # Save raw shap values
    np.save(out_dir / "shap_values.npy", shap_vals)
    print("Saved raw SHAP values to", out_dir / "shap_values.npy")

    # Compute mean |shap| per feature
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    # Align feature names (length must match)
    if len(transformed_feature_names) != shap_vals.shape[1]:
        # fallback to generic names
        transformed_feature_names = [f"f_{i}" for i in range(shap_vals.shape[1])]
    fi_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    fi_df.to_csv(out_dir / "shap_summary.csv", index=False)
    print("Saved SHAP summary to", out_dir / "shap_summary.csv")

    # Plot global summary (beeswarm)
    try:
        plt.figure()
        # shap.summary_plot expects shap_values and features (can be array or dataframe)
        shap.summary_plot(shap_vals, features=X_trans if isinstance(X_trans, np.ndarray) else np.array(X_trans), feature_names=transformed_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "summary_plot.png", bbox_inches="tight")
        plt.close()
        print("Saved global summary_plot.png")
    except Exception as e:
        print("Could not create summary plot:", e)

    # Bar plot of mean abs shap
    try:
        topk = min(len(fi_df), args.topk_features)
        plt.figure(figsize=(8, max(4, topk * 0.4)))
        shap.plots.bar(shap.Explanation(values=shap_vals, base_values=None, data=X_trans, feature_names=transformed_feature_names), show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "bar_plot.png", bbox_inches="tight")
        plt.close()
        print("Saved bar_plot.png")
    except Exception as e:
        # fallback simple matplotlib bar
        try:
            plt.figure(figsize=(8, max(4, topk * 0.4)))
            plt.barh(fi_df["feature"].head(topk)[::-1], fi_df["mean_abs_shap"].head(topk)[::-1])
            plt.xlabel("mean |SHAP value|")
            plt.tight_layout()
            plt.savefig(out_dir / "bar_plot.png", bbox_inches="tight")
            plt.close()
            print("Saved bar_plot.png (fallback)")
        except Exception as e2:
            print("Could not create bar plot:", e2)

    # Per-sample explanations: choose indices
    if args.sample_indices:
        sample_indices = args.sample_indices
    else:
        # pick top uncertain samples by predicted probability near 0.5 (if possible)
        try:
            # need model probability on transformed space
            probs = model.predict_proba(X_trans)[:, 1]
            uncertain_idx = np.argsort(np.abs(probs - 0.5))[: args.n_samples]
            sample_indices = list(map(int, uncertain_idx))
        except Exception:
            sample_indices = list(range(min(args.n_samples, shap_vals.shape[0])))

    # Ensure indices are within range
    sample_indices = [int(i) for i in sample_indices if 0 <= int(i) < shap_vals.shape[0]]

    for i in sample_indices:
        sv = shap_vals[i]
        # Compute a sensible base value for the explanation
        base_value = None
        try:
            if hasattr(explainer, "expected_value"):
                base_value = explainer.expected_value
                # if it's an array (multiclass), pick positive class if present
                try:
                    import numpy as _np
                    if _np.ndim(base_value) > 0 and len(_np.atleast_1d(base_value)) > 1:
                        # prefer class index 1 (positive) when binary
                        base_value = _np.atleast_1d(base_value)[1]
                except Exception:
                    pass
        except Exception:
            base_value = None

        # Ensure base_value is scalar or None
        try:
            import numpy as _np
            if isinstance(base_value, (_np.ndarray, list)) and len(base_value) == 1:
                base_value = float(base_value[0])
        except Exception:
            pass

        # Waterfall plot (per-sample)
        try:
            # Build a shap.Explanation with base_values set (so waterfall plots can access it)
            expl_obj = shap.Explanation(values=sv, base_values=base_value, data=X_trans[i], feature_names=transformed_feature_names)
            shap.plots.waterfall(expl_obj, show=False)
            p = out_dir / f"sample_{i}_waterfall.png"
            plt.tight_layout()
            plt.savefig(p, bbox_inches="tight")
            plt.close()
            print("Saved", p)
        except Exception as e:
            print(f"Could not create waterfall for sample {i}:", e)

        # Force plot saved as HTML (works across shap versions)
        try:
            # Determine a base value scalar for force_plot
            bv = base_value
            # shap.force_plot expects a scalar base value for binary; try to coerce
            try:
                import numpy as _np
                if isinstance(bv, (_np.ndarray, list)) and len(_np.atleast_1d(bv)) > 1:
                    bv = float(_np.atleast_1d(bv)[1])  # prefer positive class
                elif isinstance(bv, (_np.ndarray, list)):
                    bv = float(_np.atleast_1d(bv)[0])
            except Exception:
                pass

            # build force plot and save interactive HTML
            f_html = out_dir / f"sample_{i}_force.html"
            # For TreeExplainer the base value is typically explainer.expected_value
            force_vis = shap.force_plot(bv, sv, X_trans[i], feature_names=transformed_feature_names, matplotlib=False)
            try:
                shap.save_html(str(f_html), force_vis)
            except Exception:
                # fallback: write repr
                with open(f_html, "w") as fh:
                    fh.write(str(force_vis))
            print("Saved", f_html)
        except Exception as e:
            print(f"Could not create force plot for sample {i}:", e)
        
    print("Interpretation finished. Artifacts in:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/best_model_pipeline.joblib")
    parser.add_argument("--data-path", type=str, default="data/UCI_Heart_Disease_Combined.csv")
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--id-column", type=str, default="patientid")
    parser.add_argument("--output-dir", type=str, default="models/explain")
    parser.add_argument("--max-background", type=int, default=500,
                        help="Max background samples for KernelExplainer (lower -> faster but less accurate)")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of samples to compute per-sample explanations for (if not provided, top uncertain samples chosen)")
    parser.add_argument("--nsamples_per_call", type=int, default=100,
                        help="KernelExplainer nsamples (if using KernelExplainer)")
    parser.add_argument("--topk-features", type=int, default=14)
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None,
                        help="Specific sample indices (0-based) to explain. If omitted, small set of uncertain samples will be selected.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)
