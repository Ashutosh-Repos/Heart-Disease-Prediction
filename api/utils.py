# api/utils.py
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
# api/utils.py (append)
from collections import defaultdict
import re
import traceback
import pandas as pd

def load_pipeline_and_metadata(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found at {path}")
    data = joblib.load(path)
    if isinstance(data, dict) and "pipeline" in data:
        pipeline = data["pipeline"]
        metadata = data.get("metadata", {})
    else:
        pipeline = data
        metadata = {}

    # Try to derive transformed feature names and store in metadata if possible
    try:
        # 1. Prefer schema from centralized registry
        from ml.schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        numeric_feats = NUMERIC_FEATURES
        categorical_feats = CATEGORICAL_FEATURES
        
        # 2. Allow metadata to override if present in the model file
        numeric_feats = metadata.get("numeric_features", numeric_feats)
        categorical_feats = metadata.get("categorical_features", categorical_feats)

        # 3. Build/Configure preprocessor based on these groups
        from ml.preprocessing import build_preprocessor
        preprocessor = build_preprocessor(numeric_feats, categorical_feats)
        metadata["transformed_feature_names"] = list(preprocessor.get_feature_names_out(metadata.get("features", [])))
    except Exception:
        # best-effort: leave metadata as-is
        pass
    return pipeline, metadata

def normalize_input_dict(d: dict) -> dict:
    # handle gender mapping
    out = dict(d)
    if 'gender' in out:
        g = out['gender']
        if isinstance(g, str):
            g_l = g.strip().lower()
            if g_l in ['m', 'male', '1', 'true', 'yes']:
                out['gender'] = 1
            elif g_l in ['f', 'female', '0', 'false', 'no']:
                out['gender'] = 0
            else:
                try:
                    out['gender'] = int(g)
                except Exception:
                    out['gender'] = g
    return out

def build_shap_explainer_if_possible(pipeline, metadata=None) -> Tuple[Optional[object], str]:
    """
    If the inner estimator is tree-based create a shap TreeExplainer and return it.
    If SVM, create a KernelExplainer using injected background_data from metadata.
    Returns (explainer_or_None, mode_string)
    """
    try:
        import shap
    except Exception:
        return None, "no-shap"

    model = None
    if hasattr(pipeline, "named_steps"):
        model = pipeline.named_steps.get("clf", None)
        preprocessor = pipeline.named_steps.get("preprocessor", None)
    else:
        model = pipeline
        preprocessor = None

    if model is None:
        return None, "no-model"

    model_name = type(model).__name__
    if any(k in model_name for k in ["CatBoost", "LGBM", "XGB", "RandomForest", "DecisionTree", "GradientBoosting"]):
        try:
            explainer = shap.TreeExplainer(model)
            return explainer, "tree"
        except Exception:
            return None, "shap-error"

    if "SVC" in model_name or "KNeighbors" in model_name:
        try:
            if metadata and "background_data" in metadata and metadata["background_data"]:
                import pandas as pd
                bg_df = pd.DataFrame(metadata["background_data"])
                if preprocessor:
                    # ensure we only take the training features to avoid transform errors
                    bg_df = bg_df[metadata.get("features", bg_df.columns.tolist())]
                    bg_trans = preprocessor.transform(bg_df)
                else:
                    bg_trans = bg_df.values
                # We want shap.KernelExplainer to call predict_proba
                explainer = shap.KernelExplainer(model.predict_proba, bg_trans)
                return explainer, "kernel"
            else:
                return None, "no-background-data"
        except Exception as e:
            traceback.print_exc()
            return None, "shap-kernel-error"

    return None, "unsupported-model-type"

def aggregate_contributions(contrib_dict, top_k=8):
    """
    Aggregate transformed contribution names to original features.
    Input: contrib_dict: {'num__age': 0.05, 'cat__chestpain_2': 0.7, ...}
    Handles 'missing_indicator_' prefix in both num and cat transformers.
    """
    agg = defaultdict(float)
    for k, v in contrib_dict.items():
        # 1. Strip transformer prefix (num__ or cat__)
        name = re.sub(r'^(num__|cat__)', '', k)
        
        # 2. Handle OHE category tails (e.g. '_Asymptomatic' from 'cat__chestpain_Asymptomatic')
        # We only strip the tail if it's NOT a missing_indicator flag
        if "missing_indicator_" not in name:
            # Matches any tail with a single underscore (OneHotEncoder default)
            m = re.match(r"(.*)_[^_]+$", name)
            if m:
                name = m.group(1)

        # 3. Strip missing indicator prefix and aggregate
        orig = name.replace("missing_indicator_", "")
        agg[orig] += float(v)

    sorted_list = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = sorted_list[:top_k]
    return dict(agg), top
