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
        pre = pipeline.named_steps.get("preprocessor", None)
        if pre is not None:
            try:
                # scikit-learn >=1.0 supports get_feature_names_out on ColumnTransformer
                transformed = pre.get_feature_names_out()
                metadata["transformed_feature_names"] = list(transformed)
            except Exception:
                # best-effort: leave metadata as-is
                pass
    except Exception:
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

def build_shap_explainer_if_possible(pipeline) -> Tuple[Optional[object], str]:
    """
    If the inner estimator is tree-based create a shap TreeExplainer and return it.
    Returns (explainer_or_None, mode_string)
    """
    try:
        import shap
    except Exception:
        return None, "no-shap"

    model = None
    if hasattr(pipeline, "named_steps"):
        model = pipeline.named_steps.get("clf", None)
    else:
        model = pipeline
    if model is None:
        return None, "no-model"

    model_name = type(model).__name__
    if any(k in model_name for k in ["CatBoost", "LGBM", "XGB", "RandomForest", "DecisionTree", "GradientBoosting"]):
        try:
            explainer = shap.TreeExplainer(model)
            return explainer, "tree"
        except Exception:
            return None, "shap-error"
    return None, "not-tree"

def aggregate_contributions(contrib_dict, top_k=8):
    """
    Aggregate transformed contribution names to original features.
    Input: contrib_dict: {'num__age': 0.05, 'cat__chestpain_2': 0.7, ...}
    Returns:
      - agg_dict: {orig_feature: total_contribution}
      - top_list: list of (orig_feature, contribution) sorted by abs desc (length top_k)
    """
    agg = defaultdict(float)
    for k, v in contrib_dict.items():
        if k.startswith("num__"):
            name = k.replace("num__", "")
            orig = name.replace("missing_indicator_", "")
        elif k.startswith("cat__"):
            m = re.match(r"cat__([^_]+)_.*", k)
            name = m.group(1) if m else k
            orig = name.replace("missing_indicator_", "")
        else:
            orig = k.replace("missing_indicator_", "")
        agg[orig] += float(v)
    sorted_list = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = sorted_list[:top_k]
    return dict(agg), top
