# ml/preprocessing.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Tuple

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Build and return a ColumnTransformer preprocessing pipeline.
    Numeric -> median imputer + StandardScaler
    Categorical -> most_frequent imputer + OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor


def infer_feature_groups(df_columns: list, numeric_hints: list = None) -> Tuple[list, list]:
    """
    Given df columns and optional numeric_hints list (names assumed numeric),
    return (numeric_features, categorical_features).
    If numeric_hints is None, will guess by common names.
    """
    if numeric_hints is None:
        # common numeric column name hints (you can extend)
        numeric_hints = ['age', 'restingBP', 'restingbp', 'resting_blood_pressure',
                         'serumcholestrol', 'cholesterol', 'maxheartrate', 'oldpeak',
                         'triglyceride', 'fasting', 'sugar', 'glucose']

    numeric = []
    categorical = []
    for c in df_columns:
        if c.lower() in [h.lower() for h in numeric_hints]:
            numeric.append(c)
        else:
            # exclude target if present here; caller should remove target first
            categorical.append(c)

    # fallback: detect using substrings if nothing selected
    if len(numeric) == 0:
        for c in df_columns:
            if any(k in c.lower() for k in ['age', 'bp', 'pressure', 'rate', 'max', 'min', 'peak', 'chol', 'sugar']):
                numeric.append(c)
            else:
                categorical.append(c)

    return numeric, [c for c in categorical if c not in numeric]
