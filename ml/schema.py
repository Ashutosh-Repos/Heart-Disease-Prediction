# ml/schema.py
"""
Centralized Clinical Registry for the Heart Disease Prediction Project.
This serves as the single source of truth for feature groupings and clinical mappings.
"""

# ---------------------------
# 🔬 Clinical Feature Registry
# ---------------------------

# 5 continuous numeric clinical measurements
NUMERIC_FEATURES = [
    'age', 
    'restingBP', 
    'serumcholestrol', 
    'maxheartrate', 
    'oldpeak'
]

# 7 categorical/ordinal clinical observations
CATEGORICAL_FEATURES = [
    'gender', 
    'chestpain', 
    'fastingbloodsugar', 
    'restingrelectro', 
    'exerciseangia', 
    'slope', 
    'noofmajorvessels'
]

# 1 target variable (angiographic disease status)
TARGET_COLUMN = 'target'
ID_COLUMN = 'patientid'

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------
# 📖 Clinical Metadata Registry
# ---------------------------
# Expected medical ranges/mappings for validation & standardizing
FEATURE_METADATA = {
    'chestpain': {
        'standard': '1-indexed (1:Typical, 2:Atypical, 3:Non-anginal, 4:Asymptomatic)',
        'offset_required_if_zero_indexed': True
    },
    'slope': {
        'standard': '1-indexed (1:Upsloping, 2:Flat, 3:Downsloping)',
        'offset_required_if_zero_indexed': True
    },
    'gender': {
        'mapping': {'M': 1, 'F': 0, 'Male': 1, 'Female': 0}
    }
}
