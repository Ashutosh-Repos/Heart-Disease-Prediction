import pandas as pd
from pathlib import Path
import argparse
from ml.schema import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_METADATA

def standardize_heart_data(input_path: str, output_path: str, auto_offset: bool = True):
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Mapping UCI names to pipeline names
    rename_map = {
        'sex': 'gender',
        'cp': 'chestpain',
        'trestbps': 'restingBP',
        'chol': 'serumcholestrol',
        'fbs': 'fastingbloodsugar',
        'restecg': 'restingrelectro',
        'thalach': 'maxheartrate',
        'exang': 'exerciseangia',
        'ca': 'noofmajorvessels'
    }
    
    print("Renaming columns...")
    df = df.rename(columns=rename_map)
    
    # 🚨 Clinical Alignment Engine: Automated +1 Offset for 0-indexed data
    if auto_offset:
        print("Checking for zero-indexed clinical categories...")
        for feat in ['chestpain', 'slope']:
            if feat in df.columns:
                # If the minimum value is 0, we shift all values by +1 to match UCI (1-4, 1-3)
                if df[feat].min() == 0:
                    print(f"  -> Offsetting '{feat}' to 1-indexed (0-indexed detected)")
                    df[feat] = df[feat] + 1
        
    # handle gender string-to-numeric if needed
    if 'gender' in df.columns and df['gender'].dtype == object:
        print("Mapping gender strings to numeric (M:1, F:0)...")
        df['gender'] = df['gender'].map(FEATURE_METADATA['gender']['mapping'])

    # Drop thal if it exists
    if 'thal' in df.columns:
        print("Dropping 'thal' column...")
        df = df.drop(columns=['thal'])
        
    # Add dummy patientid if it doesn't exist
    if 'patientid' not in df.columns:
        print("Adding dummy 'patientid' column...")
        df.insert(0, 'patientid', range(10000, 10000 + len(df)))
        
    print(f"Saving standardized data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Standardization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize heart datasets to project pipeline format.")
    parser.add_argument("--input", type=str, default="data/heart.csv", help="Path to raw heart.csv")
    parser.add_argument("--output", type=str, default="data/heart_standardized.csv", help="Path to save output")
    parser.add_argument("--no-offset", action="store_false", dest="auto_offset", help="Disable auto +1 clinical offset")
    
    args = parser.parse_args()
    standardize_heart_data(args.input, args.output, auto_offset=args.auto_offset)
