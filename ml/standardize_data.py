# ml/standardize_data.py
import pandas as pd
from pathlib import Path
import argparse

def standardize_heart_data(input_path: str, output_path: str):
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
    
    # Drop thal if it exists
    if 'thal' in df.columns:
        print("Dropping 'thal' column...")
        df = df.drop(columns=['thal'])
        
    # Add dummy patientid if it doesn't exist (to match standard CSV structure)
    if 'patientid' not in df.columns:
        print("Adding dummy 'patientid' column...")
        df.insert(0, 'patientid', range(10000, 10000 + len(df)))
        
    print(f"Saving standardized data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Standardization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize standard UCI heart.csv to project pipeline format.")
    parser.add_argument("--input", type=str, default="data/heart.csv", help="Path to raw heart.csv")
    parser.add_argument("--output", type=str, default="data/heart_standardized.csv", help="Path to save output")
    
    args = parser.parse_args()
    standardize_heart_data(args.input, args.output)
