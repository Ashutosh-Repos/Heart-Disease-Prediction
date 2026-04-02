# ml/combine_uci_data.py
import pandas as pd
import numpy as np
from pathlib import Path

def combine_uci_data(base_dir: str, output_path: str):
    base_path = Path(base_dir)
    files = [
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data"
    ]
    
    # Official UCI 14-attribute names
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    all_dfs = []
    
    print("Starting UCI data combination...")
    for f in files:
        file_path = base_path / f
        if not file_path.exists():
            print(f"Warning: {f} not found. Skipping.")
            continue
            
        print(f"Processing {f}...")
        # Read file, '?' values are treated as NaN
        df = pd.read_csv(file_path, header=None, names=columns, na_values='?')
        all_dfs.append(df)
        
    if not all_dfs:
        print("No files found to combine.")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined {len(combined_df)} total records.")
    
    # --- Transformation Logic to match project pipeline ---
    
    # 1. Rename columns to match project schema
    rename_map = {
        'sex': 'gender',
        'cp': 'chestpain',
        'trestbps': 'restingBP',
        'chol': 'serumcholestrol',
        'fbs': 'fastingbloodsugar',
        'restecg': 'restingrelectro',
        'thalach': 'maxheartrate',
        'exang': 'exerciseangia',
        'num': 'target'
    }
    combined_df = combined_df.rename(columns=rename_map)
    combined_df['noofmajorvessels'] = combined_df['ca']
    
    # 2. Convert Target to Binary (0 = Normal, 1-4 = Disease)
    print("Converting target labels to binary (0 and 1)...")
    combined_df['target'] = combined_df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # 3. Drop non-semantic columns (like 'ca' after rename and 'thal')
    # and any other columns that don't exist in the project schema
    # Project schema features: age, gender, chestpain, restingBP, serumcholestrol, 
    # fastingbloodsugar, restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels
    cols_to_keep = [
        'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 
        'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 
        'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels', 'target'
    ]
    
    # Final cleanup
    combined_df = combined_df[cols_to_keep]
    
    # 4. Add dummy patientid for structural parity
    combined_df.insert(0, 'patientid', range(20000, 20000 + len(combined_df)))
    
    print(f"Final shape: {combined_df.shape}")
    print(f"Saving to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    print("Combination and standardization complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/heart+disease 2")
    parser.add_argument("--output", type=str, default="data/UCI_Heart_Disease_Combined.csv")
    args = parser.parse_args()
    
    combine_uci_data(args.dir, args.output)
