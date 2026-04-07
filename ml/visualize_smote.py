# ml/visualize_smote.py
"""
Generates a side-by-side bar chart showing the class distribution of the target
variable before and after SMOTE balancing.

Usage:
  python -m ml.visualize_smote --data-path data/UCI_Heart_Disease_Combined.csv --output-dir models/comparison_plots
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from imblearn.over_sampling import SMOTE
from ml.preprocessing import infer_feature_groups, build_preprocessor

def main(args):
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if args.id_column and args.id_column in df.columns:
        df = df.drop(columns=[args.id_column])
    
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Preprocess because SMOTE cannot handle missing values or categorical strings out-of-the-box
    numeric_feats, categorical_feats = infer_feature_groups(list(X.columns), args.numeric_hints)
    preprocessor = build_preprocessor(numeric_feats, categorical_feats)
    X_processed = preprocessor.fit_transform(X)

    # Calculate before
    before_counts = y.value_counts().sort_index()

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_processed, y)
    
    # Calculate after
    after_counts = y_res.value_counts().sort_index()

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Custom colors
    colors_before = ["#E63946", "#457B9D"]
    colors_after = ["#2A9D8F", "#2A9D8F"]

    sns.barplot(x=before_counts.index, y=before_counts.values, ax=axes[0], palette=colors_before)
    axes[0].set_title("Before SMOTE (Imbalanced)", fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel(f"Target Class ({args.target})", fontsize=12)
    axes[0].set_ylabel("Number of Patients", fontsize=12)
    axes[0].set_ylim(0, max(after_counts) * 1.1)

    for i, count in enumerate(before_counts.values):
        axes[0].text(i, count + max(after_counts)*0.02, str(count), ha='center', fontsize=12, fontweight='bold')

    sns.barplot(x=after_counts.index, y=after_counts.values, ax=axes[1], palette=colors_after)
    axes[1].set_title("After SMOTE (Balanced)", fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel(f"Target Class ({args.target})", fontsize=12)
    axes[1].set_ylabel("Number of Patients", fontsize=12)
    axes[1].set_ylim(0, max(after_counts) * 1.1)

    for i, count in enumerate(after_counts.values):
        axes[1].text(i, count + max(after_counts)*0.02, str(count), ha='center', fontsize=12, fontweight='bold')

    plt.suptitle("Clinical Dataset Impact: Synthetic Minority Over-sampling Technique (SMOTE)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = out_dir / "smote_distribution_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved SMOTE artifact to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/Cardiovascular_Disease_Dataset.csv")
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--id-column", type=str, default="patientid")
    parser.add_argument("--numeric-hints", type=list, default=None)
    parser.add_argument("--output-dir", type=str, default="models/comparison_plots")
    args = parser.parse_args()
    main(args)
