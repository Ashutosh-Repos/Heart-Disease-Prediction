# ml/visualize_metrics.py
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Set aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150

def plot_confusion_matrix(cm, labels, outpath: Path, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_roc(y_true, y_score, outpath: Path, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    ax.plot([0,1], [0,1], linestyle='--', color='grey')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_pr(y_true, y_score, outpath: Path, title: str = "Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label='Precision-Recall curve', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def generate_model_detail_plots(pipeline, X_test, y_test, model_name: str, output_dir: Path):
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    preds = pipeline.predict(X_test)
    if hasattr(pipeline, 'predict_proba'):
        probs = pipeline.predict_proba(X_test)[:, 1]
    else:
        # fallback to decision_function
        if hasattr(pipeline, 'decision_function'):
            scores = pipeline.decision_function(X_test)
            mn, mx = scores.min(), scores.max()
            probs = (scores - mn) / (mx - mn) if mx > mn else np.zeros_like(scores)
        else:
            probs = np.zeros_like(preds, dtype=float)

    cm = confusion_matrix(y_test, preds)
    labels = sorted([str(x) for x in np.unique(y_test)])
    
    plot_confusion_matrix(cm, labels, model_dir / "confusion_matrix.png", title=f"{model_name.upper()} - Confusion Matrix")
    if len(np.unique(y_test)) > 1:
        plot_roc(y_test, probs, model_dir / "roc_curve.png", title=f"{model_name.upper()} - ROC Curve")
        plot_pr(y_test, probs, model_dir / "pr_curve.png", title=f"{model_name.upper()} - PR Curve")
    print(f"Generated detailed plots for {model_name} in {model_dir}")

def load_cv_report(json_path: Path) -> pd.DataFrame:
    if not json_path.exists():
        print(f"Warning: {json_path} not found.")
        return pd.DataFrame()
    with open(json_path, 'r') as f:
        data = json.load(f)
    # data is { "model_name": { "metric": val, ... }, ... }
    df = pd.DataFrame(data).T
    df.index.name = 'model'
    return df.reset_index()

def load_hyperopt_data(hyperopt_dir: Path) -> pd.DataFrame:
    if not hyperopt_dir.exists():
        print(f"Warning: {hyperopt_dir} not found.")
        return pd.DataFrame()
    
    all_results = []
    for model_dir in hyperopt_dir.iterdir():
        if model_dir.is_dir():
            json_file = model_dir / "top_k_models.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for entry in data:
                    all_results.append({
                        'algorithm': model_dir.name,
                        'rank': entry.get('rank'),
                        'mean_score': entry.get('mean_test_score'),
                        'std_score': entry.get('std_test_score', 0)
                    })
    return pd.DataFrame(all_results)

def plot_metric_comparison(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    
    metrics = [m for m in ['roc_auc', 'accuracy', 'f1', 'precision', 'recall'] if m in df.columns]
    df_melt = df.melt(id_vars='model', value_vars=metrics, var_name='metric', value_name='score')
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_melt, x='metric', y='score', hue='model', palette='viridis')
    plt.title('Model Performance Comparison across Metrics', fontsize=16, fontweight='bold')
    plt.ylim(max(0, df_melt['score'].min() - 0.05), min(1.0, df_melt['score'].max() + 0.05))
    plt.ylabel('Score')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison_bar.png", bbox_inches='tight')
    plt.close()
    print(f"Saved metric_comparison_bar.png to {output_dir}")

def plot_metric_heatmap(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    
    df_plot = df.set_index('model')
    metrics = [m for m in ['roc_auc', 'accuracy', 'f1', 'precision', 'recall'] if m in df_plot.columns]
    df_plot = df_plot[metrics]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_plot, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Score'})
    plt.title('Model Metric Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "metric_heatmap.png", bbox_inches='tight')
    plt.close()
    print(f"Saved metric_heatmap.png to {output_dir}")

def plot_hyperopt_results(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    
    # Plot 1: Rank vs Score
    plt.figure(figsize=(12, 6))
    # Filter to ranks that make sense
    df_top = df[df['rank'] <= 5]
    sns.lineplot(data=df_top, x='rank', y='mean_score', hue='algorithm', marker='o', linewidth=2)
    plt.xticks([1, 2, 3, 4, 5])
    plt.title('Hyperparameter Tuning: Performance by Rank', fontsize=16, fontweight='bold')
    plt.ylabel('Mean ROC-AUC Score')
    plt.xlabel('Configuration Rank (1 = Best)')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(output_dir / "hyperopt_rank_vs_score.png", bbox_inches='tight')
    plt.close()
    
    # Plot 2: Best per algorithm (with std error bars)
    best_df = df[df['rank'] == 1]
    plt.figure(figsize=(10, 6))
    # manual error bars with sns barplot
    ax = sns.barplot(data=best_df, x='algorithm', y='mean_score', palette='magma')
    
    # Add error bars
    for i, p in enumerate(ax.patches):
        algorithm = best_df.iloc[i]['algorithm']
        mean = best_df.iloc[i]['mean_score']
        std = best_df.iloc[i]['std_score']
        plt.errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)
        
    plt.title('Best Configurations per Algorithm (with stability std)', fontsize=16, fontweight='bold')
    plt.ylim(max(0, best_df['mean_score'].min() - 0.05), min(1.0, best_df['mean_score'].max() + 0.02))
    plt.ylabel('Mean Score (Rank 1)')
    plt.tight_layout()
    plt.savefig(output_dir / "hyperopt_best_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"Saved Hyperopt plots to {output_dir}")

def load_fold_metrics(json_path: Path) -> pd.DataFrame:
    if not json_path.exists():
        return pd.DataFrame()
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # data is { "model": { "test_roc_auc": [f1, f2, ...], ... }, ... }
    rows = []
    for model, metrics in data.items():
        # find all test_ keys
        test_keys = [k for k in metrics.keys() if k.startswith('test_')]
        if not test_keys: continue
        
        # determine number of folds
        num_folds = len(metrics[test_keys[0]])
        for f_idx in range(num_folds):
            row = {'model': model, 'fold': f_idx + 1}
            for k in test_keys:
                metric_name = k.replace('test_', '')
                row[metric_name] = metrics[k][f_idx]
            rows.append(row)
            
    return pd.DataFrame(rows)

def plot_fold_variation(df: pd.DataFrame, output_dir: Path):
    if df.empty: return
    
    # We'll plot ROC-AUC variation by default as it's the primary metric
    metric = 'roc_auc' if 'roc_auc' in df.columns else df.columns[2]
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='model', y=metric, palette='Set2')
    sns.stripplot(data=df, x='model', y=metric, color='black', alpha=0.3, size=4)
    
    plt.title(f'Score Variation Across Folds: {metric.upper()}', fontsize=16, fontweight='bold')
    plt.ylabel(f'{metric.upper()} Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "fold_variation_boxplot.png", bbox_inches='tight')
    plt.close()
    print(f"Saved fold_variation_boxplot.png to {output_dir}")

def plot_all_fold_details(df_folds: pd.DataFrame, output_dir: Path):
    if df_folds.empty: return
    
    models = df_folds['model'].unique()
    for model in models:
        df_model = df_folds[df_folds['model'] == model].sort_values('fold')
        # Target metrics for detailed view
        metrics = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in df_model.columns]
        df_melt = df_model.melt(id_vars='fold', value_vars=metrics, var_name='metric', value_name='score')
        
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_melt, x='fold', y='score', hue='metric', marker='o', linewidth=2.5, markersize=8)
        
        plt.title(f'Model Details: {model.upper()} Performance across Folds', fontsize=16, fontweight='bold')
        plt.ylim(max(0, df_melt['score'].min() - 0.05), min(1.0, df_melt['score'].max() + 0.05))
        plt.xticks(df_model['fold'].unique())
        plt.ylabel('Score')
        plt.xlabel('Fold Number')
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / f"fold_details_{model}.png", bbox_inches='tight')
        plt.close()
        print(f"Saved fold_details_{model}.png to {output_dir}")

def generate_all_plots(cv_json: str, hyperopt_dir: str, output_dir: str, fold_json: str = None):
    cv_path = Path(cv_json)
    hyp_path = Path(hyperopt_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df_cv = load_cv_report(cv_path)
    df_hyp = load_hyperopt_data(hyp_path)
    df_folds = pd.DataFrame()
    if fold_json:
        df_folds = load_fold_metrics(Path(fold_json))
    
    print("Generating visualizations...")
    plot_metric_comparison(df_cv, out_path)
    plot_metric_heatmap(df_cv, out_path)
    plot_hyperopt_results(df_hyp, out_path)
    if not df_folds.empty:
        plot_fold_variation(df_folds, out_path)
        plot_all_fold_details(df_folds, out_path)
    
    # Generate a Detailed Performance Report Markdown
    if not df_cv.empty:
        # Sort by ROC-AUC by default
        df_cv_sorted = df_cv.sort_values('roc_auc', ascending=False)
        report_path = out_path / "model_performance_report.md"
        
        # Calculate stability (std across folds) if available
        stability_str = ""
        detailed_fold_str = ""
        if not df_folds.empty:
            # Stability summary
            stability_df = df_folds.groupby('model').std().reset_index()
            stability_df = stability_df[['model', 'roc_auc']].rename(columns={'roc_auc': 'roc_auc_std'})
            df_cv_sorted = df_cv_sorted.merge(stability_df, on='model', how='left')
            stability_str = "\n\n### Model Stability (Std Dev across Folds)\nLower is better (more stable).\n\n"
            stab_sorted = df_cv_sorted[['model', 'roc_auc_std']].sort_values('roc_auc_std')
            stability_str += stab_sorted.to_markdown(index=False)
            
            # Detailed Fold Tables for Each Model
            detailed_fold_str = "\n\n### Detailed Fold-by-Fold Performance Breakdown\n"
            for model_name in df_cv_sorted['model'].unique():
                df_model_folds = df_folds[df_folds['model'] == model_name].sort_values('fold')
                detailed_fold_str += f"\n#### {model_name.upper()}\n\n"
                detailed_fold_str += df_model_folds.to_markdown(index=False)
                detailed_fold_str += "\n"

        with open(report_path, "w") as f:
            f.write("# Model Performance Comparison Report\n\n")
            f.write("This report details the cross-validation performance across all models and metrics.\n\n")
            f.write("### Comparison Table\n")
            # Format as markdown table
            f.write(df_cv_sorted.to_markdown(index=False))
            
            if detailed_fold_str:
                f.write(detailed_fold_str)
                
            if stability_str:
                f.write(stability_str)
                
            f.write("\n\n### Best Model per Metric\n")
            for metric in ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']:
                if metric in df_cv.columns:
                    best_m = df_cv.iloc[df_cv[metric].idxmax()]
                    f.write(f"- **Top {metric.upper()}**: {best_m['model']} ({best_m[metric]:.4f})\n")
        
        print(f"Saved model_performance_report.md to {output_dir}")
        
        # Keep old top_performer.txt for compatibility but update content
        best_overall = df_cv_sorted.iloc[0]
        with open(out_path / "top_performer.txt", "w") as f:
            f.write(f"Best overall model: {best_overall['model']}\n")
            f.write(f"ROC-AUC: {best_overall['roc_auc']:.4f}\n")
            f.write(f"Accuracy: {best_overall['accuracy']:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model performance visualizations.")
    parser.add_argument("--cv-report", type=str, default="models/cv_report.json")
    parser.add_argument("--hyperopt-dir", type=str, default="models/hyperopt")
    parser.add_argument("--output-dir", type=str, default="models/comparison_plots")
    parser.add_argument("--fold-metrics", type=str, default="models/fold_metrics.json")
    
    args = parser.parse_args()
    generate_all_plots(args.cv_report, args.hyperopt_dir, args.output_dir, fold_json=args.fold_metrics)
