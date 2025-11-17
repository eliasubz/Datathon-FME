import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import pathlib

def plot_feature_importance_heatmap(reports_dir, output_dir):
    """
    Reads feature importance from YAML reports and plots a heatmap.

    Args:
        reports_dir (str or pathlib.Path): Directory containing the YAML report files.
        output_dir (str or pathlib.Path): Directory to save the output heatmap plot.
    """
    reports_path = pathlib.Path(reports_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_importances = {}
    r = list(reports_path.glob("*_report.yaml"))
    for report_file in reports_path.glob("*_report.yaml"):
        model_name = report_file.stem.replace("_report", "")
        with open(report_file, 'r') as f:
            report = yaml.safe_load(f)
            if 'feature_importance' in report:
                all_importances[model_name] = report['feature_importance']

    if not all_importances:
        print("No feature importance found in the reports.")
        return

    importance_df = pd.DataFrame(all_importances).fillna(0)

    # Select top 30 features based on mean importance across models
    top_features = importance_df.mean(axis=1).nlargest(30).index
    top_importance_df = importance_df.loc[top_features]

    plt.figure(figsize=(15, 10))
    sns.heatmap(top_importance_df, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Top 30 Feature Importances Across Models")
    plt.xlabel("Model")
    plt.ylabel("Feature")
    
    heatmap_path = output_path / "feature_importance_heatmap.png"
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")

if __name__ == "__main__":
    REPORTS_DIR = "models/reports"
    OUTPUT_DIR = "plotting/output"
    plot_feature_importance_heatmap(REPORTS_DIR, OUTPUT_DIR)
