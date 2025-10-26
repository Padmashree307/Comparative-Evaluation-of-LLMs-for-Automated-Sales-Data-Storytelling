"""
GENAI Multi-LLM Dynamic Visualization Script
Auto-loads your latest evaluation results and generates all research visuals:
- Bar Chart (Composite Scores)
- Radar Chart (Multi-Dimensional Quality)
- Heatmap (Metrics)
- Box Plot (Distribution)
- Side-by-Side Comparison Chart
"""

# ✅ Safe for Command Prompt execution (No GUI)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from glob import glob

# ========================== AUTO DETECT LATEST FILE ==================================
def load_latest_evaluation():
    eval_files = glob("research_outputs/evaluations_*.json")
    if not eval_files:
        raise FileNotFoundError("❌ No evaluation files found in research_outputs/. Run main_execution.py first.")
    latest_file = max(eval_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"📂 Using latest evaluation file: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    evaluations = data.get('evaluations', {})
    models = {}
    for model, metrics in evaluations.items():
        if metrics and metrics.get("composite_quality_score"):
            models[model.capitalize()] = {
                "Composite": metrics.get("composite_quality_score", 0),
                "Readability": metrics.get("readability", {}).get("flesch_reading_ease", 0),
                "Actionability": metrics.get("actionability", {}).get("actionability_score", 0),
                "Accuracy": metrics.get("statistical_accuracy", {}).get("accuracy_rate", 0),
                "Completeness": metrics.get("completeness", {}).get("completeness_score", 0)
            }
    return models


# ========================== CHARTS DIRECTORY INIT ==================================
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========================== VISUALIZATION FUNCTIONS ================================
def plot_bar_chart(data):
    """Bar chart: Composite Scores"""
    models = list(data.keys())
    scores = [data[m]['Composite'] for m in models]
    
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("husl", len(models))
    bars = plt.bar(models, scores, color=colors, edgecolor='black', alpha=0.85)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha='center', fontsize=12, fontweight='bold')
    plt.title("Composite Score Comparison", fontsize=15, fontweight='bold')
    plt.ylabel("Composite Score (Out of 100)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_composite_scores.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_radar_chart(data):
    """Radar chart: Quality across metrics"""
    metrics = ['Readability', 'Actionability', 'Accuracy', 'Completeness']
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    colors = sns.color_palette("husl", len(data))
    
    for idx, (model, values) in enumerate(data.items()):
        scores = [values[m] for m in metrics]
        scores += scores[:1]
        ax.plot(angles, scores, color=colors[idx], linewidth=2, label=model)
        ax.fill(angles, scores, color=colors[idx], alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_title("Multi-Dimensional Quality Comparison (Radar)", fontsize=15, fontweight="bold", pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_radar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(data):
    """Heatmap: Metrics per LLM"""
    models = list(data.keys())
    metrics = ['Readability', 'Actionability', 'Accuracy', 'Completeness']
    heat_data = [[data[m][k] for k in metrics] for m in models]
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(heat_data, annot=True, cmap="YlGnBu", fmt=".1f", 
                xticklabels=metrics, yticklabels=models, linewidths=1.5, linecolor='black',
                cbar_kws={'label': 'Score (0-100)'})
    plt.title("Model Quality Metric Heatmap", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_heatmap_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_box(data):
    """Box plot: Metric score volatility"""
    metric_data = {
        'Readability': [v['Readability'] for v in data.values()],
        'Actionability': [v['Actionability'] for v in data.values()],
        'Accuracy': [v['Accuracy'] for v in data.values()],
        'Completeness': [v['Completeness'] for v in data.values()]
    }
    
    plt.figure(figsize=(8, 5))
    plt.boxplot(metric_data.values(), labels=metric_data.keys(), showmeans=True, patch_artist=True,
                boxprops=dict(facecolor='#eaf2f8', color='black'),
                medianprops=dict(color='red'), meanprops=dict(color='blue', linewidth=2))
    plt.title("Metric Distribution Across Models", fontsize=15, fontweight='bold')
    plt.ylabel("Score (Out of 100)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_box_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_side_by_side(data):
    """Side-by-side bar: Compare all metrics for each model"""
    metrics = ['Readability', 'Actionability', 'Accuracy', 'Completeness']
    x = np.arange(len(metrics))
    width = 0.25
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(data))
    
    for i, (model, values) in enumerate(data.items()):
        scores = [values[m] for m in metrics]
        plt.bar(x + i*width, scores, width, label=model, color=colors[i], edgecolor='black')
    
    plt.xticks(x + width, metrics)
    plt.ylabel("Scores (Out of 100)")
    plt.title("Side-by-Side Quality Comparison")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_side_by_side.png", dpi=300, bbox_inches="tight")
    plt.close()


# ========================== RUN ALL ==================================
def generate_all():
    data = load_latest_evaluation()
    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS (Dynamic)")
    print("="*80)
    
    plot_bar_chart(data)
    plot_radar_chart(data)
    plot_heatmap(data)
    plot_box(data)
    plot_side_by_side(data)
    
    print(f"\n✓ All visualizations created successfully in: {OUTPUT_DIR.resolve()}")
    print("="*80)


if __name__ == '__main__':
    generate_all()
