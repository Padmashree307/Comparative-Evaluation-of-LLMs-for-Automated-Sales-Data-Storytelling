"""
GENAI Multi-LLM Research Visualization
Creates all visualizations: Bar Chart, Radar Chart, Heatmap, Box Plot, and Narrative Comparison
"""

# ✅ Ensures compatibility for command prompt (no GUI required)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
DATA = {
    'Gemini': {'Composite': 79.31, 'Readability': 46.3, 'Actionability': 100, 'Accuracy': 47.2, 'Completeness': 100},
    'Cohere': {'Composite': 85.86, 'Readability': 36.7, 'Actionability': 100, 'Accuracy': 88.7, 'Completeness': 100},
    'Groq':   {'Composite': 81.75, 'Readability': 55.4, 'Actionability': 100, 'Accuracy': 42.3, 'Completeness': 100}
}

OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================
# 1️⃣ BAR CHART: COMPOSITE SCORES
# ================================
def bar_chart_scores():
    models = list(DATA.keys())
    scores = [DATA[m]['Composite'] for m in models]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, scores, color=['#16a085', '#2980b9', '#c0392b'], edgecolor='black', alpha=0.85)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.2f}", ha='center', fontsize=12, fontweight='bold')
    plt.title("Composite Score Comparison Across LLMs", fontsize=15, fontweight='bold')
    plt.ylabel("Composite Score (Out of 100)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_composite_scores.png", dpi=300, bbox_inches="tight")
    plt.close()


# ================================
# 2️⃣ RADAR CHART: OVERALL QUALITY
# ================================
def radar_chart():
    labels = ['Readability', 'Actionability', 'Accuracy', 'Completeness']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    colors = ['#1abc9c', '#3498db', '#e74c3c']

    for idx, (model, metrics) in enumerate(DATA.items()):
        values = [metrics[m] for m in labels]
        values += values[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=2, label=model)
        ax.fill(angles, values, color=colors[idx], alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.set_title("Multi-Dimensional LLM Quality Comparison (Radar)", fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_radar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()


# ================================
# 3️⃣ HEATMAP: QUALITY METRICS
# ================================
def heatmap_metrics():
    models = list(DATA.keys())
    metrics = ['Readability', 'Actionability', 'Accuracy', 'Completeness']
    
    data_matrix = [[DATA[m][metric] for metric in metrics] for m in models]

    plt.figure(figsize=(8, 5))
    sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap="YlGnBu", cbar_kws={'label': 'Score (Out of 100)'}, 
                xticklabels=metrics, yticklabels=models, linewidths=1, linecolor='black', annot_kws={'fontsize':11})
    plt.title("Heatmap of LLM Quality Metrics", fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_heatmap_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


# ================================
# 4️⃣ BOX PLOT: PERFORMANCE DISTRIBUTION
# ================================
def box_plot_performance():
    metric_data = {
        'Readability': [DATA[m]['Readability'] for m in DATA],
        'Actionability': [DATA[m]['Actionability'] for m in DATA],
        'Accuracy': [DATA[m]['Accuracy'] for m in DATA],
        'Completeness': [DATA[m]['Completeness'] for m in DATA]
    }

    plt.figure(figsize=(8, 5))
    plt.boxplot(metric_data.values(), labels=metric_data.keys(), showmeans=True,
                meanline=True, patch_artist=True,
                boxprops=dict(facecolor='#ecf0f1', color='black'),
                medianprops=dict(color='red', linewidth=1.5),
                meanprops=dict(color='blue', linewidth=2))
    plt.title("Performance Distribution by Metric", fontsize=15, fontweight='bold')
    plt.ylabel("Score (Out of 100)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_box_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


# ================================
# 5️⃣ SIDE-BY-SIDE COMPARISON (BAR + HATCH STYLE)
# ================================
def side_by_side_comparison():
    metrics = ['Readability', 'Accuracy', 'Actionability', 'Completeness']
    bar_width = 0.25
    x = np.arange(len(metrics))

    values_gemini = [DATA['Gemini'][m] for m in metrics]
    values_cohere = [DATA['Cohere'][m] for m in metrics]
    values_groq = [DATA['Groq'][m] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width, values_gemini, width=bar_width, color='#8e44ad', label='Gemini', hatch='//')
    plt.bar(x, values_cohere, width=bar_width, color='#27ae60', label='Cohere', hatch='xx')
    plt.bar(x + bar_width, values_groq, width=bar_width, color='#c0392b', label='Groq', hatch='\\\\')

    plt.xticks(x, metrics, fontsize=12, fontweight='bold')
    plt.ylabel("Metric Score (Out of 100)", fontsize=12)
    plt.title("Side-by-Side Metric Comparison Across Models", fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_side_by_side_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# ================================
# 🚀 MASTER FUNCTION
# ================================
def generate_all():
    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80)
    bar_chart_scores()
    radar_chart()
    heatmap_metrics()
    box_plot_performance()
    side_by_side_comparison()
    print("\n✓ All visualizations saved successfully in:", OUTPUT_DIR)
    print("="*80)


# Execute if run directly
if __name__ == "__main__":
    generate_all()
