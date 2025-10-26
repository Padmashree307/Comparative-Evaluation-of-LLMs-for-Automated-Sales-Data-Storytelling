"""
Visualization Module for Multi-LLM Research Analysis
Generates publication-ready charts from evaluation results
"""
import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ResearchVisualizer:
    """Generate publication-quality visualizations for LLM comparison research."""
    
    def __init__(self, evaluation_json_path: str, output_dir: str = "visualizations"):
        """Initialize with evaluation results."""
        with open(evaluation_json_path, 'r') as f:
            data = json.load(f)
        
        self.evaluations = data.get('evaluations', {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract successful models only
        self.models = {k: v for k, v in self.evaluations.items() 
                      if v and v.get('composite_quality_score')}
        
        print(f"✓ Loaded {len(self.models)} models for visualization")
    
    def plot_composite_scores(self):
        """Bar chart: Overall composite quality scores."""
        models = list(self.models.keys())
        scores = [self.models[m]['composite_quality_score'] for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax.bar(models, scores, color=colors[:len(models)], 
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Composite Quality Score (out of 100)', fontsize=13, fontweight='bold')
        ax.set_xlabel('LLM Model', fontsize=13, fontweight='bold')
        ax.set_title('Multi-LLM Performance Comparison\nComposite Quality Scores', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        path = self.output_dir / '01_composite_scores.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {path}")
    
    def plot_metric_comparison(self):
        """Grouped bar chart: All quality dimensions."""
        models = list(self.models.keys())
        metrics = ['Readability', 'Actionability', 'Accuracy', 'Completeness']
        
        # Extract metric values
        data = {
            'Readability': [],
            'Actionability': [],
            'Accuracy': [],
            'Completeness': []
        }
        
        for model in models:
            eval_data = self.models[model]
            # Normalize readability (Flesch 0-100, target ~60)
            flesch = eval_data.get('readability', {}).get('flesch_reading_ease', 0)
            readability_norm = min(100, max(0, flesch))
            data['Readability'].append(readability_norm)
            
            data['Actionability'].append(
                eval_data.get('actionability', {}).get('actionability_score', 0)
            )
            data['Accuracy'].append(
                eval_data.get('statistical_accuracy', {}).get('accuracy_rate', 0)
            )
            data['Completeness'].append(
                eval_data.get('completeness', {}).get('completeness_score', 0)
            )
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, metric in enumerate(metrics):
            offset = (i - 1.5) * width
            ax.bar(x + offset, data[metric], width, label=metric,
                  color=colors[i], alpha=0.85, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Score (out of 100)', fontsize=13, fontweight='bold')
        ax.set_xlabel('LLM Model', fontsize=13, fontweight='bold')
        ax.set_title('Multi-Dimensional Quality Metrics Comparison', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        path = self.output_dir / '02_metric_comparison.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {path}")
    
    def plot_radar_chart(self):
        """Radar chart: Model quality profiles."""
        from math import pi
        
        categories = ['Readability', 'Actionability', 'Accuracy', 'Completeness']
        models = list(self.models.keys())
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for idx, model in enumerate(models):
            eval_data = self.models[model]
            values = []
            
            # Readability
            flesch = eval_data.get('readability', {}).get('flesch_reading_ease', 0)
            values.append(min(100, max(0, flesch)))
            
            # Actionability
            values.append(eval_data.get('actionability', {}).get('actionability_score', 0))
            
            # Accuracy
            values.append(eval_data.get('statistical_accuracy', {}).get('accuracy_rate', 0))
            
            # Completeness
            values.append(eval_data.get('completeness', {}).get('completeness_score', 0))
            
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2.5, label=model.upper(), 
                   color=colors[idx], markersize=8)
            ax.fill(angles, values, alpha=0.2, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_title('LLM Quality Profile Comparison\n(Radar Chart)', 
                    fontsize=15, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / '03_radar_chart.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {path}")
    
    def plot_heatmap(self):
        """Heatmap: All metrics across models."""
        models = list(self.models.keys())
        metrics_names = ['Readability\n(Flesch)', 'Actionability', 'Accuracy', 'Completeness']
        
        data_matrix = []
        for model in models:
            eval_data = self.models[model]
            row = []
            
            # Readability
            flesch = eval_data.get('readability', {}).get('flesch_reading_ease', 0)
            row.append(min(100, max(0, flesch)))
            
            # Actionability
            row.append(eval_data.get('actionability', {}).get('actionability_score', 0))
            
            # Accuracy
            row.append(eval_data.get('statistical_accuracy', {}).get('accuracy_rate', 0))
            
            # Completeness
            row.append(eval_data.get('completeness', {}).get('completeness_score', 0))
            
            data_matrix.append(row)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score (0-100)'},
                   xticklabels=metrics_names, yticklabels=[m.upper() for m in models],
                   vmin=0, vmax=100, linewidths=2, linecolor='black', ax=ax,
                   annot_kws={'fontsize': 13, 'fontweight': 'bold'})
        
        ax.set_title('LLM Performance Heatmap', fontsize=15, fontweight='bold', pad=20)
        plt.yticks(rotation=0, fontsize=12)
        plt.xticks(fontsize=11)
        plt.tight_layout()
        path = self.output_dir / '04_heatmap.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {path}")
    
    def plot_generation_times(self):
        """Bar chart: Model generation speed comparison."""
        # This would require generation time data from your pipeline
        # Placeholder for now
        pass
    
    def generate_all(self):
        """Generate all visualizations."""
        print("\n" + "="*80)
        print("GENERATING RESEARCH VISUALIZATIONS")
        print("="*80 + "\n")
        
        self.plot_composite_scores()
        self.plot_metric_comparison()
        self.plot_radar_chart()
        self.plot_heatmap()
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}/\n")
        print("="*80)

# Main execution
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        eval_path = sys.argv[1]
    else:
        # Auto-detect latest evaluation file
        from glob import glob
        eval_files = glob('research_outputs/evaluations_*.json')
        if not eval_files:
            print("❌ No evaluation files found in research_outputs/")
            sys.exit(1)
        eval_path = max(eval_files, key=lambda x: Path(x).stat().st_mtime)
    
    print(f"📊 Using evaluation file: {eval_path}")
    
    viz = ResearchVisualizer(eval_path)
    viz.generate_all()
    
    print("\n✅ Visualization generation complete!")
    print("📁 Use these charts in your research paper Results section.\n")
