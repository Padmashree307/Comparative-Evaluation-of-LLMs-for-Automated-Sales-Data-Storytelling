"""
GenAI Data Storytelling Pipeline - Complete Execution Script
Research Paper: Sales Domain Implementation
Supports: Gemini, Cohere, Groq, Hugging Face
"""
from config import set_seeds
set_seeds()

import os
import sys
from dotenv import load_dotenv


from generate_sales_dataset import generate_sales_dataset
from pipeline_orchestrator import GenAIStorytellingPipeline


def main():
    """
    Main execution function for the complete research implementation.
    """
    # Initialize reproducibility
    
    
    print("\n" + "=" * 80)
    print("GENAI-POWERED DATA STORYTELLING RESEARCH IMPLEMENTATION")
    print("Domain: Sales Analytics")
    print("Models: Gemini, Cohere, Groq, Hugging Face")
    print("=" * 80 + "\n")
    
    # Load environment variables
    load_dotenv()
    
    # ========================================================================
    # STEP 1: CONFIGURE API KEYS FOR ALL LLMs
    # ========================================================================
    print("🔑 Step 1: Configure API Keys for All LLMs\n")
    
    API_KEYS = {
        'gemini': os.getenv('GEMINI_API_KEY', ''),
        'cohere': os.getenv('COHERE_API_KEY', ''),
        'groq': os.getenv('GROQ_API_KEY', ''),
        'huggingface': os.getenv('HUGGINGFACE_API_KEY', '')
    }
    
    # Check which keys are configured
    missing_keys = []
    configured_llms = []
    
    for llm_name, api_key in API_KEYS.items():
        if not api_key or api_key.startswith('YOUR_'):
            missing_keys.append(llm_name)
            print(f"   ⚠️  {llm_name.capitalize()}: API key not configured")
        else:
            configured_llms.append(llm_name)
            print(f"   ✓ {llm_name.capitalize()}: API key configured")
    
    if not configured_llms:
        print("\n   ❌ ERROR: No API keys configured!")
        print("\n   Please configure at least one API key:")
        print("   - Gemini: https://aistudio.google.com/apikey")
        print("   - Cohere: https://dashboard.cohere.com/api-keys")
        print("   - Groq: https://console.groq.com/keys")
        print("   - Hugging Face: https://huggingface.co/settings/tokens")
        print("\n   Set them in a .env file or as environment variables.")
        print("\n   Exiting...\n")
        return None
    
    print(f"\n   🎯 Proceeding with {len(configured_llms)} LLM(s): {', '.join(configured_llms)}\n")
    
    if missing_keys:
        print(f"   💡 Note: You can add {len(missing_keys)} more LLM(s) for comprehensive comparison:")
        for llm in missing_keys:
            print(f"   - {llm.capitalize()}")
        print()
    
    # ========================================================================
    # STEP 2: GENERATE SALES DATASET
    # ========================================================================
    print("🔧 Step 2: Generate Sales Dataset\n")
    
    DATASET_PATH = 'sales_data_2024.csv'
    
    if not os.path.exists(DATASET_PATH):
        print("   📊 Generating new sales dataset...")
        sales_df = generate_sales_dataset(
            n_records=5000,
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        sales_df.to_csv(DATASET_PATH, index=False)
        print(f"   ✓ Dataset created and saved to {DATASET_PATH}\n")
    else:
        print(f"   ✓ Using existing dataset: {DATASET_PATH}\n")
    
    # ========================================================================
    # STEP 3: INITIALIZE PIPELINE
    # ========================================================================
    print("🚀 Step 3: Initialize Multi-LLM Pipeline\n")
    
    try:
        pipeline = GenAIStorytellingPipeline(
            api_keys=API_KEYS,
            output_dir='research_outputs'
        )
        print(f"   ✓ Pipeline initialized with {len(configured_llms)} LLM(s)\n")
    except Exception as e:
        print(f"\n   ❌ Pipeline initialization failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================================================
    # STEP 4: RUN COMPLETE PIPELINE
    # ========================================================================
    print("⚡ Step 4: Execute Complete Pipeline with All LLMs\n")
    
    business_context = {
        'industry': 'B2B Technology/SaaS',
        'period': 'FY 2024',
        'company': 'TechCorp Inc.',
        'strategic_priority': 'Revenue Growth & Customer Retention'
    }
    
    try:
        results = pipeline.run_complete_pipeline(
            data_path=DATASET_PATH,
            report_type='executive_report',
            context=business_context
        )

        sys.stdout.flush()

    except Exception as e:
        print(f"\n   ❌ Pipeline execution failed with error:\n   {str(e)}\n")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================================================
    # STEP 5: DISPLAY RESULTS SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 MULTI-LLM RESULTS SUMMARY")
    print("=" * 80 + "\n")
    
    print("📝 Narrative Generation Status:\n")
    for llm_name, narrative_info in results['narratives'].items():
        if narrative_info.get('success'):
            print(f"   ✓ {llm_name.capitalize()}: Successfully generated")
        else:
            error_msg = narrative_info.get('error', 'Unknown error')
            print(f"   ✗ {llm_name.capitalize()}: Failed - {error_msg}")
    print()
    
    # Check if we have evaluations
    if results['evaluations']['evaluations']:
        comparison = results['evaluations']['comparison']
        
        print("🏆 Multi-LLM Performance Comparison:\n")
        rankings = comparison.get('rankings', [])
        
        if rankings:
            for ranking in rankings:
                print(f"   {ranking['rank']}. {ranking['model'].upper()}")
                print(f"      └─ Composite Score: {ranking['composite_score']}/100")
                print(f"      └─ Readability: {ranking['readability_score']:.1f} (Flesch)")
                print(f"      └─ Actionability: {ranking['actionability_score']:.1f}/100")
                print(f"      └─ Accuracy: {ranking['accuracy_rate']:.1f}%")
                print(f"      └─ Completeness: {ranking['completeness_score']:.1f}%\n")
            
            print("🎯 Best Performing Models:\n")
            print(f"   🥇 Overall Winner: {comparison.get('best_overall', 'N/A')}")
            
            if 'best_by_category' in comparison:
                cat_winners = comparison['best_by_category']
                print(f"   📖 Best Readability: {cat_winners.get('readability', 'N/A')}")
                print(f"   🎬 Best Actionability: {cat_winners.get('actionability', 'N/A')}")
                print(f"   ✅ Best Accuracy: {cat_winners.get('accuracy', 'N/A')}")
                print(f"   📋 Best Completeness: {cat_winners.get('completeness', 'N/A')}")
        else:
            print("   ⚠️  No rankings available")
    else:
        print("⚠️  No evaluations generated. Check the status report for details.\n")
    
    print("\n" + "=" * 80)
    print("✅ MULTI-LLM RESEARCH IMPLEMENTATION COMPLETE")
    print("=" * 80 + "\n")
    
    print("📁 Generated Outputs:\n")
    print("   ├─ Statistical insights JSON")
    print("   ├─ Structured insights JSON")
    
    successful_narratives = sum(1 for n in results['narratives'].values() if n.get('success'))
    total_llms = len(results['narratives'])
    print(f"   ├─ {successful_narratives}/{total_llms} narrative markdown files generated")
    
    if results['evaluations']['evaluations']:
        print("   ├─ Comprehensive multi-LLM evaluation JSON")
        print("   └─ Comparative model analysis report")
    else:
        print("   └─ Status report (troubleshooting guide)")
    
    print(f"\n📂 All files saved in: research_outputs/\n")
    
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
        plt.boxplot(metric_data.values(), tick_labels=metric_data.keys(), showmeans=True, patch_artist=True,
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

    generate_all()

    
    return results


if __name__ == "__main__":
    results = main()
