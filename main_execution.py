"""
GenAI Data Storytelling Pipeline - Complete Execution Script
Research Paper: Sales Domain Implementation
"""

import os
from generate_sales_dataset import generate_sales_dataset
from pipeline_orchestrator import GenAIStorytellingPipeline

def main():
    """
    Main execution function for the complete research implementation.
    """
    
    print("\n" + "="*80)
    print("GENAI-POWERED DATA STORYTELLING RESEARCH IMPLEMENTATION")
    print("Domain: Sales Analytics")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 1: CONFIGURE API KEYS
    # ========================================================================
    print("🔑 Step 1: Configure API Keys\n")
    
    # Replace with your actual API keys
    API_KEYS = {
        'gemini': os.getenv('GEMINI_API_KEY', 'your-gemini-api-key'),
        'cohere': os.getenv('COHERE_API_KEY', 'your-cohere-api-key'),
        'groq': os.getenv('GROQ_API_KEY', 'your-groq-api-key'),
        'huggingface': os.getenv('HUGGINGFACE_API_KEY', 'your-huggingface-api-key')
    }
    
    print("   ✓ API keys configured\n")
    
    # ========================================================================
    # STEP 2: GENERATE SALES DATASET
    # ========================================================================
    print("🔧 Step 2: Generate Sales Dataset\n")
    
    DATASET_PATH = 'sales_data_2024.csv'
    
    if not os.path.exists(DATASET_PATH):
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
    print("🚀 Step 3: Initialize Pipeline\n")
    
    pipeline = GenAIStorytellingPipeline(
        api_keys=API_KEYS,
        output_dir='research_outputs'
    )
    
    # ========================================================================
    # STEP 4: RUN COMPLETE PIPELINE
    # ========================================================================
    print("⚡ Step 4: Execute Complete Pipeline\n")
    
    business_context = {
        'industry': 'B2B Technology/SaaS',
        'period': 'FY 2024',
        'company': 'TechCorp Inc.',
        'strategic_priority': 'Revenue Growth & Customer Retention'
    }
    
    results = pipeline.run_complete_pipeline(
        data_path=DATASET_PATH,
        report_type='executive_report',
        context=business_context
    )
    
    # ========================================================================
    # STEP 5: DISPLAY RESULTS SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80 + "\n")
    
    # Model comparison
    comparison = results['evaluations']['comparison']
    
    print("🏆 Model Rankings:")
    for ranking in comparison['rankings']:
        print(f"   {ranking['rank']}. {ranking['model']}")
        print(f"      Composite Score: {ranking['composite_score']}/100")
        print(f"      Readability: {ranking['readability_score']}")
        print(f"      Actionability: {ranking['actionability_score']}/100")
        print(f"      Accuracy: {ranking['accuracy_rate']}%")
        print(f"      Completeness: {ranking['completeness_score']}%\n")
    
    print("\n🎯 Best Models by Category:")
    print(f"   Overall: {comparison['best_overall']}")
    print(f"   Readability: {comparison['best_readability']}")
    print(f"   Actionability: {comparison['best_actionability']}")
    print(f"   Accuracy: {comparison['best_accuracy']}")
    
    print("\n" + "="*80)
    print("✅ RESEARCH IMPLEMENTATION COMPLETE")
    print("="*80 + "\n")
    
    print("📁 Generated Outputs:")
    print(f"   - Statistical insights JSON")
    print(f"   - Structured insights JSON")
    print(f"   - {len(results['narratives'])} narrative markdown files")
    print(f"   - Comprehensive evaluation JSON")
    print(f"   - Model comparison report")
    
    print(f"\n📂 All files saved in: research_outputs/\n")
    
    return results

if __name__ == "__main__":
    results = main()
