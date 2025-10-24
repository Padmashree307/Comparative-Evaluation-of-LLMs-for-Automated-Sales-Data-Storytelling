"""
GenAI Data Storytelling Pipeline - Complete Execution Script
Research Paper: Sales Domain Implementation
Optimized for Gemini API
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
    print("Model: Google Gemini 2.0 Flash")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 1: CONFIGURE API KEYS
    # ========================================================================
    print("🔑 Step 1: Configure API Keys\n")
    
    # Using only Gemini for this implementation
    # Get your free Gemini API key from: https://ai.google.dev/
    API_KEYS = {
        'gemini': os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')
    }
    
    # Validate API key
    if API_KEYS['gemini'] == 'AIzaSyBqfOWU_R5NevAiKmRLIHD5HGICC5__-3w':
        print("   ⚠️  WARNING: Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API key")
        print("   Get your free API key at: https://ai.google.dev/\n")
        print("   Exiting...\n")
        return None
    
    print("   ✓ Gemini API key configured\n")
    
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
    
    try:
        results = pipeline.run_complete_pipeline(
            data_path=DATASET_PATH,
            report_type='executive_report',
            context=business_context
        )
    except Exception as e:
        print(f"\n❌ Pipeline execution failed with error:\n{str(e)}\n")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================================================
    # STEP 5: DISPLAY RESULTS SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80 + "\n")
    
    # Check if we have evaluations
    if results['evaluations']['evaluations']:
        comparison = results['evaluations']['comparison']
        
        print("🏆 Model Performance:\n")
        for ranking in comparison.get('rankings', []):
            print(f"   {ranking['rank']}. {ranking['model']}")
            print(f"      └─ Composite Score: {ranking['composite_score']}/100")
            print(f"      └─ Readability: {ranking['readability_score']:.1f} (Flesch)")
            print(f"      └─ Actionability: {ranking['actionability_score']:.1f}/100")
            print(f"      └─ Accuracy: {ranking['accuracy_rate']:.1f}%")
            print(f"      └─ Completeness: {ranking['completeness_score']:.1f}%\n")
        
        print("\n🎯 Best Model:")
        print(f"   Overall Winner: {comparison.get('best_overall', 'N/A')}")
        
    else:
        print("⚠️  No evaluations generated. Check the status report for details.\n")
    
    print("\n" + "="*80)
    print("✅ RESEARCH IMPLEMENTATION COMPLETE")
    print("="*80 + "\n")
    
    print("📁 Generated Outputs:")
    print(f"   ├─ Statistical insights JSON")
    print(f"   ├─ Structured insights JSON")
    
    successful_narratives = sum(1 for n in results['narratives'].values() if n['success'])
    print(f"   ├─ {successful_narratives} narrative markdown file(s)")
    
    if results['evaluations']['evaluations']:
        print(f"   ├─ Comprehensive evaluation JSON")
        print(f"   └─ Model comparison report")
    else:
        print(f"   └─ Status report (troubleshooting guide)")
    
    print(f"\n📂 All files saved in: research_outputs/\n")
    
    # ========================================================================
    # STEP 6: RESEARCH PAPER GUIDANCE
    # ========================================================================
    print("=" * 80)
    print("📝 NEXT STEPS FOR YOUR RESEARCH PAPER")
    print("=" * 80 + "\n")
    
    print("Your implementation is now complete! Use these outputs:\n")
    print("1. **Methodology Section:**")
    print("   - Reference the 7-component pipeline architecture")
    print("   - Explain statistical analysis metrics (KPIs, trends, segmentation)")
    print("   - Describe prompt engineering framework\n")
    
    print("2. **Results Section:**")
    print("   - Include model comparison table from the report")
    print("   - Add charts for readability, actionability, accuracy scores")
    print("   - Show example narrative excerpts\n")
    
    print("3. **Discussion Section:**")
    print("   - Analyze which quality metrics matter most for business")
    print("   - Compare LLM performance patterns")
    print("   - Discuss prompt engineering impact\n")
    
    print("4. **Visualizations to Create:**")
    print("   - Bar chart: Composite scores by model")
    print("   - Radar chart: Multi-dimensional quality comparison")
    print("   - Heatmap: Metric correlation analysis\n")
    
    print("Good luck with your research paper! 🎓\n")
    
    return results

if __name__ == "__main__":
    results = main()
