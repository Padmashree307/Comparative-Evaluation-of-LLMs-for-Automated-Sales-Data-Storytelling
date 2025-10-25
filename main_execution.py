# """
# GenAI Data Storytelling Pipeline - Complete Execution Script
# Research Paper: Sales Domain Implementation
# Optimized for Gemini API


# import os
# from generate_sales_dataset import generate_sales_dataset
# from pipeline_orchestrator import GenAIStorytellingPipeline

# def main():
#     """
#     Main execution function for the complete research implementation.
#     """
    
#     print("\n" + "="*80)
#     print("GENAI-POWERED DATA STORYTELLING RESEARCH IMPLEMENTATION")
#     print("Domain: Sales Analytics")
#     print("Model: Google Gemini 2.0 Flash")
#     print("="*80 + "\n")
    
#     # ========================================================================
#     # STEP 1: CONFIGURE API KEYS
#     # ========================================================================
#     print("🔑 Step 1: Configure API Keys\n")
    
#     # Using only Gemini for this implementation
#     # Get your free Gemini API key from: https://ai.google.dev/
#     API_KEYS = {
#         'gemini': os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')
#     }
    
#     # Validate API key
#     if API_KEYS['gemini'] == 'AIzaSyBqfOWU_R5NevAiKmRLIHD5HGICC5__-3w':
#         print("   ⚠️  WARNING: Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual Gemini API key")
#         print("   Get your free API key at: https://ai.google.dev/\n")
#         print("   Exiting...\n")
#         return None
    
#     print("   ✓ Gemini API key configured\n")
    
#     # ========================================================================
#     # STEP 2: GENERATE SALES DATASET
#     # ========================================================================
#     print("🔧 Step 2: Generate Sales Dataset\n")
    
#     DATASET_PATH = 'sales_data_2024.csv'
    
#     if not os.path.exists(DATASET_PATH):
#         sales_df = generate_sales_dataset(
#             n_records=5000,
#             start_date='2024-01-01',
#             end_date='2024-12-31'
#         )
#         sales_df.to_csv(DATASET_PATH, index=False)
#         print(f"   ✓ Dataset created and saved to {DATASET_PATH}\n")
#     else:
#         print(f"   ✓ Using existing dataset: {DATASET_PATH}\n")
    
#     # ========================================================================
#     # STEP 3: INITIALIZE PIPELINE
#     # ========================================================================
#     print("🚀 Step 3: Initialize Pipeline\n")
    
#     pipeline = GenAIStorytellingPipeline(
#         api_keys=API_KEYS,
#         output_dir='research_outputs'
#     )
    
#     # ========================================================================
#     # STEP 4: RUN COMPLETE PIPELINE
#     # ========================================================================
#     print("⚡ Step 4: Execute Complete Pipeline\n")
    
#     business_context = {
#         'industry': 'B2B Technology/SaaS',
#         'period': 'FY 2024',
#         'company': 'TechCorp Inc.',
#         'strategic_priority': 'Revenue Growth & Customer Retention'
#     }
    
#     try:
#         results = pipeline.run_complete_pipeline(
#             data_path=DATASET_PATH,
#             report_type='executive_report',
#             context=business_context
#         )
#     except Exception as e:
#         print(f"\n❌ Pipeline execution failed with error:\n{str(e)}\n")
#         import traceback
#         traceback.print_exc()
#         return None
    
#     # ========================================================================
#     # STEP 5: DISPLAY RESULTS SUMMARY
#     # ========================================================================
#     print("\n" + "="*80)
#     print("📊 RESULTS SUMMARY")
#     print("="*80 + "\n")
    
#     # Check if we have evaluations
#     if results['evaluations']['evaluations']:
#         comparison = results['evaluations']['comparison']
        
#         print("🏆 Model Performance:\n")
#         for ranking in comparison.get('rankings', []):
#             print(f"   {ranking['rank']}. {ranking['model']}")
#             print(f"      └─ Composite Score: {ranking['composite_score']}/100")
#             print(f"      └─ Readability: {ranking['readability_score']:.1f} (Flesch)")
#             print(f"      └─ Actionability: {ranking['actionability_score']:.1f}/100")
#             print(f"      └─ Accuracy: {ranking['accuracy_rate']:.1f}%")
#             print(f"      └─ Completeness: {ranking['completeness_score']:.1f}%\n")
        
#         print("\n🎯 Best Model:")
#         print(f"   Overall Winner: {comparison.get('best_overall', 'N/A')}")
        
#     else:
#         print("⚠️  No evaluations generated. Check the status report for details.\n")
    
#     print("\n" + "="*80)
#     print("✅ RESEARCH IMPLEMENTATION COMPLETE")
#     print("="*80 + "\n")
    
#     print("📁 Generated Outputs:")
#     print(f"   ├─ Statistical insights JSON")
#     print(f"   ├─ Structured insights JSON")
    
#     successful_narratives = sum(1 for n in results['narratives'].values() if n['success'])
#     print(f"   ├─ {successful_narratives} narrative markdown file(s)")
    
#     if results['evaluations']['evaluations']:
#         print(f"   ├─ Comprehensive evaluation JSON")
#         print(f"   └─ Model comparison report")
#     else:
#         print(f"   └─ Status report (troubleshooting guide)")
    
#     print(f"\n📂 All files saved in: research_outputs/\n")
    
#     # ========================================================================
#     # STEP 6: RESEARCH PAPER GUIDANCE
#     # ========================================================================
#     print("=" * 80)
#     print("📝 NEXT STEPS FOR YOUR RESEARCH PAPER")
#     print("=" * 80 + "\n")
    
#     print("Your implementation is now complete! Use these outputs:\n")
#     print("1. **Methodology Section:**")
#     print("   - Reference the 7-component pipeline architecture")
#     print("   - Explain statistical analysis metrics (KPIs, trends, segmentation)")
#     print("   - Describe prompt engineering framework\n")
    
#     print("2. **Results Section:**")
#     print("   - Include model comparison table from the report")
#     print("   - Add charts for readability, actionability, accuracy scores")
#     print("   - Show example narrative excerpts\n")
    
#     print("3. **Discussion Section:**")
#     print("   - Analyze which quality metrics matter most for business")
#     print("   - Compare LLM performance patterns")
#     print("   - Discuss prompt engineering impact\n")
    
#     print("4. **Visualizations to Create:**")
#     print("   - Bar chart: Composite scores by model")
#     print("   - Radar chart: Multi-dimensional quality comparison")
#     print("   - Heatmap: Metric correlation analysis\n")
    
#     print("Good luck with your research paper! 🎓\n")
    
#     return results

# if __name__ == "__main__":
#     results = main()
# """






"""
GenAI Data Storytelling Pipeline - Complete Execution Script

Research Paper: Sales Domain Implementation
Supports: Gemini, Cohere, Groq, Hugging Face
"""

import os
from dotenv import load_dotenv

from generate_sales_dataset import generate_sales_dataset
from pipeline_orchestrator import GenAIStorytellingPipeline

def main():
    """
    Main execution for the comparative LLM-powered sales data storytelling pipeline.
    """

    print("\n" + "=" * 80)
    print("GENAI-POWERED DATA STORYTELLING RESEARCH IMPLEMENTATION")
    print("Domain: Sales Analytics")
    print("Models: Gemini, Cohere, Groq, Hugging Face")
    print("=" * 80 + "\n")

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
        print("   - Gemini: https://ai.google.dev/")
        print("   - Cohere: https://dashboard.cohere.com/")
        print("   - Groq: https://console.groq.com/")
        print("   - Hugging Face: https://huggingface.co/settings/tokens")
        print("\n   Set them as environment variables or update the API_KEYS dictionary.")
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
    pipeline = GenAIStorytellingPipeline(api_keys=API_KEYS, output_dir='research_outputs')
    print(f"   ✓ Pipeline initialized with {len(configured_llms)} LLM(s)\n")

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
    except Exception as e:
        print(f"\n❌ Pipeline execution failed with error:\n{str(e)}\n")
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
            print(f"   ✗ {llm_name.capitalize()}: Failed - {narrative_info.get('error', 'Unknown error')}")
    print()

    if results['evaluations']['evaluations']:
        comparison = results['evaluations']['comparison']
        print("🏆 Multi-LLM Performance Comparison:\n")
        rankings = comparison.get('rankings', [])
        if rankings:
            for ranking in rankings:
                print(f"   {ranking['rank']}. {ranking['model'].upper()}")
                print(f"   └─ Composite Score: {ranking['composite_score']}/100")
                print(f"   └─ Readability: {ranking['readability_score']:.1f} (Flesch)")
                print(f"   └─ Actionability: {ranking['actionability_score']:.1f}/100")
                print(f"   └─ Accuracy: {ranking['accuracy_rate']:.1f}%")
                print(f"   └─ Completeness: {ranking['completeness_score']:.1f}%\n")
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

    print("=" * 80)
    print("📝 NEXT STEPS FOR YOUR MULTI-LLM RESEARCH PAPER")
    print("=" * 80 + "\n")
    print("Your multi-LLM implementation is now complete! Use these outputs:\n")
    print("1. **Methodology Section:**")
    print("   - Reference the 7-component pipeline architecture")
    print("   - Explain multi-LLM comparative framework\n")
    print("2. **Results Section:**")
    print("   - Include comprehensive model comparison table")
    print("   - Add comparative charts for each quality dimension\n")
    print("3. **Discussion Section:**")
    print("   - Analyze LLM-specific strengths and weaknesses")
    print("   - Compare performance patterns across models\n")
    print(f"   - Tested {total_llms} different LLMs")
    print(f"   - Successfully generated {successful_narratives} narratives\n")
    print("Good luck with your multi-LLM research paper! 🎓\n")
    return results

if __name__ == "__main__":
    results = main()
