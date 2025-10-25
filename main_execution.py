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
    Main execution function for the complete research implementation.
    """
    
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
            print(f"
