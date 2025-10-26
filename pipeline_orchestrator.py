import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import os
import traceback
from dotenv import load_dotenv

from sales_statistical_engine import SalesStatisticalEngine
from insight_structuring import InsightStructurer
from sales_prompt_templates import SalesPromptEngineer
from multi_llm_generator import MultiLLMNarrativeGenerator
from quality_evaluator import NarrativeQualityEvaluator


class GenAIStorytellingPipeline:
    """
    Complete GenAI-Powered Data Storytelling Pipeline.
    Orchestrates end-to-end workflow from raw data to evaluated narratives.
    """

    def __init__(self, api_keys: Dict[str, str], output_dir: str = "outputs"):
        """
        Initialize the complete pipeline.
        """
        load_dotenv()
        
        self.api_keys = api_keys or {}
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate API keys
        def _is_valid_key(k: str) -> bool:
            if not k:
                return False
            if isinstance(k, str) and (k.strip() == "" or k.lower().startswith("your") or len(k) < 20):
                return False
            return True
        
        self.api_keys_valid = all(_is_valid_key(v) for v in self.api_keys.values())
        self.api_keys_validated = [k for k, v in self.api_keys.items() if _is_valid_key(v)]
        
        if not self.api_keys_valid:
            print(" ⚠️  One or more API keys appear invalid or placeholder values were used.")
            print(" ⚠️  Generation with external LLMs will be skipped - status report will be produced.")
        
        # Pipeline components
        self.statistical_engine = None
        self.insight_structurer = None
        self.prompt_engineer = SalesPromptEngineer()
        self.narrative_generator = MultiLLMNarrativeGenerator(self.api_keys) if self.api_keys_valid else None
        self.quality_evaluator = None
        
        # Runtime state - INITIALIZE AS EMPTY DICTS
        self.raw_data = None
        self.insights = {}
        self.structured_insights = {}
        self.prompts: Dict[str, str] = {}
        self.narratives: Dict[str, Dict] = {}
        self.evaluations: Dict[str, Dict] = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load CSV data into a pandas DataFrame."""
        print(f" 📂 Loading data from {data_path}...")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        self.raw_data = df
        print(f" ✓ Loaded {len(df)} records with {len(df.columns)} columns\n")
        return df

    def run_statistical_analysis(self) -> Dict:
        """Execute statistical analysis engine."""
        print(" 📊 Running Statistical Analysis Engine...")
        
        self.statistical_engine = SalesStatisticalEngine(self.raw_data)
        self.insights = self.statistical_engine.generate_full_analysis()
        
        print(f" ✓ Extracted {len(self.insights)} insight categories\n")
        
        # Save insights to JSON
        insights_path = os.path.join(self.output_dir, f"insights_{self.timestamp}.json")
        try:
            with open(insights_path, 'w', encoding='utf-8') as f:
                json.dump(self.insights, f, indent=2, default=str)
            print(f" ✓ Insights saved to {insights_path}\n")
        except Exception:
            print(" ⚠️  Failed to save insights file.\n")
        
        return self.insights

    def structure_insights(self) -> Dict:
        """Structure and organize insights."""
        print(" 🏗️  Structuring Insights...")
        
        self.insight_structurer = InsightStructurer(self.insights)
        self.structured_insights = self.insight_structurer.generate_structured_output()
        
        print(" ✓ Insights structured and categorized\n")
        
        # Save structured insights to JSON
        structured_path = os.path.join(self.output_dir, f"structured_insights_{self.timestamp}.json")
        try:
            with open(structured_path, 'w', encoding='utf-8') as f:
                json.dump(self.structured_insights, f, indent=2, default=str)
            print(f" ✓ Structured insights saved to {structured_path}\n")
        except Exception:
            print(" ⚠️  Failed to save structured insights file.\n")
        
        return self.structured_insights

    def generate_prompts(self, report_types: List[str] = None, context: Dict = None) -> Dict[str, str]:
        """Generate domain-specific prompts."""
        if report_types is None:
            report_types = ['executive_report']
        
        print(" ✍️  Generating Domain-Specific Prompts...")
        
        for report_type in report_types:
            prompt = self.prompt_engineer.generate_prompt(
                template_type=report_type,
                insights=self.structured_insights,
                context=context
            )
            self.prompts[report_type] = prompt
            print(f" ✓ Generated {report_type} prompt\n")
        
        return self.prompts

    def generate_narratives(self, prompt_type: str = 'executive_report') -> Dict:
        """Generate narratives using all LLMs."""
        if prompt_type not in self.prompts:
            raise ValueError(f"Prompt type '{prompt_type}' not found. Call generate_prompts() first.")
        
        prompt = self.prompts[prompt_type]
        print(f" 📝 Generating Narratives for '{prompt_type}'...\n")
        
        # CASE 1: API keys invalid - create placeholder failures
        if not self.api_keys_valid or not self.narrative_generator:
            model_keys = list(self.api_keys.keys()) if self.api_keys else ['gemini']
            self.narratives = {
                mk: {
                    'success': False,
                    'error': 'API key invalid or missing - generation skipped.',
                    'model': mk,
                    'content': None
                } for mk in model_keys
            }
            print(f" ⚠️  Skipped generation because API keys are invalid.")
            print(f" ⚠️  Created failure placeholders for: {', '.join(self.narratives.keys())}\n")
            return self.narratives
        
        # CASE 2: Normal generation path
        print(" 🚀 Starting Multi-LLM Narrative Generation...\n")
        try:
            self.narratives = self.narrative_generator.generate_all_narratives(prompt)
        except Exception as e:
            print(f" ❌ Narrative generation failed: {str(e)}")
            traceback.print_exc()
            
            # Create consistent failure shape if generator crashes
            model_keys = list(self.api_keys.keys()) if self.api_keys else ['gemini']
            self.narratives = {
                mk: {
                    'success': False,
                    'error': f'Generation failed: {str(e)}',
                    'model': mk,
                    'content': None
                } for mk in model_keys
            }
        
        # ENSURE self.narratives is always a dict
        if self.narratives is None:
            self.narratives = {}
        
        # SAFE ACCESS to narratives values
        successful_narratives = [n for n in self.narratives.values() if n.get('success')]
        failed_narratives = [n for n in self.narratives.values() if not n.get('success')]
        
        print(f"\n 📊 Generation Summary:")
        print(f" ✓ Successful: {len(successful_narratives)}")
        print(f" ✗ Failed: {len(failed_narratives)}")
        
        if failed_narratives:
            print(f"\n ⚠️  Failed Models:")
            for n in failed_narratives:
                print(f" - {n.get('model', 'unknown')}: {n.get('error', 'Unknown error')}")
        print()
        
        # Save narratives to files
        for model_key, narrative_data in self.narratives.items():
            try:
                out_path = os.path.join(self.output_dir, f"{model_key}_{self.timestamp}.md")
                with open(out_path, 'w', encoding='utf-8') as fh:
                    if narrative_data.get('success'):
                        fh.write(narrative_data.get('narrative', ''))
                    else:
                        fh.write(f"# Generation failed for {model_key}\n\n{narrative_data.get('error')}")
            except Exception:
                print(f" ⚠️  Failed to save narrative file for {model_key}")
        
        return self.narratives

    def evaluate_narratives(self) -> Dict:
        """Evaluate all generated narratives."""
        print(" 🎯 Evaluating Narrative Quality...\n")
        
        # ENSURE evaluations exists
        self.evaluations = {}
        
        # SAFE CHECK for successful narratives
        if not self.narratives or not isinstance(self.narratives, dict):
            print(" ⚠️  No narratives available to evaluate\n")
            return {'evaluations': self.evaluations, 'comparison': {}}
        
        successful_narratives = {k: v for k, v in self.narratives.items() if v.get('success')}
        
        if not successful_narratives:
            print(" ⚠️  No successful narratives to evaluate\n")
            
            # Generate a status report
            status_path = os.path.join(self.output_dir, f"status_report_{self.timestamp}.md")
            try:
                with open(status_path, 'w', encoding='utf-8') as f:
                    f.write("# Status Report\n\n")
                    f.write("No successful narrative generations. Check API keys and connectivity.\n")
                print(f" 📄 No evaluations available - generating error report")
                print(f" ✓ Status report saved to {status_path}\n")
            except Exception:
                print(" ⚠️  Failed to write status report.\n")
            
            return {'evaluations': self.evaluations, 'comparison': {}}
        
        # Initialize evaluator with Gemini as judge (if available)
        gemini_client = None
        try:
            if self.narrative_generator and hasattr(self.narrative_generator, 'clients'):
                gemini_client = self.narrative_generator.clients.get('gemini')
        except Exception:
            gemini_client = None
        
        self.quality_evaluator = NarrativeQualityEvaluator(
            ground_truth_insights=self.insights,
            llm_judge_client=gemini_client
        )
        
        # Evaluate each successful narrative
        for model_key, narrative_data in successful_narratives.items():
            try:
                eval_result = self.quality_evaluator.evaluate_narrative(
                    narrative_data['narrative'],
                    model_key
                )
                eval_result['model'] = model_key
                self.evaluations[model_key] = eval_result
            except Exception as e:
                self.evaluations[model_key] = {
                    'error': f'Evaluation failed: {str(e)}',
                    'model': model_key
                }
        
        # Generate comparison
        try:
            comparison = self.quality_evaluator.compare_models()
        except Exception:
            comparison = {}
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, f"evaluations_{self.timestamp}.json")
        try:
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {'evaluations': self.evaluations, 'comparison': comparison},
                    f,
                    indent=2,
                    default=str
                )
            print(f" ✓ Evaluation results saved to {eval_path}\n")
        except Exception:
            print(" ⚠️  Failed to save evaluation results.\n")
        
        return {'evaluations': self.evaluations, 'comparison': comparison}

    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if not self.evaluations:
            report = f"# GenAI Data Storytelling Pipeline - Status Report\n\n"
            report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += "**Status**: No successful narrative evaluations\n\n"
            return report
        
        # Normal comparison report path
        try:
            comparison = self.quality_evaluator.compare_models()
        except Exception as e:
            print(f" ⚠️  Failed to generate comparison: {e}")
            comparison = {}
        
        report = f"# GenAI Data Storytelling Pipeline - Model Comparison Report\n\n"
        report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Models Evaluated**: {len(self.evaluations)}\n\n"
        
        return report

    def run_complete_pipeline(self,data_path: str,report_type: str = 'executive_report',context: Dict = None) -> Dict:
        """Execute the complete end-to-end pipeline."""
        print("\n" + "=" * 80)
        print("STARTING COMPLETE GENAI DATA STORYTELLING PIPELINE")
        print("=" * 80 + "\n")
        
        # Step 1: Load Data
        self.load_data(data_path)
        
        # Step 2: Statistical Analysis
        self.run_statistical_analysis()
        
        # Step 3: Structure Insights
        self.structure_insights()
        
        # Step 4: Generate Prompts
        self.generate_prompts(report_types=[report_type], context=context)
        
        # Step 5: Generate Narratives
        self.generate_narratives(prompt_type=report_type)
        
        # Step 6: Evaluate Narratives
        eval_results = self.evaluate_narratives()
        
        # Step 7: Generate Comparison Report
        try:
            comparison_report = self.generate_comparison_report()
        except Exception as e:
            print(f" ⚠️  Comparison report generation skipped due to error: {e}")
            comparison_report = eval_results.get('comparison', {})
        
        import sys
        sys.stdout.flush()

        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 80 + "\n")

        # ==================== START: PIPELINE SUMMARY SECTION ====================
        # Generate and print pipeline execution summary
        print("📊 Pipeline Summary:")

        # Calculate summary statistics
        total_models = len(self.api_keys_validated)
        successful_gens = len([m for m, n in self.narratives.items() if n and n.get('success')])
        failed_gens = total_models - successful_gens

        print(f"   - Total models attempted: {total_models}")
        print(f"   - Successful generations: {successful_gens}")
        print(f"   - Failed generations: {failed_gens}")

        # Calculate generation times if available
        if self.narratives:
            gen_times = [n.get('generation_time', 0) for n in self.narratives.values() 
                        if n and n.get('generation_time')]
            if gen_times:
                avg_time = sum(gen_times) / len(gen_times)
                print(f"   - Average generation time: {avg_time:.1f}s")
                
                # Find fastest model
                fastest_model = min(self.narratives.items(), 
                                key=lambda x: x[1].get('generation_time', float('inf')) 
                                                if x[1] else float('inf'))
                if fastest_model[1] and fastest_model[1].get('generation_time'):
                    model_display = fastest_model[0].replace('_', ' ').title()
                    print(f"   - Fastest model: {model_display}")

        # Find best overall model based on composite score
        if self.evaluations:
            best_model = max(self.evaluations.items(), 
                        key=lambda x: x[1].get('composite_quality_score', 0) 
                                        if isinstance(x[1], dict) else 0)
            if best_model[1] and isinstance(best_model[1], dict) and \
            best_model[1].get('composite_quality_score', 0) > 0:
                model_display = best_model[0].replace('_', ' ').title()
                score = best_model[1].get('composite_quality_score', 0)
                print(f"   - Best overall model: {model_display} ({score:.1f}/100)")
            else:
                print(f"   - Best overall model: N/A")
        else:
            print(f"   - Best overall model: N/A")

        print()
        print(f"📁 All outputs saved to: {self.output_dir}/")
        print()
    
        return {
            'insights': self.insights,
            'structured_insights': self.structured_insights,
            'narratives': self.narratives,
            'evaluations': eval_results,
            'comparison_report': comparison_report
        }
