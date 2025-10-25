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
        
        # Validate API keys (simple heuristic)
        def _is_valid_key(k: str) -> bool:
            if not k:
                return False
            if isinstance(k, str) and (k.strip() == "" or k.lower().startswith("your") or len(k) < 20):
                return False
            return True
        
        self.api_keys_valid = all(_is_valid_key(v) for v in self.api_keys.values())
        
        if not self.api_keys_valid:
            print(" ⚠️ One or more API keys appear invalid or placeholder values were used.")
            print(" ⚠️ Generation with external LLMs will be skipped - status report will be produced.")
        
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
        self.narratives: Dict[str, Dict] = {}  # ✅ Initialize as empty dict
        self.evaluations: Dict[str, Dict] = {}  # ✅ Initialize as empty dict

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load CSV data into a pandas DataFrame and store as raw_data."""
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
            print(" ⚠️ Failed to save insights file.\n")
        
        return self.insights

    def structure_insights(self) -> Dict:
        """Structure and organize insights."""
        print(" 🏗️ Structuring Insights...")
        
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
            print(" ⚠️ Failed to save structured insights file.\n")
        
        return self.structured_insights

    def generate_prompts(self, report_types: List[str] = None, context: Dict = None) -> Dict[str, str]:
        """Generate domain-specific prompts."""
        if report_types is None:
            report_types = ['executive_report']
        
        print(" ✍️ Generating Domain-Specific Prompts...")
        
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
            print(f" ⚠️ Skipped generation because API keys are invalid.")
            print(f" ⚠️ Created failure placeholders for: {', '.join(self.narratives.keys())}\n")
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
        
        # ✅ ENSURE self.narratives is always a dict
        if self.narratives is None:
            self.narratives = {}
        
        # ✅ SAFE ACCESS to narratives values
        successful_narratives = [n for n in self.narratives.values() if n.get('success')]
        failed_narratives = [n for n in self.narratives.values() if not n.get('success')]
        
        print(f"\n 📊 Generation Summary:")
        print(f" ✓ Successful: {len(successful_narratives)}")
        print(f" ✗ Failed: {len(failed_narratives)}")
        
        if failed_narratives:
            print(f"\n ⚠️ Failed Models:")
            for n in failed_narratives:
                print(f" - {n.get('model', 'unknown')}: {n.get('error', 'Unknown error')}")
        print()
        
        # Save narratives to files
        for model_key, narrative_data in self.narratives.items():
            try:
                out_path = os.path.join(self.output_dir, f"{model_key}_{self.timestamp}.md")
                with open(out_path, 'w', encoding='utf-8') as fh:
                    if narrative_data.get('success'):
                        fh.write(narrative_data.get('content', ''))
                    else:
                        fh.write(f"# Generation failed for {model_key}\n\n{narrative_data.get('error')}")
            except Exception:
                print(f" ⚠️ Failed to save narrative file for {model_key}")
        
        return self.narratives

    def evaluate_narratives(self) -> Dict:
        """Evaluate all generated narratives."""
        print(" 🎯 Evaluating Narrative Quality...\n")
        
        # ✅ ENSURE evaluations exists
        self.evaluations = {}
        
        # ✅ SAFE CHECK for successful narratives
        if not self.narratives or not isinstance(self.narratives, dict):
            print(" ⚠️ No narratives available to evaluate\n")
            return {'evaluations': self.evaluations, 'comparison': {}}
        
        successful_narratives = {k: v for k, v in self.narratives.items() if v.get('success')}
        
        if not successful_narratives:
            print(" ⚠️ No successful narratives to evaluate\n")
            
            # Generate a status report
            status_path = os.path.join(self.output_dir, f"status_report_{self.timestamp}.md")
            try:
                with open(status_path, 'w', encoding='utf-8') as f:
                    f.write("# Status Report\n\n")
                    f.write("No successful narrative generations. Check API keys and connectivity.\n")
                print(f" 📄 No evaluations available - generating error report")
                print(f" ✓ Status report saved to {status_path}\n")
            except Exception:
                print(" ⚠️ Failed to write status report.\n")
            
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
                    narrative_data['content'],
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
            print(" ⚠️ Failed to save evaluation results.\n")
        
        return {'evaluations': self.evaluations, 'comparison': comparison}

    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        # ... (rest of the method stays the same, just ensure safe dict access)
        # I'll provide the key fix:
        
        if not self.evaluations:
            report = f"# GenAI Data Storytelling Pipeline - Status Report\n\n"
            report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += "**Status**: No successful narrative evaluations\n\n"
            # ... rest of status report logic
            return report
        
        # Normal comparison report path
        try:
            comparison = self.quality_evaluator.compare_models()
        except Exception as e:
            print(f" ⚠️ Failed to generate comparison: {e}")
            comparison = {}
        
        # ... continue with report generation
        return report

    def run_complete_pipeline(
        self,
        data_path: str,
        report_type: str = 'executive_report',
        context: Dict = None
    ) -> Dict:
