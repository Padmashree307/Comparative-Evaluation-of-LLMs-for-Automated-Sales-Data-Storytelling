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
            if isinstance(k, str) and (k.strip() == "" or k.lower().startswith("your_") or len(k) < 20):
                return False
            return True

        self.api_keys_valid = all(_is_valid_key(v) for v in self.api_keys.values())
        if not self.api_keys_valid:
            print("⚠️ One or more API keys appear invalid or placeholder values were used.")
            print("   → Generation with external LLM(s) will be skipped; status report will be produced.\n")

        # Pipeline components
        self.statistical_engine = None
        self.insight_structurer = None
        self.prompt_engineer = SalesPromptEngineer()
        self.narrative_generator = MultiLLMNarrativeGenerator(self.api_keys) if self.api_keys_valid else None
        self.quality_evaluator = None

        # Runtime state
        self.raw_data = None
        self.insights = {}
        self.structured_insights = {}
        self.prompts: Dict[str, str] = {}
        self.narratives: Dict[str, Dict] = {}
        self.evaluations: Dict[str, Dict] = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load CSV data into a pandas DataFrame and store as raw_data.
        """
        print(f"📂 Loading data from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        self.raw_data = df
        print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} columns\n")
        return df

    def run_statistical_analysis(self) -> Dict:
        """
        Execute statistical analysis engine.

        Returns:
            Complete insights dictionary
        """
        print("📊 Running Statistical Analysis Engine...")
        self.statistical_engine = SalesStatisticalEngine(self.raw_data)
        self.insights = self.statistical_engine.generate_full_analysis()

        print(f"   ✓ Extracted {len(self.insights)} insight categories\n")

        # Save insights to JSON
        insights_path = os.path.join(self.output_dir, f"insights_{self.timestamp}.json")
        try:
            with open(insights_path, 'w', encoding='utf-8') as f:
                json.dump(self.insights, f, indent=2, default=str)
            print(f"   💾 Insights saved to {insights_path}\n")
        except Exception:
            print("   ⚠️ Failed to save insights file.")

        return self.insights

    def structure_insights(self) -> Dict:
        """
        Structure and organize insights.

        Returns:
            Structured insights dictionary
        """
        print("🔧 Structuring Insights...")
        self.insight_structurer = InsightStructurer(self.insights)
        self.structured_insights = self.insight_structurer.generate_structured_output()

        print("   ✓ Insights structured and categorized\n")

        # Save structured insights
        structured_path = os.path.join(self.output_dir, f"structured_insights_{self.timestamp}.json")
        try:
            with open(structured_path, 'w', encoding='utf-8') as f:
                json.dump(self.structured_insights, f, indent=2, default=str)
            print(f"   💾 Structured insights saved to {structured_path}\n")
        except Exception:
            print("   ⚠️ Failed to save structured insights file.")

        return self.structured_insights

    def generate_prompts(self, report_types: List[str] = None, context: Dict = None) -> Dict[str, str]:
        """
        Generate domain-specific prompts.

        Args:
            report_types: List of report types to generate prompts for
            context: Additional business context

        Returns:
            Dictionary of prompts by report type
        """
        if report_types is None:
            report_types = ['executive_report']

        print("✍️  Generating Domain-Specific Prompts...")

        for report_type in report_types:
            prompt = self.prompt_engineer.generate_prompt(
                template_type=report_type,
                insights=self.structured_insights,
                context=context
            )
            self.prompts[report_type] = prompt
            print(f"   ✓ Generated {report_type} prompt\n")

        return self.prompts

    def generate_narratives(self, prompt_type: str = 'executive_report') -> Dict:
        """
        Generate narratives using all LLMs.
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Prompt type '{prompt_type}' not found. Call generate_prompts() first.")

        prompt = self.prompts[prompt_type]

        print(f"🤖 Generating Narratives for {prompt_type}...\n")

        if not self.api_keys_valid or not self.narrative_generator:
            # Create consistent failed entries so downstream code can operate
            model_keys = list(self.api_keys.keys()) if self.api_keys else ['gemini']
            self.narratives = {
                mk: {
                    'success': False,
                    'error': 'API key invalid or missing; generation skipped.',
                    'model': mk,
                    'content': None
                }
                for mk in model_keys
            }
            print(f"   ⚠️ Skipped generation because API keys are invalid. Created failure placeholders for: {', '.join(self.narratives.keys())}\n")
            return self.narratives

        # Normal generation path
        print("🚀 Starting Multi-LLM Narrative Generation...\n")
        try:
            self.narratives = self.narrative_generator.generate_all_narratives(prompt)
        except Exception as e:
            # Ensure consistent failure shape if generator crashes
            print(f"   ✗ Narrative generation failed: {str(e)}")
            traceback.print_exc()
            model_keys = list(self.api_keys.keys()) if self.api_keys else ['gemini']
            self.narratives = {
                mk: {
                    'success': False,
                    'error': f"Generation failed: {str(e)}",
                    'model': mk,
                    'content': None
                }
                for mk in model_keys
            }

        successful_narratives = [n for n in self.narratives.values() if n.get('success')]
        failed_narratives = [n for n in self.narratives.values() if not n.get('success')]

        print(f"\n📊 Generation Summary:")
        print(f"   ✅ Successful: {len(successful_narratives)}")
        print(f"   ❌ Failed: {len(failed_narratives)}")

        if failed_narratives:
            print("\n   ⚠️  Failed Models:")
            for n in failed_narratives:
                print(f"      • {n.get('model', 'unknown')}: {n.get('error', 'Unknown error')}")
        print()

        # Save narratives
        for model_key, narrative_data in self.narratives.items():
            try:
                out_path = os.path.join(self.output_dir, f"{model_key}_{self.timestamp}.md")
                with open(out_path, 'w', encoding='utf-8') as fh:
                    if narrative_data.get('success'):
                        fh.write(narrative_data.get('content', ''))
                    else:
                        fh.write(f"# Generation failed for {model_key}\n\n{narrative_data.get('error')}\n")
            except Exception:
                # don't crash saving outputs
                print(f"   ⚠️ Failed to save narrative file for {model_key}")

        print("\n")
        return self.narratives

    def evaluate_narratives(self) -> Dict:
        """
        Evaluate all generated narratives.
        """
        print("📈 Evaluating Narrative Quality...\n")

        # Ensure evaluations exists
        self.evaluations = {}

        # Check if we have any successful narratives
        successful_narratives = {k: v for k, v in self.narratives.items() if v.get('success')}

        if not successful_narratives:
            print("   ⚠️  No successful narratives to evaluate\n")
            # Generate a status report (consistent structure)
            status_path = os.path.join(self.output_dir, f"status_report_{self.timestamp}.md")
            try:
                with open(status_path, 'w', encoding='utf-8') as f:
                    f.write("# Status Report\n\n")
                    f.write("No successful narrative generations. Check API keys and connectivity.\n")
                print(f"   ⚠️  No evaluations available - generating error report\n")
                print(f"   📄 Status report saved to {status_path}\n")
            except Exception:
                print("   ⚠️ Failed to write status report.")

            # Ensure we return a consistent dict structure
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
                eval_result = self.quality_evaluator.evaluate_narrative(narrative_data['content'], model_key)
                eval_result['model'] = model_key
                self.evaluations[model_key] = eval_result
            except Exception as e:
                self.evaluations[model_key] = {
                    'error': f"Evaluation failed: {str(e)}",
                    'model': model_key
                }

        # Generate comparison (safe)
        try:
            comparison = self.quality_evaluator.compare_models()
        except Exception:
            comparison = {}

        # Save evaluation results
        eval_path = os.path.join(self.output_dir, f"evaluations_{self.timestamp}.json")
        try:
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump({'evaluations': self.evaluations, 'comparison': comparison}, f, indent=2, default=str)
            print(f"   💾 Evaluation results saved to {eval_path}\n")
        except Exception:
            print("   ⚠️ Failed to save evaluation results.")

        return {'evaluations': self.evaluations, 'comparison': comparison}

    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        Returns a markdown report string. If no evaluations exist, returns a status report.
        """
        if not self.evaluations:
            # Build and save a status report
            report = f"""# GenAI Data Storytelling Pipeline - Status Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status:** ❌ No successful narrative evaluations

---

## Issue Summary

No LLM models successfully generated narratives for evaluation.

### Possible Causes:
1. **Invalid API Keys** - Check that your API keys are correct and active
2. **API Rate Limits** - You may have exceeded free tier limits
3. **Network Issues** - Check your internet connection
4. **Model Availability** - Some models might be temporarily unavailable

### Models Attempted:
"""
            for model_key, narrative_data in self.narratives.items():
                status = "✅ SUCCESS" if narrative_data.get('success') else "❌ FAILED"
                report += f"- {narrative_data.get('model', model_key)}: {status}\n"
                if not narrative_data.get('success'):
                    report += f"  - Error: {narrative_data.get('error', 'Unknown')}\n"

            report += """

### Next Steps:
1. Verify your API keys are correct
2. Check API usage limits at provider dashboards
3. Ensure you have active internet connection
4. Try running with a single model first (e.g., just Gemini)
"""
            # Save error report
            report_path = os.path.join(self.output_dir, f"status_report_{self.timestamp}.md")
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"   📄 Status report saved to {report_path}\n")
            except Exception:
                print("   ⚠️ Failed to write status report file.")
            return report

        # Normal comparison report path
        try:
            comparison = self.quality_evaluator.compare_models()
        except Exception as e:
            print(f"   ⚠️ Failed to generate comparison: {e}")
            comparison = {}

        report = f"""# GenAI Data Storytelling Pipeline - Model Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Type:** Sales Performance Analytics
**Models Evaluated:** {len(self.evaluations)}

---

## Executive Summary

**Best Overall Model:** {comparison.get('best_overall', 'N/A')}
**Best Readability:** {comparison.get('best_readability', 'N/A')}
**Best Actionability:** {comparison.get('best_actionability', 'N/A')}
**Best Accuracy:** {comparison.get('best_accuracy', 'N/A')}

---

## Model Rankings
| Rank | Model | Composite Score | Readability | Actionability | Accuracy | Completeness |
|------|-------|----------------|-------------|---------------|----------|--------------|
"""
        for ranking in comparison.get('rankings', []):
            # Use safe access and formatting defaults
            report += "| {rank} | {model} | {cs} | {r:.1f} | {a:.1f} | {acc:.1f}% | {comp:.1f}% |\n".format(
                rank=ranking.get('rank', 'N/A'),
                model=ranking.get('model', 'N/A'),
                cs=ranking.get('composite_score', 'N/A'),
                r=ranking.get('readability_score', 0) or 0,
                a=ranking.get('actionability_score', 0) or 0,
                acc=ranking.get('accuracy_rate', 0) or 0,
                comp=ranking.get('completeness_score', 0) or 0
            )

        report += "\n---\n\n## Detailed Evaluation Metrics\n\n"

        for model_key, eval_data in self.evaluations.items():
            report += f"### {eval_data.get('model', model_key)}\n\n"
            report += f"**Word Count:** {eval_data.get('word_count', 'N/A')}\n\n"
            report += f"**Composite Quality Score:** {eval_data.get('composite_quality_score', 'N/A')}/100\n\n"

            rd = eval_data.get('readability', {})
            report += "#### Readability Metrics\n"
            report += f"- Flesch Reading Ease: {rd.get('flesch_reading_ease', 'N/A')}\n"
            report += f"- Flesch-Kincaid Grade: {rd.get('flesch_kincaid_grade', 'N/A')}\n"
            report += f"- Assessment: {rd.get('readability_assessment', 'N/A')}\n\n"

            act = eval_data.get('actionability', {})
            report += "#### Actionability Metrics\n"
            report += f"- Actionability Score: {act.get('actionability_score', 'N/A')}/100\n"
            report += f"- Action Verbs: {act.get('action_verbs_count', 'N/A')}\n"
            report += f"- Specific Metrics: {act.get('specific_metrics_count', 'N/A')}\n"
            report += f"- Timeframes Mentioned: {act.get('timeframes_mentioned', 'N/A')}\n"
            report += f"- Has Recommendation Section: {'Yes' if act.get('has_recommendation_section') else 'No'}\n\n"

            sa = eval_data.get('statistical_accuracy', {})
            report += "#### Statistical Accuracy\n"
            report += f"- Accuracy Rate: {sa.get('accuracy_rate', 'N/A')}%\n"
            report += f"- Matched Numbers: {sa.get('matched_numbers', 'N/A')}/{sa.get('total_numbers_in_narrative', 'N/A')}\n"
            report += f"- Assessment: {sa.get('assessment', 'N/A')}\n\n"

            comp = eval_data.get('completeness', {})
            report += "#### Completeness\n"
            report += f"- Completeness Score: {comp.get('completeness_score', 'N/A')}%\n"
            report += f"- Covered Themes: {', '.join(comp.get('covered_themes', [])) if comp.get('covered_themes') else 'None'}\n"
            report += f"- Missing Themes: {', '.join(comp.get('missing_themes', [])) if comp.get('missing_themes') else 'None'}\n"
            report += f"- Assessment: {comp.get('assessment', 'N/A')}\n\n"

            if 'llm_judge_scores' in eval_data and not eval_data.get('llm_judge_scores', {}).get('error'):
                scores = eval_data['llm_judge_scores']
                report += "#### LLM Judge Evaluation\n"
                report += f"- Business Relevance: {scores.get('business_relevance', 'N/A')}/5\n"
                report += f"- Executive Readiness: {scores.get('executive_readiness', 'N/A')}/5\n"
                report += f"- Narrative Flow: {scores.get('narrative_flow', 'N/A')}/5\n"
                report += f"- Persuasiveness: {scores.get('persuasiveness', 'N/A')}/5\n"
                report += f"- Average Score: {scores.get('average_score', 'N/A')}/5\n"
                if 'overall_assessment' in scores:
                    report += f"- Overall Assessment: {scores['overall_assessment']}\n"
                report += "\n"

            report += "---\n\n"

        # Save comparison report
        report_path = os.path.join(self.output_dir, f"comparison_report_{self.timestamp}.md")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📊 Comparison report generated: {report_path}\n")
        except Exception:
            print("   ⚠️ Failed to save comparison report file.")

        return report

    def run_complete_pipeline(self, data_path: str, report_type: str = 'executive_report',
                              context: Dict = None) -> Dict:
        """
        Execute the complete end-to-end pipeline.

        Args:
            data_path: Path to sales data CSV
            report_type: Type of report to generate
            context: Additional business context

        Returns:
            Complete pipeline results
        """
        print("=" * 80)
        print("🚀 STARTING COMPLETE GENAI DATA STORYTELLING PIPELINE")
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

        # Step 7: Generate Comparison Report (guarded)
        try:
            comparison_report = self.generate_comparison_report()
        except Exception as e:
            print(f"⚠️ Comparison report generation skipped due to error: {e}\n")
            comparison_report = eval_results.get('comparison', {})

        print("=" * 80)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print("=" * 80 + "\n")

        # Safe generation summary
        if self.narrative_generator and hasattr(self.narrative_generator, 'get_generation_summary'):
            try:
                gen_summary = self.narrative_generator.get_generation_summary()
            except Exception:
                gen_summary = {}
        else:
            # Derive basic summary from narratives dict
            total = len(self.narratives)
            successful = sum(1 for v in self.narratives.values() if v.get('success'))
            failed = total - successful
            gen_summary = {
                'total_models': total,
                'successful_generations': successful,
                'failed_generations': failed,
                'average_generation_time': None,
                'fastest_model': None,
                'best_overall_model': None
            }

        print("📊 Pipeline Summary:")
        print(f"   - Total models attempted: {gen_summary.get('total_models', 0)}")
        print(f"   - Successful generations: {gen_summary.get('successful_generations', 0)}")
        print(f"   - Failed generations: {gen_summary.get('failed_generations', 0)}")

        if gen_summary.get('successful_generations', 0) > 0:
            print(f"   - Average generation time: {gen_summary.get('average_generation_time', 0)}s")
            print(f"   - Fastest model: {gen_summary.get('fastest_model', 'N/A')}")
            try:
                best_overall = eval_results.get('comparison', {}).get('best_overall', 'N/A')
            except Exception:
                best_overall = 'N/A'
            print(f"   - Best overall model: {best_overall}")

        print(f"\n📁 All outputs saved to: {self.output_dir}/\n")

        return {
            'insights': self.insights,
            'structured_insights': self.structured_insights,
            'narratives': self.narratives,
            'evaluations': eval_results,
            'comparison_report': comparison_report,
            'generation_summary': gen_summary
        }
# filepath: d:\sacaim\pipeline_orchestrator.py
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import os
import traceback

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
        self.api_keys = api_keys or {}
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Validate API keys (simple heuristic)
        def _is_valid_key(k: str) -> bool:
            if not k:
                return False
            if isinstance(k, str) and (k.strip() == "" or k.lower().startswith("your_") or len(k) < 20):
                return False
            return True

        self.api_keys_valid = all(_is_valid_key(v) for v in self.api_keys.values())
        if not self.api_keys_valid:
            print("⚠️ One or more API keys appear invalid or placeholder values were used.")
            print("   → Generation with external LLM(s) will be skipped; status report will be produced.\n")

        # Pipeline components
        self.statistical_engine = None
        self.insight_structurer = None
        self.prompt_engineer = SalesPromptEngineer()
        self.narrative_generator = MultiLLMNarrativeGenerator(self.api_keys) if self.api_keys_valid else None
        self.quality_evaluator = None

        # Runtime state
        self.raw_data = None
        self.insights = {}
        self.structured_insights = {}
        self.prompts: Dict[str, str] = {}
        self.narratives: Dict[str, Dict] = {}
        self.evaluations: Dict[str, Dict] = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load CSV data into a pandas DataFrame and store as raw_data.
        """
        print(f"📂 Loading data from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        self.raw_data = df
        print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} columns\n")
        return df

    def run_statistical_analysis(self) -> Dict:
        """
        Execute statistical analysis engine.

        Returns:
            Complete insights dictionary
        """
        print("📊 Running Statistical Analysis Engine...")
        self.statistical_engine = SalesStatisticalEngine(self.raw_data)
        self.insights = self.statistical_engine.generate_full_analysis()

        print(f"   ✓ Extracted {len(self.insights)} insight categories\n")

        # Save insights to JSON
        insights_path = os.path.join(self.output_dir, f"insights_{self.timestamp}.json")
        try:
            with open(insights_path, 'w', encoding='utf-8') as f:
                json.dump(self.insights, f, indent=2, default=str)
            print(f"   💾 Insights saved to {insights_path}\n")
        except Exception:
            print("   ⚠️ Failed to save insights file.")

        return self.insights

    def structure_insights(self) -> Dict:
        """
        Structure and organize insights.

        Returns:
            Structured insights dictionary
        """
        print("🔧 Structuring Insights...")
        self.insight_structurer = InsightStructurer(self.insights)
        self.structured_insights = self.insight_structurer.generate_structured_output()

        print("   ✓ Insights structured and categorized\n")

        # Save structured insights
        structured_path = os.path.join(self.output_dir, f"structured_insights_{self.timestamp}.json")
        try:
            with open(structured_path, 'w', encoding='utf-8') as f:
                json.dump(self.structured_insights, f, indent=2, default=str)
            print(f"   💾 Structured insights saved to {structured_path}\n")
        except Exception:
            print("   ⚠️ Failed to save structured insights file.")

        return self.structured_insights

    def generate_prompts(self, report_types: List[str] = None, context: Dict = None) -> Dict[str, str]:
        """
        Generate domain-specific prompts.

        Args:
            report_types: List of report types to generate prompts for
            context: Additional business context

        Returns:
            Dictionary of prompts by report type
        """
        if report_types is None:
            report_types = ['executive_report']

        print("✍️  Generating Domain-Specific Prompts...")

        for report_type in report_types:
            prompt = self.prompt_engineer.generate_prompt(
                template_type=report_type,
                insights=self.structured_insights,
                context=context
            )
            self.prompts[report_type] = prompt
            print(f"   ✓ Generated {report_type} prompt\n")

        return self.prompts

    def generate_narratives(self, prompt_type: str = 'executive_report') -> Dict:
        """
        Generate narratives using all LLMs.
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Prompt type '{prompt_type}' not found. Call generate_prompts() first.")

        prompt = self.prompts[prompt_type]

        print(f"🤖 Generating Narratives for {prompt_type}...\n")

        if not self.api_keys_valid or not self.narrative_generator:
            # Create consistent failed entries so downstream code can operate
            model_keys = list(self.api_keys.keys()) if self.api_keys else ['gemini']
            self.narratives = {
                mk: {
                    'success': False,
                    'error': 'API key invalid or missing; generation skipped.',
                    'model': mk,
                    'content': None
                }
                for mk in model_keys
            }
            print(f"   ⚠️ Skipped generation because API keys are invalid. Created failure placeholders for: {', '.join(self.narratives.keys())}\n")
            return self.narratives

        # Normal generation path
        print("🚀 Starting Multi-LLM Narrative Generation...\n")
        try:
            self.narratives = self.narrative_generator.generate_all_narratives(prompt)
        except Exception as e:
            # Ensure consistent failure shape if generator crashes
            print(f"   ✗ Narrative generation failed: {str(e)}")
            traceback.print_exc()
            model_keys = list(self.api_keys.keys()) if self.api_keys else ['gemini']
            self.narratives = {
                mk: {
                    'success': False,
                    'error': f"Generation failed: {str(e)}",
                    'model': mk,
                    'content': None
                }
                for mk in model_keys
            }

        successful_narratives = [n for n in self.narratives.values() if n.get('success')]
        failed_narratives = [n for n in self.narratives.values() if not n.get('success')]

        print(f"\n📊 Generation Summary:")
        print(f"   ✅ Successful: {len(successful_narratives)}")
        print(f"   ❌ Failed: {len(failed_narratives)}")

        if failed_narratives:
            print("\n   ⚠️  Failed Models:")
            for n in failed_narratives:
                print(f"      • {n.get('model', 'unknown')}: {n.get('error', 'Unknown error')}")
        print()

        # Save narratives
        for model_key, narrative_data in self.narratives.items():
            try:
                out_path = os.path.join(self.output_dir, f"{model_key}_{self.timestamp}.md")
                with open(out_path, 'w', encoding='utf-8') as fh:
                    if narrative_data.get('success'):
                        fh.write(narrative_data.get('content', ''))
                    else:
                        fh.write(f"# Generation failed for {model_key}\n\n{narrative_data.get('error')}\n")
            except Exception:
                # don't crash saving outputs
                print(f"   ⚠️ Failed to save narrative file for {model_key}")

        print("\n")
        return self.narratives

    def evaluate_narratives(self) -> Dict:
        """
        Evaluate all generated narratives.
        """
        print("📈 Evaluating Narrative Quality...\n")

        # Ensure evaluations exists
        self.evaluations = {}

        # Check if we have any successful narratives
        successful_narratives = {k: v for k, v in self.narratives.items() if v.get('success')}

        if not successful_narratives:
            print("   ⚠️  No successful narratives to evaluate\n")
            # Generate a status report (consistent structure)
            status_path = os.path.join(self.output_dir, f"status_report_{self.timestamp}.md")
            try:
                with open(status_path, 'w', encoding='utf-8') as f:
                    f.write("# Status Report\n\n")
                    f.write("No successful narrative generations. Check API keys and connectivity.\n")
                print(f"   ⚠️  No evaluations available - generating error report\n")
                print(f"   📄 Status report saved to {status_path}\n")
            except Exception:
                print("   ⚠️ Failed to write status report.")

            # Ensure we return a consistent dict structure
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
                eval_result = self.quality_evaluator.evaluate_narrative(narrative_data['content'], model_key)
                eval_result['model'] = model_key
                self.evaluations[model_key] = eval_result
            except Exception as e:
                self.evaluations[model_key] = {
                    'error': f"Evaluation failed: {str(e)}",
                    'model': model_key
                }

        # Generate comparison (safe)
        try:
            comparison = self.quality_evaluator.compare_models()
        except Exception:
            comparison = {}

        # Save evaluation results
        eval_path = os.path.join(self.output_dir, f"evaluations_{self.timestamp}.json")
        try:
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump({'evaluations': self.evaluations, 'comparison': comparison}, f, indent=2, default=str)
            print(f"   💾 Evaluation results saved to {eval_path}\n")
        except Exception:
            print("   ⚠️ Failed to save evaluation results.")

        return {'evaluations': self.evaluations, 'comparison': comparison}

    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        Returns a markdown report string. If no evaluations exist, returns a status report.
        """
        if not self.evaluations:
            # Build and save a status report
            report = f"""# GenAI Data Storytelling Pipeline - Status Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status:** ❌ No successful narrative evaluations

---

## Issue Summary

No LLM models successfully generated narratives for evaluation.

### Possible Causes:
1. **Invalid API Keys** - Check that your API keys are correct and active
2. **API Rate Limits** - You may have exceeded free tier limits
3. **Network Issues** - Check your internet connection
4. **Model Availability** - Some models might be temporarily unavailable

### Models Attempted:
"""
            for model_key, narrative_data in self.narratives.items():
                status = "✅ SUCCESS" if narrative_data.get('success') else "❌ FAILED"
                report += f"- {narrative_data.get('model', model_key)}: {status}\n"
                if not narrative_data.get('success'):
                    report += f"  - Error: {narrative_data.get('error', 'Unknown')}\n"

            report += """

### Next Steps:
1. Verify your API keys are correct
2. Check API usage limits at provider dashboards
3. Ensure you have active internet connection
4. Try running with a single model first (e.g., just Gemini)
"""
            # Save error report
            report_path = os.path.join(self.output_dir, f"status_report_{self.timestamp}.md")
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"   📄 Status report saved to {report_path}\n")
            except Exception:
                print("   ⚠️ Failed to write status report file.")
            return report

        # Normal comparison report path
        try:
            comparison = self.quality_evaluator.compare_models()
        except Exception as e:
            print(f"   ⚠️ Failed to generate comparison: {e}")
            comparison = {}

        report = f"""# GenAI Data Storytelling Pipeline - Model Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Type:** Sales Performance Analytics
**Models Evaluated:** {len(self.evaluations)}

---

## Executive Summary

**Best Overall Model:** {comparison.get('best_overall', 'N/A')}
**Best Readability:** {comparison.get('best_readability', 'N/A')}
**Best Actionability:** {comparison.get('best_actionability', 'N/A')}
**Best Accuracy:** {comparison.get('best_accuracy', 'N/A')}

---

## Model Rankings
| Rank | Model | Composite Score | Readability | Actionability | Accuracy | Completeness |
|------|-------|----------------|-------------|---------------|----------|--------------|
"""
        for ranking in comparison.get('rankings', []):
            # Use safe access and formatting defaults
            report += "| {rank} | {model} | {cs} | {r:.1f} | {a:.1f} | {acc:.1f}% | {comp:.1f}% |\n".format(
                rank=ranking.get('rank', 'N/A'),
                model=ranking.get('model', 'N/A'),
                cs=ranking.get('composite_score', 'N/A'),
                r=ranking.get('readability_score', 0) or 0,
                a=ranking.get('actionability_score', 0) or 0,
                acc=ranking.get('accuracy_rate', 0) or 0,
                comp=ranking.get('completeness_score', 0) or 0
            )

        report += "\n---\n\n## Detailed Evaluation Metrics\n\n"

        for model_key, eval_data in self.evaluations.items():
            report += f"### {eval_data.get('model', model_key)}\n\n"
            report += f"**Word Count:** {eval_data.get('word_count', 'N/A')}\n\n"
            report += f"**Composite Quality Score:** {eval_data.get('composite_quality_score', 'N/A')}/100\n\n"

            rd = eval_data.get('readability', {})
            report += "#### Readability Metrics\n"
            report += f"- Flesch Reading Ease: {rd.get('flesch_reading_ease', 'N/A')}\n"
            report += f"- Flesch-Kincaid Grade: {rd.get('flesch_kincaid_grade', 'N/A')}\n"
            report += f"- Assessment: {rd.get('readability_assessment', 'N/A')}\n\n"

            act = eval_data.get('actionability', {})
            report += "#### Actionability Metrics\n"
            report += f"- Actionability Score: {act.get('actionability_score', 'N/A')}/100\n"
            report += f"- Action Verbs: {act.get('action_verbs_count', 'N/A')}\n"
            report += f"- Specific Metrics: {act.get('specific_metrics_count', 'N/A')}\n"
            report += f"- Timeframes Mentioned: {act.get('timeframes_mentioned', 'N/A')}\n"
            report += f"- Has Recommendation Section: {'Yes' if act.get('has_recommendation_section') else 'No'}\n\n"

            sa = eval_data.get('statistical_accuracy', {})
            report += "#### Statistical Accuracy\n"
            report += f"- Accuracy Rate: {sa.get('accuracy_rate', 'N/A')}%\n"
            report += f"- Matched Numbers: {sa.get('matched_numbers', 'N/A')}/{sa.get('total_numbers_in_narrative', 'N/A')}\n"
            report += f"- Assessment: {sa.get('assessment', 'N/A')}\n\n"

            comp = eval_data.get('completeness', {})
            report += "#### Completeness\n"
            report += f"- Completeness Score: {comp.get('completeness_score', 'N/A')}%\n"
            report += f"- Covered Themes: {', '.join(comp.get('covered_themes', [])) if comp.get('covered_themes') else 'None'}\n"
            report += f"- Missing Themes: {', '.join(comp.get('missing_themes', [])) if comp.get('missing_themes') else 'None'}\n"
            report += f"- Assessment: {comp.get('assessment', 'N/A')}\n\n"

            if 'llm_judge_scores' in eval_data and not eval_data.get('llm_judge_scores', {}).get('error'):
                scores = eval_data['llm_judge_scores']
                report += "#### LLM Judge Evaluation\n"
                report += f"- Business Relevance: {scores.get('business_relevance', 'N/A')}/5\n"
                report += f"- Executive Readiness: {scores.get('executive_readiness', 'N/A')}/5\n"
                report += f"- Narrative Flow: {scores.get('narrative_flow', 'N/A')}/5\n"
                report += f"- Persuasiveness: {scores.get('persuasiveness', 'N/A')}/5\n"
                report += f"- Average Score: {scores.get('average_score', 'N/A')}/5\n"
                if 'overall_assessment' in scores:
                    report += f"- Overall Assessment: {scores['overall_assessment']}\n"
                report += "\n"

            report += "---\n\n"

        # Save comparison report
        report_path = os.path.join(self.output_dir, f"comparison_report_{self.timestamp}.md")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📊 Comparison report generated: {report_path}\n")
        except Exception:
            print("   ⚠️ Failed to save comparison report file.")

        return report

    def run_complete_pipeline(self, data_path: str, report_type: str = 'executive_report',
                              context: Dict = None) -> Dict:
        """
        Execute the complete end-to-end pipeline.

        Args:
            data_path: Path to sales data CSV
            report_type: Type of report to generate
            context: Additional business context

        Returns:
            Complete pipeline results
        """
        print("=" * 80)
        print("🚀 STARTING COMPLETE GENAI DATA STORYTELLING PIPELINE")
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

        # Step 7: Generate Comparison Report (guarded)
        try:
            comparison_report = self.generate_comparison_report()
        except Exception as e:
            print(f"⚠️ Comparison report generation skipped due to error: {e}\n")
            comparison_report = eval_results.get('comparison', {})

        print("=" * 80)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print("=" * 80 + "\n")

        # Safe generation summary
        if self.narrative_generator and hasattr(self.narrative_generator, 'get_generation_summary'):
            try:
                gen_summary = self.narrative_generator.get_generation_summary()
            except Exception:
                gen_summary = {}
        else:
            # Derive basic summary from narratives dict
            total = len(self.narratives)
            successful = sum(1 for v in self.narratives.values() if v.get('success'))
            failed = total - successful
            gen_summary = {
                'total_models': total,
                'successful_generations': successful,
                'failed_generations': failed,
                'average_generation_time': None,
                'fastest_model': None,
                'best_overall_model': None
            }

        print("📊 Pipeline Summary:")
        print(f"   - Total models attempted: {gen_summary.get('total_models', 0)}")
        print(f"   - Successful generations: {gen_summary.get('successful_generations', 0)}")
        print(f"   - Failed generations: {gen_summary.get('failed_generations', 0)}")

        if gen_summary.get('successful_generations', 0) > 0:
            print(f"   - Average generation time: {gen_summary.get('average_generation_time', 0)}s")
            print(f"   - Fastest model: {gen_summary.get('fastest_model', 'N/A')}")
            try:
                best_overall = eval_results.get('comparison', {}).get('best_overall', 'N/A')
            except Exception:
                best_overall = 'N/A'
            print(f"   - Best overall model: {best_overall}")

        print(f"\n📁 All outputs saved to: {self.output_dir}/\n")

        return {
            'insights': self.insights,
            'structured_insights': self.structured_insights,
            'narratives': self.narratives,
            'evaluations': eval_results,
            'comparison_report': comparison_report,
            'generation_summary': gen_summary
        }