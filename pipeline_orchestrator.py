import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import os

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
        
        Args:
            api_keys: Dictionary containing API keys for all LLMs
            output_dir: Directory to save outputs
        """
        self.api_keys = api_keys
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Pipeline components
        self.statistical_engine = None
        self.insight_structurer = None
        self.prompt_engineer = SalesPromptEngineer()
        self.narrative_generator = MultiLLMNarrativeGenerator(api_keys)
        self.quality_evaluator = None
        
        # Data storage
        self.raw_data = None
        self.insights = None
        self.structured_insights = None
        self.prompts = {}
        self.narratives = {}
        self.evaluations = {}
        
        print("🚀 GenAI Data Storytelling Pipeline Initialized")
        print(f"📁 Output Directory: {output_dir}\n")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load sales dataset.
        
        Args:
            data_path: Path to sales CSV file
            
        Returns:
            Loaded DataFrame
        """
        print(f"📂 Loading data from {data_path}...")
        self.raw_data = pd.read_csv(data_path)
        print(f"   ✓ Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns\n")
        return self.raw_data
    
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
        with open(insights_path, 'w') as f:
            json.dump(self.insights, f, indent=2, default=str)
        print(f"   💾 Insights saved to {insights_path}\n")
        
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
        structured_path = os.path.join(
            self.output_dir, 
            f"structured_insights_{self.timestamp}.json"
        )
        with open(structured_path, 'w') as f:
            json.dump(self.structured_insights, f, indent=2, default=str)
        print(f"   💾 Structured insights saved to {structured_path}\n")
        
        return self.structured_insights
    
    def generate_prompts(self, report_types: List[str] = None, 
                        context: Dict = None) -> Dict[str, str]:
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
        
        Args:
            prompt_type: Type of prompt to use for generation
            
        Returns:
            Dictionary of narratives from all models
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Prompt type '{prompt_type}' not generated yet")
        
        prompt = self.prompts[prompt_type]
        
        print(f"🤖 Generating Narratives for {prompt_type}...\n")
        self.narratives = self.narrative_generator.generate_all_narratives(prompt)
        
        # Save narratives
        for model_key, narrative_data in self.narratives.items():
            if narrative_data['success']:
                narrative_path = os.path.join(
                    self.output_dir,
                    f"narrative_{model_key}_{self.timestamp}.md"
                )
                with open(narrative_path, 'w') as f:
                    f.write(f"# {narrative_data['model']} - {prompt_type}\n\n")
                    f.write(f"**Generated:** {narrative_data['timestamp']}\n\n")
                    f.write(f"**Generation Time:** {narrative_data['generation_time']}s\n\n")
                    f.write("---\n\n")
                    f.write(narrative_data['narrative'])
                
                print(f"   💾 Saved {model_key} narrative to {narrative_path}")
        
        print("\n")
        return self.narratives
    
    def evaluate_narratives(self) -> Dict:
        """
        Evaluate all generated narratives.
        
        Returns:
            Dictionary of evaluation results
        """
        print("📈 Evaluating Narrative Quality...\n")
        
        # Initialize evaluator with Gemini as judge
        gemini_client = self.narrative_generator.clients.get('gemini')
        self.quality_evaluator = NarrativeQualityEvaluator(
            ground_truth_insights=self.insights,
            llm_judge_client=gemini_client
        )
        
        # Evaluate each narrative
        for model_key, narrative_data in self.narratives.items():
            if narrative_data['success']:
                evaluation = self.quality_evaluator.evaluate_narrative(
                    narrative=narrative_data['narrative'],
                    model_name=narrative_data['model']
                )
                self.evaluations[model_key] = evaluation
        
        # Generate comparison
        comparison = self.quality_evaluator.compare_models()
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, f"evaluations_{self.timestamp}.json")
        with open(eval_path, 'w') as f:
            json.dump({
                'individual_evaluations': self.evaluations,
                'model_comparison': comparison
            }, f, indent=2, default=str)
        
        print(f"   💾 Evaluation results saved to {eval_path}\n")
        
        return {
            'evaluations': self.evaluations,
            'comparison': comparison
        }
    
    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        
        Returns:
            Markdown formatted comparison report
        """
        if not self.evaluations:
            raise ValueError("No evaluations available. Run evaluate_narratives() first.")
        
        comparison = self.quality_evaluator.compare_models()
        
        report = f"""# GenAI Data Storytelling Pipeline - Model Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Type:** Sales Performance Analytics
**Models Evaluated:** {len(self.evaluations)}

---

## Executive Summary

**Best Overall Model:** {comparison['best_overall']}
**Best Readability:** {comparison['best_readability']}
**Best Actionability:** {comparison['best_actionability']}
**Best Accuracy:** {comparison['best_accuracy']}

---

## Model Rankings

| Rank | Model | Composite Score | Readability | Actionability | Accuracy | Completeness |
|------|-------|----------------|-------------|---------------|----------|--------------|
"""
        
        for ranking in comparison['rankings']:
            report += f"| {ranking['rank']} | {ranking['model']} | {ranking['composite_score']} | {ranking['readability_score']} | {ranking['actionability_score']} | {ranking['accuracy_rate']}% | {ranking['completeness_score']}% |\n"
        
        report += "\n---\n\n## Detailed Evaluation Metrics\n\n"
        
        for model_key, eval_data in self.evaluations.items():
            report += f"### {eval_data['model']}\n\n"
            report += f"**Word Count:** {eval_data['word_count']}\n\n"
            report += f"**Composite Quality Score:** {eval_data['composite_quality_score']}/100\n\n"
            
            report += "#### Readability Metrics\n"
            report += f"- Flesch Reading Ease: {eval_data['readability']['flesch_reading_ease']}\n"
            report += f"- Assessment: {eval_data['readability']['readability_assessment']}\n\n"
            
            report += "#### Actionability Metrics\n"
            report += f"- Actionability Score: {eval_data['actionability']['actionability_score']}/100\n"
            report += f"- Action Verbs: {eval_data['actionability']['action_verbs_count']}\n"
            report += f"- Specific Metrics: {eval_data['actionability']['specific_metrics_count']}\n"
            report += f"- Timeframes: {eval_data['actionability']['timeframes_mentioned']}\n\n"
            
            report += "#### Statistical Accuracy\n"
            report += f"- Accuracy Rate: {eval_data['statistical_accuracy']['accuracy_rate']}%\n"
            report += f"- Matched Numbers: {eval_data['statistical_accuracy']['matched_numbers']}/{eval_data['statistical_accuracy']['total_numbers_in_narrative']}\n\n"
            
            report += "#### Completeness\n"
            report += f"- Completeness Score: {eval_data['completeness']['completeness_score']}%\n"
            report += f"- Covered Themes: {', '.join(eval_data['completeness']['covered_themes'])}\n\n"
            
            if 'llm_judge_scores' in eval_data:
                report += "#### LLM Judge Evaluation\n"
                report += f"- Business Relevance: {eval_data['llm_judge_scores'].get('business_relevance', 'N/A')}/5\n"
                report += f"- Executive Readiness: {eval_data['llm_judge_scores'].get('executive_readiness', 'N/A')}/5\n"
                report += f"- Narrative Flow: {eval_data['llm_judge_scores'].get('narrative_flow', 'N/A')}/5\n"
                report += f"- Persuasiveness: {eval_data['llm_judge_scores'].get('persuasiveness', 'N/A')}/5\n\n"
            
            report += "---\n\n"
        
        # Save comparison report
        report_path = os.path.join(self.output_dir, f"comparison_report_{self.timestamp}.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📊 Comparison report generated: {report_path}\n")
        
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
        
        # Step 7: Generate Comparison Report
        comparison_report = self.generate_comparison_report()
        
        print("=" * 80)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print("=" * 80 + "\n")
        
        # Print summary
        gen_summary = self.narrative_generator.get_generation_summary()
        print("📊 Generation Summary:")
        print(f"   - Models used: {gen_summary['successful_generations']}")
        print(f"   - Average generation time: {gen_summary['average_generation_time']}s")
        print(f"   - Fastest model: {gen_summary['fastest_model']}")
        print(f"   - Best overall model: {eval_results['comparison']['best_overall']}")
        print(f"\n📁 All outputs saved to: {self.output_dir}/\n")
        
        return {
            'insights': self.insights,
            'structured_insights': self.structured_insights,
            'narratives': self.narratives,
            'evaluations': eval_results,
            'comparison_report': comparison_report,
            'generation_summary': gen_summary
        }
