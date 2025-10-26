import re
import json
from typing import Dict, List
import textstat
from collections import Counter
import numpy as np  # ← ADD THIS
import random      # ← ADD THIS
from config import RANDOM_SEED  # ← ADD THIS

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Import for Gemini LLM judge
from google.genai import types

class NarrativeQualityEvaluator:
    """
    Comprehensive quality evaluation system for business narratives.
    Implements readability, actionability, accuracy, and completeness metrics.
    """
    
    def __init__(self, ground_truth_insights: Dict, llm_judge_client=None):
        """
        Initialize evaluator with ground truth data.
        
        Args:
            ground_truth_insights: Original statistical insights for accuracy verification
            llm_judge_client: Optional LLM client for qualitative evaluation (Gemini)
        """
        self.ground_truth = ground_truth_insights
        self.llm_judge = llm_judge_client
        self.evaluation_results = {}
    
    def calculate_readability_score(self, narrative: str) -> Dict:
        """
        Calculate multiple readability metrics.
        Target: Flesch Reading Ease 60-70 for executive audiences.
        
        Args:
            narrative: Generated narrative text
            
        Returns:
            Dictionary of readability scores
        """
        try:
            scores = {
                'flesch_reading_ease': round(textstat.flesch_reading_ease(narrative), 2),
                'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(narrative), 2),
                'gunning_fog': round(textstat.gunning_fog(narrative), 2),
                'automated_readability_index': round(textstat.automated_readability_index(narrative), 2),
                'coleman_liau_index': round(textstat.coleman_liau_index(narrative), 2)
            }
            
            # Interpretation
            fre = scores['flesch_reading_ease']
            if 60 <= fre <= 70:
                scores['readability_assessment'] = 'Excellent - Executive appropriate'
            elif 50 <= fre < 60 or 70 < fre <= 80:
                scores['readability_assessment'] = 'Good - Acceptable'
            else:
                scores['readability_assessment'] = 'Needs improvement'
            
            return scores
        except Exception as e:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'automated_readability_index': 0,
                'coleman_liau_index': 0,
                'readability_assessment': f'Error: {str(e)}'
            }
    
    def calculate_actionability_index(self, narrative: str) -> Dict:
        """
        Calculate actionability based on SMART recommendations.
        Counts action verbs, specific metrics, and timeframes.
        
        Args:
            narrative: Generated narrative text
            
        Returns:
            Dictionary with actionability metrics
        """
        # Action verb patterns
        action_verbs = [
            'implement', 'develop', 'launch', 'increase', 'reduce', 'improve',
            'optimize', 'expand', 'focus', 'prioritize', 'target', 'invest',
            'build', 'create', 'enhance', 'streamline', 'accelerate', 'establish',
            'execute', 'deploy', 'leverage', 'capitalize', 'maximize', 'minimize'
        ]
        
        # Count action verbs
        action_verb_count = sum(
            len(re.findall(rf'\b{verb}\b', narrative.lower())) 
            for verb in action_verbs
        )
        
        # Count specific metrics (numbers with % or $)
        metric_pattern = r'\$?[\d,]+\.?\d*%?|\d+%'
        metrics_count = len(re.findall(metric_pattern, narrative))
        
        # Count timeframes
        timeframe_patterns = [
            r'\d+\s*(days?|weeks?|months?|quarters?|years?)',
            r'Q\d\s*\d{4}',
            r'by\s+\d{4}',
            r'within\s+\d+',
            r'next\s+\d+',
            r'in\s+\d{4}'
        ]
        timeframe_count = sum(
            len(re.findall(pattern, narrative, re.IGNORECASE))
            for pattern in timeframe_patterns
        )
        
        # Identify recommendation sections
        rec_headers = ['recommendation', 'action', 'next steps', 'initiative', 'strategy']
        has_recommendation_section = any(
            header in narrative.lower() for header in rec_headers
        )
        
        # Calculate actionability score (0-100)
        score = min(100, (
            (action_verb_count * 3) +
            (metrics_count * 2) +
            (timeframe_count * 5) +
            (30 if has_recommendation_section else 0)
        ))
        
        return {
            'actionability_score': round(score, 2),
            'action_verbs_count': action_verb_count,
            'specific_metrics_count': metrics_count,
            'timeframes_mentioned': timeframe_count,
            'has_recommendation_section': has_recommendation_section,
            'assessment': 'High' if score >= 70 else 'Medium' if score >= 40 else 'Low'
        }
    
    def calculate_statistical_accuracy(self, narrative: str) -> Dict:
        """
        Verify numerical accuracy against ground truth insights.
        
        Args:
            narrative: Generated narrative text
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Extract all numbers from narrative
        numbers_in_narrative = re.findall(r'[\d,]+\.?\d*', narrative)
        numbers_in_narrative = [n.replace(',', '') for n in numbers_in_narrative]
        
        # Extract ground truth numbers
        def extract_numbers_from_dict(d, numbers_list=None):
            if numbers_list is None:
                numbers_list = []
            
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, (int, float)):
                        numbers_list.append(str(round(float(value), 2)))
                    elif isinstance(value, dict):
                        extract_numbers_from_dict(value, numbers_list)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, (int, float)):
                                numbers_list.append(str(round(float(item), 2)))
                            elif isinstance(item, dict):
                                extract_numbers_from_dict(item, numbers_list)
            return numbers_list
        
        ground_truth_numbers = extract_numbers_from_dict(self.ground_truth)
        
        # Calculate accuracy with tolerance for rounding
        matched_numbers = 0
        for n in numbers_in_narrative:
            try:
                n_float = float(n)
                for gt in ground_truth_numbers:
                    try:
                        gt_float = float(gt)
                        # Allow 1% tolerance for rounding
                        if abs(n_float - gt_float) / max(gt_float, 1) < 0.01:
                            matched_numbers += 1
                            break
                    except:
                        continue
            except:
                continue
        
        accuracy_rate = (
            matched_numbers / len(numbers_in_narrative) * 100 
            if numbers_in_narrative else 0
        )
        
        return {
            'accuracy_rate': round(accuracy_rate, 2),
            'total_numbers_in_narrative': len(numbers_in_narrative),
            'matched_numbers': matched_numbers,
            'assessment': 'High' if accuracy_rate >= 80 else 'Medium' if accuracy_rate >= 60 else 'Low'
        }
    
    def calculate_completeness_score(self, narrative: str) -> Dict:
        """
        Measure coverage of key insights from ground truth.
        
        Args:
            narrative: Generated narrative text
            
        Returns:
            Dictionary with completeness metrics
        """
        # Key themes to check
        required_themes = [
            'revenue', 'sales', 'customer', 'product', 'trend', 
            'growth', 'performance', 'recommendation'
        ]
        
        covered_themes = [
            theme for theme in required_themes 
            if theme in narrative.lower()
        ]
        
        completeness_percentage = (len(covered_themes) / len(required_themes)) * 100
        
        # Check for key insight categories
        has_kpis = 'revenue' in narrative.lower() or 'sales' in narrative.lower()
        has_trends = 'trend' in narrative.lower() or 'growth' in narrative.lower()
        has_segments = 'customer' in narrative.lower() or 'segment' in narrative.lower()
        has_recommendations = 'recommend' in narrative.lower() or 'should' in narrative.lower()
        
        return {
            'completeness_score': round(completeness_percentage, 2),
            'covered_themes': covered_themes,
            'missing_themes': [t for t in required_themes if t not in covered_themes],
            'has_kpis': has_kpis,
            'has_trends': has_trends,
            'has_segments': has_segments,
            'has_recommendations': has_recommendations,
            'assessment': 'Complete' if completeness_percentage >= 75 else 'Partial'
        }
    
    def evaluate_with_llm_judge(self, narrative: str) -> Dict:
        """
        Qualitative evaluation using LLM-as-judge methodology.
        Evaluates business relevance, executive readiness, and persuasiveness.
        
        Args:
            narrative: Generated narrative text
            
        Returns:
            Dictionary with qualitative scores (1-5 scale)
        """
        if not self.llm_judge:
            return {'error': 'No LLM judge client configured'}
        
        # Truncate narrative if too long
        narrative_sample = narrative[:3000] if len(narrative) > 3000 else narrative
        
        judge_prompt = f"""You are an expert business consultant evaluating the quality of a sales analytics report.

Rate the following narrative on these dimensions (1-5 scale):

**Narrative to Evaluate:**
{narrative_sample}

**Evaluation Criteria:**

1. **Business Relevance (1-5)**: Does the narrative focus on insights that matter to business decision-makers?
   - 5: Highly relevant, directly actionable business insights
   - 3: Moderately relevant, some business value
   - 1: Not relevant to business decisions

2. **Executive Readiness (1-5)**: Is the narrative appropriate for C-suite presentation?
   - 5: Polished, concise, executive-appropriate
   - 3: Acceptable but needs refinement
   - 1: Not suitable for executives

3. **Narrative Flow (1-5)**: Does the narrative have logical progression and coherence?
   - 5: Excellent flow, easy to follow
   - 3: Adequate structure
   - 1: Disjointed, hard to follow

4. **Persuasiveness (1-5)**: Does the narrative drive action and convey urgency?
   - 5: Highly persuasive, compelling call to action
   - 3: Moderately persuasive
   - 1: Not persuasive

Return ONLY a valid JSON object with numeric scores (no markdown formatting, no code blocks):
{{
    "business_relevance": <number between 1-5>,
    "executive_readiness": <number between 1-5>,
    "narrative_flow": <number between 1-5>,
    "persuasiveness": <number between 1-5>,
    "overall_assessment": "<brief 1-2 sentence comment>"
}}"""
        
        try:
            # FIXED: Correct Gemini API call with new SDK
            response = self.llm_judge.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=judge_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                    top_p=0.95
                )
            )
            
            # Clean response text
            response_text = response.text.strip()
            
            
            # Remove any leading/trailing whitespace
            response_text = response_text.strip()
            
            # Parse JSON
            scores = json.loads(response_text)
            
            # Validate scores are in range
            for key in ['business_relevance', 'executive_readiness', 'narrative_flow', 'persuasiveness']:
                if key in scores:
                    scores[key] = max(1, min(5, int(scores[key])))
            
            # Calculate average
            scores['average_score'] = round(
                (scores.get('business_relevance', 0) + 
                 scores.get('executive_readiness', 0) + 
                 scores.get('narrative_flow', 0) + 
                 scores.get('persuasiveness', 0)) / 4, 2
            )
            
            return scores
            
        except json.JSONDecodeError as e:
            return {
                'error': f'JSON parsing failed: {str(e)}',
                'raw_response': response.text if 'response' in locals() else 'No response'
            }
        except Exception as e:
            return {
                'error': f'LLM judge evaluation failed: {str(e)}',
                'error_type': type(e).__name__
            }
    
    def evaluate_narrative(self, narrative: str, model_name: str) -> Dict:
        """
        Run complete evaluation pipeline on a narrative.
        
        Args:
            narrative: Generated narrative text
            model_name: Name of the model that generated the narrative
            
        Returns:
            Complete evaluation results
        """
        print(f"\n📊 Evaluating narrative from {model_name}...")
        
        evaluation = {
            'model': model_name,
            'word_count': len(narrative.split()),
            'character_count': len(narrative),
            'readability': self.calculate_readability_score(narrative),
            'actionability': self.calculate_actionability_index(narrative),
            'statistical_accuracy': self.calculate_statistical_accuracy(narrative),
            'completeness': self.calculate_completeness_score(narrative)
        }
        
        # Optional LLM judge evaluation
        if self.llm_judge:
            print(f"   → Running LLM judge evaluation...")
            llm_scores = self.evaluate_with_llm_judge(narrative)
            if 'error' not in llm_scores:
                evaluation['llm_judge_scores'] = llm_scores
                print(f"   ✓ LLM judge completed")
            else:
                print(f"   ⚠️ LLM judge skipped: {llm_scores.get('error', 'Unknown error')}")
                evaluation['llm_judge_scores'] = llm_scores
        
        # Calculate composite quality score
        evaluation['composite_quality_score'] = self._calculate_composite_score(evaluation)
        
        self.evaluation_results[model_name] = evaluation
        
        print(f"   ✓ Composite Quality Score: {evaluation['composite_quality_score']}/100\n")
        
        return evaluation
    
    def _calculate_composite_score(self, evaluation: Dict) -> float:
        """
        Calculate weighted composite quality score (0-100).
        
        Weights:
        - Readability: 20%
        - Actionability: 30%
        - Statistical Accuracy: 25%
        - Completeness: 25%
        """
        # Normalize readability (FRE 60-70 is ideal)
        fre = evaluation['readability']['flesch_reading_ease']
        if fre == 0:  # Handle error case
            readability_normalized = 0
        else:
            readability_normalized = 100 - abs(65 - fre) * 2
            readability_normalized = max(0, min(100, readability_normalized))
        
        actionability = evaluation['actionability']['actionability_score']
        accuracy = evaluation['statistical_accuracy']['accuracy_rate']
        completeness = evaluation['completeness']['completeness_score']
        
        composite = (
            (readability_normalized * 0.20) +
            (actionability * 0.30) +
            (accuracy * 0.25) +
            (completeness * 0.25)
        )
        
        return round(composite, 2)
    
    # ...existing code...
    def compare_models(self) -> Dict:
        """
        Generate comparative analysis of all evaluated models.
        
        Returns:
            Ranking and comparison of models
        """
        if not self.evaluation_results:
            return {'error': 'No evaluations available'}
        
        # Rank by composite score (each item is (model, results))
        ranked = sorted(
            self.evaluation_results.items(),
            key=lambda item: item[1].get('composite_quality_score', 0),
            reverse=True
        )
        
        rankings = [
            {
                'rank': idx + 1,
                'model': model,
                'composite_score': results.get('composite_quality_score'),
                'readability_score': results.get('readability', {}).get('flesch_reading_ease'),
                'actionability_score': results.get('actionability', {}).get('actionability_score'),
                'accuracy_rate': results.get('statistical_accuracy', {}).get('accuracy_rate'),
                'completeness_score': results.get('completeness', {}).get('completeness_score')
            }
            for idx, (model, results) in enumerate(ranked)
        ]
        
        best_overall = ranked[0] if ranked else None
        
        best_readability = None
        if self.evaluation_results:
            best_readability = min(
                self.evaluation_results.items(),
                key=lambda item: abs(65 - item[1].get('readability', {}).get('flesch_reading_ease', 0))
            )  # returns (model, results)
        
        best_actionability = None
        if self.evaluation_results:
            best_actionability = max(
                self.evaluation_results.items(),
                key=lambda item: item[1].get('actionability', {}).get('actionability_score', 0)
            )
        
        best_accuracy = None
        if self.evaluation_results:
            best_accuracy = max(
                self.evaluation_results.items(),
                key=lambda item: item[1].get('statistical_accuracy', {}).get('accuracy_rate', 0)
            )
        
        return {
            'rankings': rankings,
            'best_overall': best_overall,
            'best_readability': best_readability,
            'best_actionability': best_actionability,
            'best_accuracy': best_accuracy
        }