import json
from typing import Dict, List

class InsightStructurer:
    """
    Structures and prioritizes statistical insights for business narratives.
    Creates hierarchical organization with impact scoring.
    """
    
    def __init__(self, insights: Dict):
        self.raw_insights = insights
        self.structured_insights = {}
        
    def calculate_impact_score(self, metric_name: str, value: float) -> int:
        """
        Calculate business impact score (1-10) based on metric type and value.
        """
        # Impact scoring logic based on metric type
        impact_map = {
            'revenue': {'high': 1000000, 'medium': 500000},
            'growth': {'high': 15, 'medium': 5},
            'churn': {'high': 20, 'medium': 10},
            'customer': {'high': 1000, 'medium': 500}
        }
        
        # Simplified scoring - enhance based on business rules
        if 'Revenue' in metric_name or 'Sales' in metric_name:
            if value > impact_map['revenue']['high']:
                return 9
            elif value > impact_map['revenue']['medium']:
                return 6
            else:
                return 4
        elif 'Growth' in metric_name or 'Trend' in metric_name:
            if abs(value) > impact_map['growth']['high']:
                return 8
            elif abs(value) > impact_map['growth']['medium']:
                return 5
            else:
                return 3
        else:
            return 5  # Default medium impact
    
    def categorize_insights(self) -> Dict:
        """
        Categorize insights into business themes.
        """
        categories = {
            'Performance': [],
            'Growth Opportunities': [],
            'Risk Factors': [],
            'Customer Behavior': [],
            'Operational Efficiency': []
        }
        
        # KPIs -> Performance
        if 'KPIs' in self.raw_insights:
            categories['Performance'].append({
                'theme': 'Overall Business Performance',
                'insights': self.raw_insights['KPIs'],
                'priority': 'High'
            })
        
        # Trends -> Growth Opportunities or Risk
        if 'Trends' in self.raw_insights:
            trend_direction = self.raw_insights['Trends'].get('Trend Direction', 'Neutral')
            if trend_direction == 'Upward':
                categories['Growth Opportunities'].append({
                    'theme': 'Positive Sales Momentum',
                    'insights': self.raw_insights['Trends'],
                    'priority': 'High'
                })
            else:
                categories['Risk Factors'].append({
                    'theme': 'Declining Sales Trend',
                    'insights': self.raw_insights['Trends'],
                    'priority': 'Critical'
                })
        
        # Customer Segmentation
        if 'Customer Segmentation' in self.raw_insights:
            categories['Customer Behavior'].append({
                'theme': 'Customer Value Distribution',
                'insights': self.raw_insights['Customer Segmentation'],
                'priority': 'High'
            })
        
        # Product Performance
        if 'Product Performance' in self.raw_insights:
            categories['Growth Opportunities'].append({
                'theme': 'Product Portfolio Optimization',
                'insights': self.raw_insights['Product Performance'],
                'priority': 'Medium'
            })
        
        self.structured_insights['categories'] = categories
        return categories
    
    def create_narrative_hierarchy(self) -> Dict:
        """
        Create hierarchical structure for narrative flow.
        """
        hierarchy = {
            'Executive Summary': {
                'key_metrics': [],
                'critical_findings': [],
                'top_recommendations': []
            },
            'Detailed Analysis': {
                'performance_deep_dive': [],
                'trend_analysis': [],
                'segment_insights': []
            },
            'Strategic Recommendations': {
                'immediate_actions': [],
                'medium_term_initiatives': [],
                'long_term_strategy': []
            }
        }
        
        # Populate Executive Summary
        if 'KPIs' in self.raw_insights:
            hierarchy['Executive Summary']['key_metrics'] = [
                f"Total Revenue: ${self.raw_insights['KPIs'].get('Total Revenue', 0):,.2f}",
                f"Total Orders: {self.raw_insights['KPIs'].get('Total Orders', 0):,}",
                f"Average Order Value: ${self.raw_insights['KPIs'].get('Average Order Value', 0):,.2f}"
            ]
        
        # Add critical findings based on priority
        categories = self.categorize_insights()
        for category, items in categories.items():
            for item in items:
                if item['priority'] in ['High', 'Critical']:
                    hierarchy['Executive Summary']['critical_findings'].append({
                        'category': category,
                        'theme': item['theme']
                    })
        
        self.structured_insights['hierarchy'] = hierarchy
        return hierarchy
    
    def generate_structured_output(self) -> Dict:
        """
        Generate complete structured insights for LLM consumption.
        """
        output = {
            'metadata': {
                'analysis_type': 'Sales Performance Analysis',
                'domain': 'Sales & Marketing',
                'insight_count': len(self.raw_insights)
            },
            'raw_insights': self.raw_insights,
            'categorized_insights': self.categorize_insights(),
            'narrative_hierarchy': self.create_narrative_hierarchy()
        }
        
        return output
