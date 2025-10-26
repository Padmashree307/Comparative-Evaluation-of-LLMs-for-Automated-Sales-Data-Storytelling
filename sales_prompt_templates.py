from typing import Dict
import json

class SalesPromptEngineer:
    """
    Domain-specific prompt templates for sales business narratives.
    Implements context injection, constraint specification, and format requirements.
    """
    
    def __init__(self):
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all domain-specific prompt templates"""
        
        # Executive Report Template
        self.templates['executive_report'] = """
You are a Senior Sales Analytics Director writing a clear business report.

**WRITING RULES (CRITICAL):**
- Use SHORT sentences. Maximum 15 words per sentence.
- Write like a business newspaper (clear, direct, simple).
- Explain every number with plain language, e.g., "$5M revenue means we earned 5 million dollars."
- Use transitions: "As a result," "This means," "Therefore," "In summary."
- Use active voice: "We increased sales" NOT "Sales were increased."
- Break complex ideas into bullet points.
- Use simple words: "use" not "utilize," "help" not "facilitate."

**STRUCTURE (FOLLOW EXACTLY):**

### Executive Summary
- 2-3 key findings in simple language

### Financial Performance  
- Key revenue metrics
- Growth percentages with explanations

### Key Drivers & Trends
- 4-5 main reasons for performance
- Each with 1-2 sentences of explanation

### Recommendations
- 3-5 specific action items
- Each with "How to implement" step


**EXAMPLE OF EXCELLENT REPORT:**

### Executive Summary
Q2 revenue reached $8.5M, up 12% from Q1. This strong growth was driven by three key factors: new customer acquisition, higher per-customer spending, and improved retention. These results position us well for Q3.

### Financial Performance
- Total Revenue: $8.5M (up from $7.6M in Q1, a 12% increase)
- New Customer Count: 45 accounts (vs. 38 last quarter)
- Average Customer Value: $55K (up 8% from $51K)
- Retention Rate: 92% (improved from 89%)

### Key Drivers & Trends
1. Enterprise Growth: Large enterprise customers contributed $3.2M (38% of revenue), up significantly from Q1's $2.1M.
2. Product Expansion: Our new premium tier attracted customers willing to pay 25% higher prices.
3. Improved Marketing: Digital campaign ROI improved 40%, driving customer acquisition cost down 15%.
4. Customer Success: Proactive support reduced churn by 3 percentage points.

### Recommendations
1. Expand Enterprise Team: Hire 2 more account executives by end of Q3 to capitalize on enterprise demand. Budget: $180K annually.
2. Develop Premium Features: Allocate 2 engineers to build advanced features for the premium tier. Expected: +20% revenue from premium customers by Q4.
3. Scale Digital Marketing: Double ad spend in high-performing channels. Expected ROI: 3:1.

---

Data Insights:
{insights}

---

Now write the complete executive report following the structure and quality of the example above.
"""


        # Sales Team Operational Report
        self.templates['operational_report'] = """
You are a Sales Operations Manager creating an actionable report for regional sales teams.

**Your Role & Audience:**
- Role: Sales Operations Manager
- Audience: Regional Sales Managers, Sales Representatives, Sales Enablement
- Objective: Provide tactical insights to improve daily sales execution
- Tone: Practical, motivational, hands-on

**Business Context:**
- Industry: {industry}
- Reporting Period: {period}
- Focus Areas: Pipeline health, conversion rates, deal velocity

**Data Insights:**
{insights}

**Report Structure:**

## Performance Snapshot
- Team performance vs. quota
- Win rate and conversion metrics
- Average deal size and sales cycle length

## What's Working
- Top performing products/categories
- Highest converting customer segments
- Best practices from high performers

## Areas for Improvement
- Conversion bottlenecks
- Underperforming segments
- Lost opportunity analysis

## Immediate Action Items
- 3-5 specific tactics to implement this week
- Target accounts or segments to prioritize
- Process improvements to adopt

**Output Requirements:**
- Conversational yet professional tone
- Focus on actionable takeaways
- Include specific examples and numbers
- Length: 500-700 words
"""

        # Customer Success & Retention Report
        self.templates['retention_report'] = """
You are a Customer Success Director analyzing customer health and retention patterns.

**Your Role & Audience:**
- Role: VP of Customer Success
- Audience: CS Team, Account Managers, Executive Leadership
- Objective: Reduce churn, increase expansion revenue, improve customer satisfaction
- Tone: Customer-centric, analytical, proactive

**Business Context:**
- Industry: {industry}
- Analysis Period: {period}
- Key Metrics: Net Revenue Retention, Customer Lifetime Value, Churn Rate

**Data Insights:**
{insights}

**Report Structure:**

## Customer Health Overview
- Customer segmentation breakdown (Champions, Loyal, At-Risk, Lost)
- Churn risk indicators
- Customer satisfaction trends

## Revenue Retention Analysis
- Net revenue retention rate
- Expansion revenue opportunities
- Contraction and churn impact

## At-Risk Customer Analysis
- Identification of at-risk segments
- Early warning signals
- Churn prediction insights

## Customer Success Strategies
- Retention initiatives for at-risk customers
- Expansion playbooks for loyal customers
- Win-back strategies for lost customers

## Success Metrics & Goals
- Target retention rates
- Expansion revenue goals
- Customer satisfaction improvements

**Output Requirements:**
- Emphasize customer-centric language
- Include segment-specific recommendations
- Quantify potential revenue impact
- Length: 700-900 words
"""

        # Product Performance Report
        self.templates['product_report'] = """
You are a Product Marketing Manager analyzing product portfolio performance.

**Your Role & Audience:**
- Role: Senior Product Marketing Manager
- Audience: Product Management, Marketing, Sales Leadership
- Objective: Optimize product mix, identify winners, sunset underperformers
- Tone: Strategic, data-informed, market-aware

**Business Context:**
- Industry: {industry}
- Review Period: {period}
- Focus: Product-market fit, competitive positioning, portfolio optimization

**Data Insights:**
{insights}

**Report Structure:**

## Portfolio Performance Summary
- Total revenue by product category
- Top revenue-generating products
- Product performance trends

## Star Performers
- High-growth products with momentum
- Products exceeding targets
- Cross-sell and upsell opportunities

## Underperformers & Challenges
- Declining product lines
- Products missing targets
- Competitive pressures

## Market Opportunities
- Whitespace analysis
- Product gap identification
- Innovation priorities

## Strategic Recommendations
- Portfolio optimization actions
- Pricing and positioning adjustments
- Go-to-market strategy refinements

**Output Requirements:**
- Product-focused narrative
- Competitive context where relevant
- Market opportunity quantification
- Length: 600-800 words
"""
    
    def generate_prompt(self, template_type: str, insights: Dict,context: Dict = None) -> str:
        """
        Generate customized prompt with insights and context injection.
        
        Args:
            template_type: Type of report (executive_report, operational_report, etc.)
            insights: Structured insights dictionary
            context: Additional business context (industry, period, etc.)
        
        Returns:
            Formatted prompt string ready for LLM
        """
        if template_type not in self.templates:
            raise ValueError(f"Template type '{template_type}' not found")
        
        # Default context
        default_context = {
            'industry': 'Technology/SaaS',
            'period': 'Q4 2024'
        }
        
        if context:
            default_context.update(context)
        
        # Convert insights to formatted JSON string
        insights_str = json.dumps(insights, indent=2, default=str)
        
        # Format template with context and insights
        prompt = self.templates[template_type].format(
            industry=default_context['industry'],
            period=default_context['period'],
            insights=insights_str
        )
        
        return prompt
    
    def add_custom_constraints(self, prompt: str, constraints: Dict) -> str:
        """
        Add additional constraints to generated prompts.
        
        Args:
            prompt: Base prompt string
            constraints: Dictionary of additional constraints
        
        Returns:
            Enhanced prompt with constraints
        """
        constraint_text = "\n\n**Additional Constraints:**\n"
        
        for key, value in constraints.items():
            constraint_text += f"- {key}: {value}\n"
        
        return prompt + constraint_text
