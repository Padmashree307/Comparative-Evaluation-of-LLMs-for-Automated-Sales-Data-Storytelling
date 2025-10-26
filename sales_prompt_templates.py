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
You are a Senior Sales Analytics Director preparing a quarterly business review for the C-suite.

**Your Role & Audience:**
- Role: Chief Revenue Officer's trusted analytics advisor
- Audience: CEO, CFO, CMO, Board Members
- Objective: Drive strategic decisions on sales optimization and revenue growth
- Tone: Professional, data-driven, action-oriented

**Business Context:**
- Industry: {industry}
- Fiscal Period: {period}
- Strategic Priorities: Revenue growth, customer retention, market expansion

**Data Insights:**
{insights}

**Report Structure:**
Generate a comprehensive executive report with these exact sections:

## Executive Summary
- Lead with the single most critical finding (1-2 sentences)
- 3-4 key performance highlights with specific numbers
- Primary strategic recommendation

## Revenue Performance Analysis
- Total revenue achievement vs. targets
- Revenue trend analysis with growth rates
- Product/category performance breakdown
- Geographic/regional performance patterns

## Customer Insights & Behavior
- Customer segmentation analysis (Champions, Loyal, At-Risk, Lost)
- Customer lifetime value trends
- Retention and churn indicators
- High-value customer characteristics

## Growth Opportunities
- Top 3 revenue expansion opportunities with quantified potential
- Underperforming areas with recovery potential
- Market trends to capitalize on

## Risk Assessment
- Revenue risks and declining trends
- Customer churn vulnerabilities
- Operational inefficiencies impacting sales

## Strategic Recommendations
Provide 5-7 prioritized, SMART recommendations:
- Specific action items with clear owners
- Measurable success metrics
- Achievable timeframes
- Relevant to current business context
- Time-bound with implementation phases

**Output Requirements:**
- Use markdown formatting with clear headers
- Include specific numbers, percentages, and dollar amounts
- Write in active voice with executive-appropriate language
- Each insight must be backed by data from the provided analysis
- Recommendations must be actionable within 30-90 days
- Target reading level: Executive (Flesch Reading Ease 60-70)
- Length: 800-1200 words
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
    
    def generate_prompt(self, template_type: str, insights: Dict, context: Dict = None) -> str:
        """Generate domain-specific prompts with readability optimization."""
        
        if template_type == 'executive_report':
            prompt = f"""
        SALES EXECUTIVE REPORT GENERATION

        Your task: Create a clear, actionable sales report for business executives.

        AUDIENCE: Busy executives who need insights quickly
        GOAL: Explain what happened, why it matters, and what to do next

        WRITING RULES (VERY IMPORTANT - FOLLOW EXACTLY):
        1. Use simple words. NO jargon or technical terms.
        2. Keep sentences SHORT. Never more than 15 words per sentence.
        3. One main idea per sentence.
        4. Use bullet points when listing things.
        5. Always use active voice: "We increased sales" NOT "Sales were increased"
        6. Always explain numbers with context: "$5M is up 20% from last quarter"

        REPORT STRUCTURE (FOLLOW EXACTLY):
        1. HEADLINE - Your #1 finding in ONE sentence
        2. KEY METRICS - Top 5 numbers with clear explanations
        3. WHAT HAPPENED - 5-7 bullet points explaining the data
        4. WHY IT MATTERS - Explain the business impact in simple terms
        5. WHAT TO DO - 3-5 specific recommendations

        DATA TO ANALYZE:
        {json.dumps(insights, indent=2)}

        BEFORE YOU OUTPUT - CHECK EVERY SENTENCE:
        ☐ Each sentence has FEWER than 15 words
        ☐ Each number is explained (e.g., "$2M up 15%")
        ☐ Uses active voice throughout (We/Our/The company)
        ☐ No passive sentences like "It was noted that..."
        ☐ Every number comes from the data provided
        ☐ Each recommendation is specific and actionable
        ☐ A 10th grader could understand every sentence

        EXAMPLE OF EXCELLENT OUTPUT:

        Sales Hit $5M. That's up 20% from last quarter.

        Why?
        • We got 15 new enterprise customers
        • Average deal size grew from $50K to $62K
        • Customer retention improved to 85%

        What We Should Do:
        1. Hire 5 more enterprise sales reps immediately
        2. Build custom features for top 10 customers
        3. Create a retention bonus program for key clients

        ---

        Now generate the report following all these rules:
        """
            return prompt
            
            # For other template types, return default
            return f"Generate a {template_type} based on: {json.dumps(insights, indent=2)}"

    
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
