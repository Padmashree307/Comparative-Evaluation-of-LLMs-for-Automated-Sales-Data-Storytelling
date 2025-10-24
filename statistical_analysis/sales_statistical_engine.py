import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class SalesStatisticalEngine:
    """
    Comprehensive statistical analysis engine for sales data.
    Extracts KPIs, trends, correlations, segments, and anomalies.
    """
    
    def __init__(self, df):
        self.df = df
        self.insights = {}
        
    def extract_kpis(self):
        """Calculate key sales performance indicators"""
        self.insights['KPIs'] = {
            'Total Revenue': round(self.df['sales_amount'].sum(), 2),
            'Total Orders': int(len(self.df)),
            'Average Order Value': round(self.df['sales_amount'].mean(), 2),
            'Total Units Sold': int(self.df['quantity'].sum()),
            'Unique Customers': int(self.df['customer_id'].nunique()),
            'Average Revenue Per Customer': round(
                self.df.groupby('customer_id')['sales_amount'].sum().mean(), 2
            ),
            'Total Products Sold': int(self.df['product_id'].nunique())
        }
        return self.insights['KPIs']
    
    def analyze_trends(self):
        """Identify temporal sales trends and patterns"""
        # Convert to datetime if not already
        self.df['order_date'] = pd.to_datetime(self.df['order_date'])
        
        # Monthly trends
        monthly_sales = self.df.groupby(
            self.df['order_date'].dt.to_period('M')
        )['sales_amount'].sum()
        
        # Calculate growth rates
        growth_rates = monthly_sales.pct_change() * 100
        
        # Trend analysis using linear regression
        X = np.arange(len(monthly_sales)).reshape(-1, 1)
        y = monthly_sales.values
        model = LinearRegression().fit(X, y)
        trend_slope = model.coef_[0]
        
        self.insights['Trends'] = {
            'Monthly Sales': monthly_sales.to_dict(),
            'Average Monthly Growth (%)': round(growth_rates.mean(), 2),
            'Trend Direction': 'Upward' if trend_slope > 0 else 'Downward',
            'Trend Strength': round(abs(trend_slope), 2),
            'Best Month': str(monthly_sales.idxmax()),
            'Worst Month': str(monthly_sales.idxmin()),
            'Peak Sales': round(monthly_sales.max(), 2),
            'Lowest Sales': round(monthly_sales.min(), 2)
        }
        return self.insights['Trends']
    
    def analyze_product_performance(self):
        """Analyze performance by product category and individual products"""
        # Category analysis
        category_perf = self.df.groupby('product_category').agg({
            'sales_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum'
        }).round(2)
        
        category_perf.columns = ['_'.join(col) for col in category_perf.columns]
        
        # Top products
        top_products = self.df.groupby('product_name')['sales_amount'].sum(
        ).nlargest(5)
        
        self.insights['Product Performance'] = {
            'Top Categories': category_perf.nlargest(
                3, 'sales_amount_sum'
            ).to_dict(),
            'Top 5 Products': top_products.to_dict(),
            'Category Distribution': self.df.groupby(
                'product_category'
            )['sales_amount'].sum().to_dict()
        }
        return self.insights['Product Performance']
    
    def analyze_customer_segments(self):
        """Customer segmentation using RFM and clustering"""
        # RFM Analysis
        snapshot_date = self.df['order_date'].max() + pd.Timedelta(days=1)
        
        rfm = self.df.groupby('customer_id').agg({
            'order_date': lambda x: (snapshot_date - x.max()).days,
            'order_id': 'count',
            'sales_amount': 'sum'
        }).rename(columns={
            'order_date': 'Recency',
            'order_id': 'Frequency',
            'sales_amount': 'Monetary'
        })
        
        # RFM scoring
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
        rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
        
        # Segment classification
        def classify_segment(score):
            if score >= 10:
                return 'Champions'
            elif score >= 8:
                return 'Loyal Customers'
            elif score >= 6:
                return 'Potential Loyalists'
            elif score >= 4:
                return 'At Risk'
            else:
                return 'Lost Customers'
        
        rfm['Segment'] = rfm['RFM_Score'].apply(classify_segment)
        
        segment_summary = rfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']
        }).round(2)
        
        self.insights['Customer Segmentation'] = {
            'Segment Distribution': rfm['Segment'].value_counts().to_dict(),
            'Segment Metrics': segment_summary.to_dict(),
            'High Value Customers (%)': round(
                (rfm['Segment'] == 'Champions').sum() / len(rfm) * 100, 2
            ),
            'At Risk Customers (%)': round(
                (rfm['Segment'] == 'At Risk').sum() / len(rfm) * 100, 2
            )
        }
        return self.insights['Customer Segmentation']
    
    def analyze_regional_performance(self):
        """Analyze sales by geographic regions"""
        if 'region' in self.df.columns:
            regional = self.df.groupby('region').agg({
                'sales_amount': ['sum', 'mean'],
                'order_id': 'count',
                'customer_id': 'nunique'
            }).round(2)
            
            regional.columns = ['_'.join(col) for col in regional.columns]
            
            self.insights['Regional Performance'] = {
                'Total Sales by Region': regional['sales_amount_sum'].to_dict(),
                'Top Performing Region': regional['sales_amount_sum'].idxmax(),
                'Regional Distribution': regional.to_dict()
            }
        else:
            self.insights['Regional Performance'] = {'message': 'No regional data available'}
        
        return self.insights['Regional Performance']
    
    def detect_anomalies(self):
        """Identify unusual patterns and outliers"""
        # Daily sales analysis
        daily_sales = self.df.groupby('order_date')['sales_amount'].sum()
        
        # Statistical anomaly detection using Z-score
        mean_sales = daily_sales.mean()
        std_sales = daily_sales.std()
        z_scores = np.abs((daily_sales - mean_sales) / std_sales)
        
        anomalies = daily_sales[z_scores > 2]
        
        self.insights['Anomalies'] = {
            'Unusual Sales Days': len(anomalies),
            'Anomaly Dates': anomalies.to_dict(),
            'Highest Spike': {
                'Date': str(daily_sales.idxmax()),
                'Amount': round(daily_sales.max(), 2)
            }
        }
        return self.insights['Anomalies']
    
    def generate_full_analysis(self):
        """Run complete statistical analysis pipeline"""
        self.extract_kpis()
        self.analyze_trends()
        self.analyze_product_performance()
        self.analyze_customer_segments()
        self.analyze_regional_performance()
        self.detect_anomalies()
        
        return self.insights
    