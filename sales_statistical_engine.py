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
        self.df = df.copy()  # Work with a copy to avoid modifying original
        self.insights = {}
        
    def extract_kpis(self):
        """Calculate key sales performance indicators"""
        self.insights['KPIs'] = {
            'Total Revenue': float(round(self.df['sales_amount'].sum(), 2)),
            'Total Orders': int(len(self.df)),
            'Average Order Value': float(round(self.df['sales_amount'].mean(), 2)),
            'Total Units Sold': int(self.df['quantity'].sum()),
            'Unique Customers': int(self.df['customer_id'].nunique()),
            'Average Revenue Per Customer': float(round(
                self.df.groupby('customer_id')['sales_amount'].sum().mean(), 2
            )),
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
        
        # Convert Period index to string for JSON serialization
        monthly_sales_dict = {str(k): float(v) for k, v in monthly_sales.items()}
        
        self.insights['Trends'] = {
            'Monthly Sales': monthly_sales_dict,
            'Average Monthly Growth (%)': float(round(growth_rates.mean(), 2)),
            'Trend Direction': 'Upward' if trend_slope > 0 else 'Downward',
            'Trend Strength': float(round(abs(trend_slope), 2)),
            'Best Month': str(monthly_sales.idxmax()),
            'Worst Month': str(monthly_sales.idxmin()),
            'Peak Sales': float(round(monthly_sales.max(), 2)),
            'Lowest Sales': float(round(monthly_sales.min(), 2))
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
        top_products = self.df.groupby('product_name')['sales_amount'].sum().nlargest(5)
        
        # Convert to JSON-serializable format
        category_dict = {}
        for category in category_perf.index:
            category_dict[category] = {
                'Total Sales': float(category_perf.loc[category, 'sales_amount_sum']),
                'Average Sale': float(category_perf.loc[category, 'sales_amount_mean']),
                'Order Count': int(category_perf.loc[category, 'sales_amount_count']),
                'Units Sold': int(category_perf.loc[category, 'quantity_sum'])
            }
        
        self.insights['Product Performance'] = {
            'Category Breakdown': category_dict,
            'Top 5 Products': {str(k): float(v) for k, v in top_products.items()},
            'Total Categories': int(self.df['product_category'].nunique())
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
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4], duplicates='drop')
        
        # Convert to numeric for summing
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
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
        
        # Convert to JSON-serializable format
        segment_dict = {}
        for segment in segment_summary.index:
            segment_dict[segment] = {
                'Average Recency (days)': float(segment_summary.loc[segment, ('Recency', 'mean')]),
                'Average Frequency': float(segment_summary.loc[segment, ('Frequency', 'mean')]),
                'Average Monetary': float(segment_summary.loc[segment, ('Monetary', 'mean')]),
                'Customer Count': int(segment_summary.loc[segment, ('Monetary', 'count')])
            }
        
        self.insights['Customer Segmentation'] = {
            'Segment Breakdown': segment_dict,
            'Segment Distribution': rfm['Segment'].value_counts().to_dict(),
            'High Value Customers (%)': float(round(
                (rfm['Segment'] == 'Champions').sum() / len(rfm) * 100, 2
            )),
            'At Risk Customers (%)': float(round(
                (rfm['Segment'] == 'At Risk').sum() / len(rfm) * 100, 2
            )),
            'Total Customers Analyzed': int(len(rfm))
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
            
            # Convert to JSON-serializable format
            regional_dict = {}
            for region in regional.index:
                regional_dict[region] = {
                    'Total Sales': float(regional.loc[region, 'sales_amount_sum']),
                    'Average Sale': float(regional.loc[region, 'sales_amount_mean']),
                    'Order Count': int(regional.loc[region, 'order_id_count']),
                    'Unique Customers': int(regional.loc[region, 'customer_id_nunique'])
                }
            
            self.insights['Regional Performance'] = {
                'Regional Breakdown': regional_dict,
                'Top Performing Region': str(regional['sales_amount_sum'].idxmax()),
                'Total Regions': int(self.df['region'].nunique())
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
        
        # Convert to JSON-serializable format
        anomaly_dict = {str(k): float(v) for k, v in anomalies.items()}
        
        self.insights['Anomalies'] = {
            'Unusual Sales Days': int(len(anomalies)),
            'Anomaly Dates': anomaly_dict,
            'Highest Spike': {
                'Date': str(daily_sales.idxmax()),
                'Amount': float(round(daily_sales.max(), 2))
            },
            'Average Daily Sales': float(round(mean_sales, 2)),
            'Standard Deviation': float(round(std_sales, 2))
        }
        return self.insights['Anomalies']
    
    def generate_full_analysis(self):
        """Run complete statistical analysis pipeline"""
        analysis = {}

        print("   → Extracting KPIs...")
        self.extract_kpis()
        
        print("   → Analyzing trends...")
        self.analyze_trends()
        
        print("   → Analyzing product performance...")
        self.analyze_product_performance()
        
        print("   → Segmenting customers...")
        self.analyze_customer_segments()
        
        print("   → Analyzing regional performance...")
        self.analyze_regional_performance()
        
        print("   → Detecting anomalies...")
        self.detect_anomalies()
        
        return self.insights

        # Add verification metrics for accuracy checking
        analysis['verification_metrics'] = {
            'total_records': len(self.data),
            'total_revenue': float(self.data['Amount'].sum()) if 'Amount' in self.data.columns else 0,
            'average_transaction': float(self.data['Amount'].mean()) if 'Amount' in self.data.columns else 0,
            'date_range_start': str(self.data['Date'].min()) if 'Date' in self.data.columns else 'N/A',
            'date_range_end': str(self.data['Date'].max()) if 'Date' in self.data.columns else 'N/A',
        }
        
        return analysis
