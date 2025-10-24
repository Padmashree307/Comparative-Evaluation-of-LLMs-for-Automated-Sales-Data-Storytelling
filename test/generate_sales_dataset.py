import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sales_dataset(n_records: int = 5000, 
                          start_date: str = '2024-01-01',
                          end_date: str = '2024-12-31') -> pd.DataFrame:
    """
    Generate realistic sales dataset for research testing.
    
    Args:
        n_records: Number of sales records to generate
        start_date: Start date for sales data
        end_date: End date for sales data
        
    Returns:
        DataFrame with sales data
    """
    np.random.seed(42)
    random.seed(42)
    
    print(f"🔧 Generating {n_records} sales records...")
    
    # Date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end - start).days
    
    # Product catalog
    products = {
        'Electronics': ['Laptop Pro', 'Smartphone X', 'Tablet Plus', 'Smartwatch', 'Headphones'],
        'Furniture': ['Office Chair', 'Standing Desk', 'Bookshelf', 'Filing Cabinet', 'Conference Table'],
        'Software': ['CRM Suite', 'Analytics Platform', 'Project Manager', 'Security Tools', 'Cloud Storage'],
        'Office Supplies': ['Printer', 'Paper Pack', 'Pens Set', 'Notebooks', 'Organizer']
    }
    
    # Regions
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
    
    # Generate data
    data = []
    
    for i in range(n_records):
        category = random.choice(list(products.keys()))
        product = random.choice(products[category])
        
        # Generate order date with some seasonality
        days_offset = int(np.random.exponential(date_range / 4))
        days_offset = min(days_offset, date_range)
        order_date = start + timedelta(days=days_offset)
        
        # Pricing based on category
        price_ranges = {
            'Electronics': (500, 2500),
            'Furniture': (300, 1500),
            'Software': (200, 1000),
            'Office Supplies': (20, 200)
        }
        
        base_price = np.random.uniform(*price_ranges[category])
        quantity = np.random.choice([1, 2, 3, 5, 10], p=[0.5, 0.25, 0.15, 0.07, 0.03])
        
        # Add some discount variation
        discount = np.random.choice([0, 0.05, 0.10, 0.15, 0.20], p=[0.5, 0.2, 0.15, 0.10, 0.05])
        
        sales_amount = base_price * quantity * (1 - discount)
        
        data.append({
            'order_id': f'ORD-{i+1:06d}',
            'order_date': order_date.strftime('%Y-%m-%d'),
            'customer_id': f'CUST-{np.random.randint(1, n_records//5):05d}',
            'product_id': f'PROD-{hash(product) % 10000:04d}',
            'product_name': product,
            'product_category': category,
            'quantity': quantity,
            'unit_price': round(base_price, 2),
            'discount': discount,
            'sales_amount': round(sales_amount, 2),
            'region': random.choice(regions)
        })
    
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('order_date').reset_index(drop=True)
    
    print(f"   ✓ Generated dataset with shape {df.shape}")
    print(f"   ✓ Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    print(f"   ✓ Total revenue: ${df['sales_amount'].sum():,.2f}")
    print(f"   ✓ Unique customers: {df['customer_id'].nunique()}")
    
    return df

if __name__ == "__main__":
    # Generate and save dataset
    df = generate_sales_dataset(n_records=5000)
    df.to_csv('sales_data_2024.csv', index=False)
    print(f"\n   💾 Dataset saved to 'sales_data_2024.csv'\n")
