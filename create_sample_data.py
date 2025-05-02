# create_sample_data.py
# Create a subset of data for testing the EarningsWhisperer pipeline

import pandas as pd
import numpy as np
import os

def main():
    print("Creating sample data for EarningsWhisperer...")
    
    # Create directories
    os.makedirs('data/raw/stock_prices', exist_ok=True)
    os.makedirs('data/raw/earnings_dates', exist_ok=True)
    os.makedirs('data/raw/earnings_filings/sec-edgar-filings', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Sample companies (subset of the 10 in the full dataset)
    companies = ['AAPL', 'MSFT', 'GOOGL']
    
    print("Generating stock price data...")
    # Create sample stock price data
    for ticker in companies:
        # Create date range (using just 1 year instead of 4)
        dates = pd.date_range(start='2021-01-01', end='2021-12-31')
        
        # Create random price data
        np.random.seed(42)  # for reproducibility
        close = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        open_price = close - np.random.normal(0, 1, len(dates))
        high = np.maximum(close, open_price) + np.random.normal(0, 0.5, len(dates))
        low = np.minimum(close, open_price) - np.random.normal(0, 0.5, len(dates))
        volume = np.random.randint(1000000, 10000000, len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        
        # Save to CSV
        output_file = f'data/raw/stock_prices/{ticker}_prices.csv'
        df.to_csv(output_file)
        print(f"Created sample price data for {ticker} at {output_file}")
        
        # Create sample earnings dates (quarterly)
        earnings_dates = pd.date_range(start='2021-01-15', end='2021-12-15', freq='3M')
        earnings_df = pd.DataFrame({
            'Earnings Date': earnings_dates,
            'EPS Estimate': np.random.normal(1.0, 0.2, len(earnings_dates)),
            'Reported EPS': np.random.normal(1.1, 0.3, len(earnings_dates)),
            'Surprise(%)': np.random.normal(5, 10, len(earnings_dates))
        }, index=earnings_dates)
        
        output_file = f'data/raw/earnings_dates/{ticker}_earnings_dates.csv'
        earnings_df.to_csv(output_file)
        print(f"Created sample earnings dates for {ticker} at {output_file}")
        
        # Create sample SEC filings directory structure
        print(f"Creating sample SEC filings for {ticker}...")
        
        # Create sample filings for each quarter
        for i, date in enumerate(earnings_dates):
            quarter = i + 1
            filing_dir = f'data/raw/earnings_filings/sec-edgar-filings/{ticker}/8-K/filing_{quarter}'
            os.makedirs(filing_dir, exist_ok=True)
            
            # Create a sample filing text file
            date_str = date.strftime('%Y%m%d')
            
            with open(f'{filing_dir}/filing.txt', 'w') as f:
                f.write(f'<SEC-DOCUMENT>123456-21-123456.txt : {date_str}\n')
                f.write(f'COMPANY: {ticker}\n')
                f.write(f'FORM 8-K\n\n')
                
                # Add some positive and negative words for sentiment analysis
                sentiment = np.random.choice(['positive', 'neutral', 'negative'])
                if sentiment == 'positive':
                    f.write('We are pleased to report strong growth in our quarterly results.\n')
                    f.write('Revenue increased significantly and exceeded our expectations.\n')
                    f.write('Profits hit record levels.\n')
                elif sentiment == 'neutral':
                    f.write('Our quarterly results were in line with expectations.\n')
                    f.write('Revenue remained stable compared to the previous quarter.\n')
                    f.write('Profits were consistent with our forecasts.\n')
                else:
                    f.write('We faced challenging market conditions this quarter.\n')
                    f.write('Revenue declined due to decreased demand.\n')
                    f.write('Profits fell below our expectations.\n')
                    
            print(f"Created sample SEC filing for {ticker} Q{quarter}")
    
    # Create S&P 500 data for comparison
    print("Creating sample S&P 500 data...")
    dates = pd.date_range(start='2021-01-01', end='2021-12-31')
    np.random.seed(100)  # different seed for S&P 500
    close = 3700 + np.cumsum(np.random.normal(0, 10, len(dates)))
    open_price = close - np.random.normal(0, 5, len(dates))
    high = np.maximum(close, open_price) + np.random.normal(0, 3, len(dates))
    low = np.minimum(close, open_price) - np.random.normal(0, 3, len(dates))
    volume = np.random.randint(2000000000, 5000000000, len(dates))
    
    sp500_df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    output_file = 'data/raw/stock_prices/SP500_prices.csv'
    sp500_df.to_csv(output_file)
    print(f"Created sample S&P 500 data at {output_file}")
    
    print("Sample data creation complete! You can now proceed with data processing.")

if __name__ == "__main__":
    main()