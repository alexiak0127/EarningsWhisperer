# data_processing.py processes the raw data

import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import timedelta

# Import RoBERTa sentiment analysis
try:
    from enhanced_sentiment_analysis import process_sec_filings_with_roberta
    HAS_ROBERTA = False  # Force disable RoBERTa even if available
    print("RoBERTa sentiment analysis is available but disabled")
except ImportError:
    HAS_ROBERTA = False
    print("RoBERTa sentiment analysis not available, falling back to keyword-based method")

# -- Define companies --
# To ensure feasibility and relevance, 
# the project focuses on 10 major publicly traded tech companies 
# with significant market influence and high trading volume

COMPANIES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corporation',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'AMD': 'Advanced Micro Devices Inc.',
    'INTC': 'Intel Corporation',
    'CRM': 'Salesforce Inc.'
}

# < Calculate technical indicators>
# -- Relative Strength Index (RSI) --
# shows if a stock is overbought or oversold
def calculate_rsi(prices, window=14):
    # Calculate price changes
    delta = prices.diff()
    
    # Splits changes into gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the last 14 days
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # RS formula
    rs = avg_gain / avg_loss
    # RSI formula
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# -- Process raw stock price data --
def process_stock_data():
    print("Processing stock price data...")
    
    processed_data = {}
    
    # Process each ticker
    for ticker in COMPANIES.keys():

        # Load stock price data
        price_file = f'data/raw/stock_prices/{ticker}_prices.csv'
        if not os.path.exists(price_file):
            print(f"Price data not found for {ticker}")
            continue
            
        try:
            # Load price data
            df = pd.read_csv(price_file, index_col=0, parse_dates=True)
            
            # New column: Calculate returns
            df['Daily_Return'] = df['Close'].pct_change() * 100
            
            # New Column: Calculate 5-day moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            
            # New Column: how much trading volume changed compared to the previous day
            df['Volume_Change'] = df['Volume'].pct_change() * 100
            
            # New Column: RSI
            df['RSI'] = calculate_rsi(df['Close'])
            
            # Load earnings dates
            earnings_file = f'data/raw/earnings_dates/{ticker}_earnings_dates.csv'
            if os.path.exists(earnings_file):
                earnings_dates = pd.read_csv(earnings_file, index_col=0, parse_dates=True)
                
                # New column: post-earnings days
                df['Post_Earnings'] = 0
                
                for date in earnings_dates.index:
                    # Find closest trading day to earnings date
                    # stock markets don't open every day (not open on weekends or holidays)
                    date_dt = pd.to_datetime(date)
                    closest_dates = df.index[df.index >= date_dt]
                    
                    if len(closest_dates) > 0:
                        earnings_idx = closest_dates[0]
                        
                        # Mark 5 days after earnings
                        for i in range(0, 6):
                            try:
                                # get the position of the earnings date
                                idx = df.index.get_loc(earnings_idx) + i
                                # check - valid range
                                if 0 <= idx < len(df):
                                    df.loc[df.index[idx], 'Post_Earnings'] = 1
                            except:
                                pass
            
            # Save processed data
            processed_data[ticker] = df
            output_file = f'data/processed/{ticker}_processed.csv'
            df.to_csv(output_file)
            print(f"Saved processed stock data for {ticker} to {output_file}")
            
        except Exception as e:
            print(f"Error processing stock data for {ticker}: {e}")
    
    print("Stock data processing complete.")
    return processed_data


# -- Processes SEC filings for sentiment (original keyword-based method) --

# This is kept for reference and fallback only, the preferred method is RoBERTa

def process_sec_filings_keyword():
    print("Processing SEC filings...")
    
    sentiment_data = []

    # Chose 2021-01-01 to 2024-12-31 to focus on recent market behaviors while 
    # avoiding the extreme volatility of 2020 (COVID-19)    
    start_date = pd.to_datetime("2021-01-01")
    end_date = pd.to_datetime("2024-12-31")
    
    # Find all SEC filings
    filings_dir = 'data/raw/earnings_filings/sec-edgar-filings'
    
    # Check if directory exists
    if not os.path.exists(filings_dir):
        print(f"SEC filings directory not found: {filings_dir}")
    
    # Each company has its own 8-K directory
    for ticker in COMPANIES.keys():
        company_dir = f'{filings_dir}/{ticker}/8-K'
        if not os.path.exists(company_dir):
            print(f"No SEC filings found for {ticker}")
            continue
            
        # Find all filing directories for this company
        filing_dirs = glob.glob(f'{company_dir}/*/')
        
        for filing_dir in filing_dirs:
            try:
                # Find all text files in this filing directory
                filing_files = glob.glob(f'{filing_dir}/*.txt')
                if not filing_files:
                    continue
                
                # Use the first text file 
                # Usually the first one is the main filing
                filing_path = filing_files[0]
                
                # Read the beginning of the file to extract the date
                with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                
                # Extract date in format <SEC-DOCUMENT>XXXXXXXX-XX-XXXXXX.txt : YYYYMMDD
                date_match = re.search(r': (\d{8})$', first_line)
                
                # Extract the year, month, and convert the month into a quarter
                if date_match:
                    date_str = date_match.group(1)
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    quarter = (month - 1) // 3 + 1
                    
                    # Ignore any filings outside of our chosen date range.
                    filing_date = pd.to_datetime(f"{year}-{month:02d}-01")
                    if filing_date < start_date or filing_date > end_date:
                        print(f"Skipping filing from {filing_date.strftime('%Y-%m-%d')} - outside target date range")
                        continue
                else:
                    # Default if date not found
                    print(f"Could not extract date from: {first_line}")
                    continue

                # Read filing text
                with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
                    filing_text = f.read()
            
                # Sentiment analysis: count positive and negative words
                positive_words = ['increase', 'growth', 'improved', 'higher', 'strong', 'positive', 
                                 'exceeded', 'beat', 'record', 'success', 'profit', 'gain']
                negative_words = ['decrease', 'decline', 'fell', 'lower', 'weak', 'negative', 
                                 'missed', 'loss', 'challenging', 'difficult', 'down', 'reduced']
                
                # Count each type of word's occurence
                filing_lower = filing_text.lower()
                positive_count = sum(filing_lower.count(word) for word in positive_words)
                negative_count = sum(filing_lower.count(word) for word in negative_words)
                
                # Calculate sentiment score 
                # - positive = +1
                # - negative = -1
                # - neutral = 0
                total_count = positive_count + negative_count
                if total_count > 0:
                    sentiment_score = (positive_count - negative_count) / total_count
                else:
                    # Neutral if no sentiment words found
                    sentiment_score = 0 
                
                # Make the score category
                if sentiment_score > 0.1:
                    sentiment_category = 'positive'
                elif sentiment_score < -0.1:
                    sentiment_category = 'negative'
                else:
                    sentiment_category = 'neutral'
                
                # Add to sentiment data
                sentiment_data.append({
                    'ticker': ticker,
                    'company': COMPANIES[ticker],
                    'year': year,
                    'quarter': quarter,
                    'sentiment_score': sentiment_score,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'sentiment': sentiment_category,
                    'filing_path': filing_path
                })
                
                print(f"Processed sentiment for {ticker} {year} Q{quarter}: {sentiment_category} ({sentiment_score:.2f})")
                
            except Exception as e:
                print(f"Error processing {filing_dir}: {e}")
    
    # Convert to DataFrame
    if sentiment_data:
        df = pd.DataFrame(sentiment_data)
        
        # Save to CSV
        output_file = 'data/processed/earnings_sentiment_keyword.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved keyword-based sentiment analysis to {output_file}")
        
        return df
        
        return df
    else:
        print("No sentiment data created from SEC filings.")


# -- Create combined feature dataset --

def create_feature_dataset(stock_data, sentiment_data):

    print("Creating feature dataset...")
    
    all_features = []
    
    # Loop through each row in the sentiment file
    for r, sentiment in sentiment_data.iterrows():
        ticker = sentiment['ticker']
        
        if ticker not in stock_data:
            continue
      
        year = int(sentiment['year'])
        quarter = int(sentiment['quarter'])

        #check - fixed
        if year < 1900 or year > 2100 or quarter < 1 or quarter > 4:
            print(f"Skipping invalid date: Year={year}, Quarter={quarter}")
            continue
        
        # Convert quarter into approx. month
        month = (quarter * 3) - 1 

        # check - fixed
        if month < 1 or month > 12:
            print(f"Invalid month calculated: {month} from quarter {quarter}")
                # Fix the month to be in valid range
            month = max(1, min(12, month))
        
        # appoximation
        estimated_date_str = f"{year}-{month:02d}-15"
        estimated_date = pd.to_datetime(estimated_date_str)
        
        # Find closest trading day in stock data - stock markets don’t open on weekends and holidays
        df = stock_data[ticker]
        estimated_date_str = estimated_date.strftime('%Y-%m-%d')
        closest_dates = df.index[[date.strftime('%Y-%m-%d') >= estimated_date_str for date in df.index]]
        
        if len(closest_dates) == 0:
            continue
            
        report_date = closest_dates[0]
        
        # next 5 days of stock data. Find post-earnings days 
        post_earnings = df.loc[report_date:report_date + timedelta(days=5)]
        
        # need at least 2 days to compute a 1-day return
        if len(post_earnings) < 2:
            continue
            
        # Get 1-day return
        return_1d = post_earnings['Daily_Return'].iloc[1] if len(post_earnings) > 1 else np.nan
        
        # Label the movement
        if not np.isnan(return_1d):
            if return_1d > 1:
                target = 1  # up
            elif return_1d < -1:
                target = -1  # down
            else:
                target = 0  # stable
        else:
            target = None
        
        # Create feature record
        feature = {
            'ticker': ticker,
            'company': sentiment['company'],
            'year': year,
            'quarter': quarter,
            'report_date': report_date,
            'sentiment_score': sentiment['sentiment_score'],
            'sentiment': sentiment['sentiment'],
            'return_1d': return_1d,
            'target': target
        }

        # 7-day window before the earnings report - technical indicator calculation
        pre_date = report_date - timedelta(days=7)
        pre_data = df[(df.index >= pre_date) & (df.index < report_date)]
        
        if not pre_data.empty:
            # average return before earnings
            feature['pre_return'] = pre_data['Daily_Return'].mean()
            # average volume change
            feature['pre_volume_change'] = pre_data['Volume_Change'].mean()
            # RSI
            feature['pre_rsi'] = pre_data['RSI'].iloc[-1] if 'RSI' in pre_data else np.nan
        
        all_features.append(feature)
    
    # Convert to dataframe
    if all_features:
        df = pd.DataFrame(all_features)
        
        # Remove rows with missing target
        df = df.dropna(subset=['target'])
        
        # Save to CSV
        output_file = 'data/features/combined_features.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved combined features to {output_file}")
        
        return df
    else:
        print("No features created.")
        return None


# Main function
def main():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    
    # Process stock data (this remains the same)
    stock_data = process_stock_data()
    
    # Use RoBERTa for sentiment analysis (preferred method)
    if HAS_ROBERTA:
        try:
            print("Using RoBERTa for sentiment analysis...")
            sentiment_data = process_sec_filings_with_roberta()
            if sentiment_data is None:
                print("RoBERTa sentiment analysis failed, falling back to keyword method.")
                sentiment_data = process_sec_filings_keyword()
        except Exception as e:
            print(f"Error in RoBERTa sentiment analysis: {e}")
            print("Falling back to keyword-based method.")
            sentiment_data = process_sec_filings_keyword()
    else:
        print("RoBERTa not available. Using keyword-based sentiment analysis.")
        sentiment_data = process_sec_filings_keyword()
    
    if sentiment_data is not None:
        create_feature_dataset(stock_data, sentiment_data)
    else:
        print("Cannot create feature dataset: No sentiment data available")
    
    print("Data processing complete")

if __name__ == "__main__":
    main()