# -- data_collection.py collects all the raw data --

import pandas as pd
import numpy as np
import os
import yfinance as yf # https://pypi.org/project/yfinance/
from sec_edgar_downloader import Downloader # https://pypi.org/project/sec-edgar-downloader/


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

# -- Set up directory structure --
def create_directories():

    directories = [
        'data/raw/stock_prices', # raw stock price data
        'data/raw/earnings_dates', # information about when companies announced their quarterly earnings
        'data/raw/earnings_filings',
        'data/processed', # cleaned and processed versions of the raw data
        'data/features', # final datasets used
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created.")


# -- Collect stock price data from Yahoo Finance --
# Downloads daily stock prices (open, high, low, close, volume),
# saves them as CSV,
# and gets earnings announcement dates
# Chose 2021-01-01 to 2024-12-31 to focus on recent market behaviors while 
# avoiding the extreme volatility of 2020 (COVID-19)

def collect_stock_data(start_date='2021-01-01', end_date='2024-12-31'):
    
    print(f"Collecting stock data from {start_date} to {end_date}...")
    
    for ticker in COMPANIES:
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date)
            
            # if no data was found, skips to next comp
            if hist_data.empty:
                print(f"No data found for {ticker}")
                continue
                
            # Save to CSV
            output_file = f'data/raw/stock_prices/{ticker}_prices.csv'
            hist_data.to_csv(output_file)
            print(f"Saved stock data for {ticker} to {output_file}")
            
            # Get earnings dates
            earnings_dates = stock.earnings_dates
            
            # checks whether successful
            if earnings_dates is not None and not earnings_dates.empty:
                # include only relevant date range
                earnings_dates = earnings_dates[
                    (earnings_dates.index >= start_date) & 
                    (earnings_dates.index <= end_date)
                ]
                
                # Save to CSV
                earnings_file = f'data/raw/earnings_dates/{ticker}_earnings_dates.csv'
                earnings_dates.to_csv(earnings_file)
                print(f"Saved earnings dates for {ticker} to {earnings_file}")
            else:
                print(f"No earnings dates found for {ticker}")
                
        except Exception as e:
            print(f"Error collecting data for {ticker}: {e}")
    
    print("Stock data collection complete.")


# -- Collect S&P 500 data for market comparison --

def collect_sp500_data(start_date='2021-01-01', end_date='2024-12-31'):

    try:
        print(f"Collecting S&P 500 data from {start_date} to {end_date}...")
        
        # Get S&P 500 data
        sp500 = yf.Ticker('^GSPC') # ^GSPC is the ticker symbol used in Yahoo finance for S&P500
        sp500_data = sp500.history(start=start_date, end=end_date)
        
        # if no data was found, exit
        if sp500_data.empty:
            print("No S&P 500 data found")
            return
        
        # Save to CSV
        output_file = 'data/raw/stock_prices/SP500_prices.csv'
        sp500_data.to_csv(output_file)
        print(f"Saved S&P 500 data to {output_file}")
        
    except Exception as e:
        print(f"Error collecting S&P 500 data: {e}")



# -- Download SEC earnings filings using sec-edgar-downloader --

def download_earnings_filings(form_type="8-K", amount=10):
    print("Downloading earnings filings (8-Ks)...")
    
    # Replace with your actual email address
    email_address = "alexiak@bu.edu"  
    
    dl = Downloader("BU_CS506", email_address, "data/raw/earnings_filings")
    for ticker in COMPANIES.keys():
        dl.get(form_type, ticker)
        print(f"Downloaded {form_type} filings for {ticker}")
    print("All filings downloaded.")


def main():
    create_directories()
    
    collect_stock_data()
    
    collect_sp500_data()
    
    download_earnings_filings()
    
    print("Data collection complete. Check data/ for collected data.")

if __name__ == "__main__":
    main()