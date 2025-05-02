# enhanced_sentiment_analysis.py
# This module provides transformer-based sentiment analysis using RoBERTa

import pandas as pd
import numpy as np
import os
import re
import traceback  # Added for better error reporting
from datetime import timedelta
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from tqdm import tqdm
from collections import Counter

print("Starting enhanced_sentiment_analysis.py...")

# Define companies (imported from original code for reference)
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

SAMPLE_RATE = 0.1


print(f"Defined {len(COMPANIES)} companies for analysis")

# Split long text into chunks that RoBERTa can handle
def chunk_text(text, max_length=1000):
   
    print(f"Chunking text of length {len(text)} chars with max_length={max_length}")
    words = text.split()
    print(f"Text contains {len(words)} words")
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        # Approximate token count (this is rough)
        word_tokens = len(word) // 4 + 1
        
        if current_length + word_tokens > max_length:
            # Current chunk is full, save it and start a new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_tokens
        else:
            # Add word to current chunk
            current_chunk.append(word)
            current_length += word_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    print(f"Split text into {len(chunks)} chunks")
    return chunks


# Extract date from an SEC filing, handling different formats
def extract_date_from_filing(filing_path):
    
    print(f"Extracting date from {filing_path}")
    # Read the first few lines of the file to find date information
    with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Read first 30 lines to search for date
        header_lines = [f.readline().strip() for _ in range(30)]
    
    print(f"Read {len(header_lines)} header lines")
    
    # Try standard format: <SEC-DOCUMENT>XXXXXXXX-XX-XXXXXX.txt : YYYYMMDD
    for line in header_lines:
        date_match = re.search(r': (\d{8})$', line)
        if date_match:
            date_str = date_match.group(1)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            quarter = (month - 1) // 3 + 1
            print(f"Found date using standard format: Year={year}, Month={month}, Quarter={quarter}")
            return year, month, quarter
    
    # Try PEM format - look for FILING-DATE or ACCEPTANCE-DATETIME
    for i, line in enumerate(header_lines):
        if 'FILING-DATE' in line or 'ACCEPTANCE-DATETIME' in line:
            # Check next line for date
            if i+1 < len(header_lines):
                date_line = header_lines[i+1]
                date_match = re.search(r'(\d{8})', date_line)
                if date_match:
                    date_str = date_match.group(1)
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    quarter = (month - 1) // 3 + 1
                    print(f"Found date using PEM format: Year={year}, Month={month}, Quarter={quarter}")
                    return year, month, quarter
            
    # Try looking for date in different format: MM/DD/YYYY or similar
    for line in header_lines:
        date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', line)
        if date_match:
            month = int(date_match.group(1))
            # day = int(date_match.group(2))  # Not used
            year = int(date_match.group(3))
            quarter = (month - 1) // 3 + 1
            print(f"Found date using MM/DD/YYYY format: Year={year}, Month={month}, Quarter={quarter}")
            return year, month, quarter
    
    # If we got here, no date was found
    print(f"No date found in {filing_path}")
    return None

# dded new function to sample filing content
def sample_filing_content(filing_text, sample_rate=SAMPLE_RATE):
    import random
    
    print(f"Sampling {sample_rate*100:.1f}% of filing content")
    print(f"Original text length: {len(filing_text)} characters")
    
    # Split the filing into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', filing_text) if p.strip()]
    
    if not paragraphs:
        print(" No paragraphs found in text. Using original text.")
        return filing_text  # Return original if no paragraphs found
    
    # Print paragraph stats
    print(f"Text split into {len(paragraphs)} paragraphs")
    
    # Randomly select sample_rate percentage of paragraphs
    sample_size = max(1, int(len(paragraphs) * sample_rate))  # At least 1 paragraph
    selected_paragraphs = random.sample(paragraphs, sample_size)
    
    # Combine the selected paragraphs back into a text
    sampled_text = '\n\n'.join(selected_paragraphs)
    
    # Print sampling stats
    print(f"Sampled {sample_size}/{len(paragraphs)} paragraphs ({sample_rate*100:.1f}% of content)")
    print(f"Sampled text length: {len(sampled_text)} characters ({len(sampled_text)/len(filing_text)*100:.1f}% of original)")
    
    return sampled_text

# RoBERTa-based sentiment analyzer
class RoBERTaSentimentAnalyzer:
    
    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment'):
        
        print(f"Initializing RoBERTaSentimentAnalyzer with model {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {self.device}")
        
        # Load pre-trained RoBERTa model and tokenizer
        try:
            print(f"Loading tokenizer from {model_name}")
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            print(f"Loading model from {model_name}")
            self.model = RobertaForSequenceClassification.from_pretrained(model_name).to(self.device)
            print(f"Successfully loaded {model_name}")
            
            # Using the pipeline for easier sentiment analysis
            print(f"Setting up sentiment pipeline")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"Pipeline setup complete")
            
            # Test the pipeline with examples
            print(f"Testing pipeline with example texts")
            test_examples = [
                "I really love this product, it's excellent!",
                "This is terrible, worst experience ever.",
                "The meeting was held on Tuesday."
            ]
            
            for i, example in enumerate(test_examples):
                print(f"Test example {i+1}: '{example}'")
                result = self.sentiment_pipeline(example)[0]
                print(f"Result: {result}")
            
            self.model_loaded = True
            print(f"RoBERTaSentimentAnalyzer initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to load RoBERTa model: {e}")
            traceback.print_exc()  # Print full error details
            self.model_loaded = False
    
    def analyze_sentiment(self, text):
        print(f"analyze_sentiment called with text length: {len(text)}")
        
        if not self.model_loaded:
            print(f"Model not loaded, returning neutral sentiment")
            return 0, 'neutral'
            
        if not text or len(text.strip()) == 0:
            print(f"Empty text provided, returning neutral sentiment")
            return 0, 'neutral'  # Default for empty text
        
        try:
            # For very long texts, chunk and analyze each chunk
            if len(text) > 5000:  # If text is very long
                print(f"Text is long ({len(text)} chars), splitting into chunks")
                chunks = chunk_text(text)
                print(f"Split into {len(chunks)} chunks")
                scores = []
                categories = []
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        print(f"Chunk {i+1} is empty, skipping")
                        continue
                        
                    # Process each chunk
                    try:
                        print(f" Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                        result = self.sentiment_pipeline(chunk[:512])[0]  # Limit to 512 tokens
                        label = result['label']
                        score = result['score']
                        print(f"Chunk {i+1} raw result: {result}")
                        
                        # Convert labels using helper method
                        sentiment_score, sentiment_category = self._convert_label(label, score)
                        categories.append(sentiment_category)
                        scores.append(sentiment_score)
                        print(f"Chunk {i+1}: {sentiment_category} ({sentiment_score:.2f})")
                        
                    except Exception as chunk_error:
                        print(f"ERROR: Error processing chunk {i+1}: {chunk_error}")
                        traceback.print_exc()
                        continue
                
                # Aggregate results - use the most common category and average score
                if not scores:
                    print(f"No valid chunks processed, returning neutral sentiment")
                    return 0, 'neutral'
                    
                # Find most common category
                category_counts = Counter(categories)
                print(f"Category counts: {category_counts}")
                most_common = category_counts.most_common(1)[0][0]
                
                # Calculate average score
                avg_score = sum(scores) / len(scores)
                print(f"Final sentiment for all chunks: {most_common} with score {avg_score:.2f}")
                
                return avg_score, most_common
            else:
                # Use the pipeline for shorter texts
                print(f"Processing shorter text directly ({len(text)} chars)")
                result = self.sentiment_pipeline(text[:512])[0]  # Limit to 512 tokens
                print(f"Raw model output: {result}")
                label = result['label']
                score = result['score']
                
                # Convert labels using helper method
                sentiment_score, sentiment_category = self._convert_label(label, score)
                print(f" Sentiment: {sentiment_category} ({sentiment_score:.2f})")
                    
                return sentiment_score, sentiment_category
        except Exception as e:
            print(f"ERROR: Error analyzing sentiment with RoBERTa: {e}")
            traceback.print_exc()
            return 0, 'neutral'
       
        

    #  Convert model label to sentiment category and score
    def _convert_label(self, label, score, threshold=0.5):
   
        print(f"Converting label '{label}' with score {score}")
    
        # Map labels to sentiment categories
        if label == 'POSITIVE' or label == 'positive' or label == '2' or label == 'LABEL_2':
            print(f"Identified as POSITIVE")
            sentiment_category = 'positive'
            sentiment_score = score
        elif label == 'NEGATIVE' or label == 'negative' or label == '0' or label == 'LABEL_0':
            print(f"Identified as NEGATIVE")
            sentiment_category = 'negative'
            sentiment_score = -score
        else:
            print(f" Identified as NEUTRAL: '{label}'")
            sentiment_category = 'neutral'
            sentiment_score = 0
            
        return sentiment_score, sentiment_category
    
    def batch_analyze(self, texts, batch_size=16):
        print(f"batch_analyze called with {len(texts)} texts, batch_size={batch_size}")
        
        if not self.model_loaded:
            print(f"Model not loaded, returning neutral sentiment for all texts")
            return [(0, 'neutral') for _ in texts]
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i:i+batch_size]
            print(f"Processing batch {batch_num}/{total_batches} with {len(batch)} texts")
            
            try:
                # Process each text individually to handle long texts properly
                batch_results = []
                for j, text in enumerate(batch):
                    print(f"Processing text {j+1}/{len(batch)} in batch {batch_num}")
                    sentiment_score, sentiment_category = self.analyze_sentiment(text)
                    batch_results.append((sentiment_score, sentiment_category))
                    print(f"Text {j+1} result: {sentiment_category} ({sentiment_score:.2f})")
                
                results.extend(batch_results)
                print(f"Completed batch {batch_num}/{total_batches}")
            except Exception as e:
                print(f"ERROR: Error in batch processing: {e}")
                traceback.print_exc()
                
                # Fall back to individual processing on errors
                print(f"Falling back to individual processing for batch {batch_num}")
                for j, text in enumerate(batch):
                    try:
                        print(f"Fallback processing for text {j+1}/{len(batch)}")
                        result = self.analyze_sentiment(text)
                        results.append(result)
                    except Exception as text_error:
                        print(f"ERROR: Error in fallback processing for text {j+1}: {text_error}")
                        traceback.print_exc()
                        results.append((0, 'neutral'))
        
        print(f" Completed batch processing, got {len(results)} results")
        return results


# Main function to process SEC filings with RoBERTa
def process_sec_filings_with_roberta():

    print("Starting process_sec_filings_with_roberta")
    
    sentiment_data = []
    filings_processed = 0
    
    # Date range as in original code
    start_date = pd.to_datetime("2021-01-01")
    end_date = pd.to_datetime("2024-12-31")
    print(f"Using date range: {start_date} to {end_date}")
    
    # Initialize the RoBERTa sentiment analyzer
    print("Initializing RoBERTa sentiment analyzer")
    roberta_analyzer = RoBERTaSentimentAnalyzer()
    if not roberta_analyzer.model_loaded:
        print("ERROR: Failed to load RoBERTa model.")
        return None
    
    # Find all SEC filings
    filings_dir = 'data/raw/earnings_filings/sec-edgar-filings'
    print(f"Looking for SEC filings in {filings_dir}")
    
    # Check if directory exists
    if not os.path.exists(filings_dir):
        print(f"ERROR: SEC filings directory not found: {filings_dir}")
        return None
    
    # Process each company's filings
    for ticker in COMPANIES.keys():
        print(f"\nProcessing {ticker} ({COMPANIES[ticker]})")
        company_dir = f'{filings_dir}/{ticker}/8-K'
        
        if not os.path.exists(company_dir):
            print(f"No SEC filings found for {ticker}")
            continue
        
        # Find and process all filing directories for this company
        import glob
        filing_dirs = glob.glob(f'{company_dir}/*/')
        print(f"Found {len(filing_dirs)} filing directories for {ticker}")
        
        # List to collect all filing texts for batch processing
        filing_texts = []
        filing_metadata = []
        
        for i, filing_dir in enumerate(filing_dirs):
            filing_id = os.path.basename(os.path.dirname(filing_dir))
            print(f" Processing filing {i+1}/{len(filing_dirs)}: {filing_id}")
            
            try:
                # Find all text files in this filing directory
                filing_files = glob.glob(f'{filing_dir}/*.txt')
                if not filing_files:
                    print(f"No text files found in {filing_id}")
                    continue
                
                # Use the first text file (usually the main filing)
                filing_path = filing_files[0]
                print(f"Using file: {os.path.basename(filing_path)}")
                
                # Extract date using improved method
                date_info = extract_date_from_filing(filing_path)
                if date_info:
                    year, month, quarter = date_info
                    
                    # Skip filings outside date range
                    filing_date = pd.to_datetime(f"{year}-{month:02d}-01")
                    if filing_date < start_date or filing_date > end_date:
                        print(f"Skipping filing from {filing_date.strftime('%Y-%m-%d')} - outside target date range")
                        continue
                else:
                    print(f"Could not extract date from file: {filing_path}")
                    continue
                
                # Read filing text
                print(f"Reading filing text")
                with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
                    filing_text = f.read()
                
                print(f"Read {len(filing_text)} characters from filing")

                sampled_text = sample_filing_content(filing_text, SAMPLE_RATE)

                # Collect texts for batch processing
                filing_texts.append(sampled_text) 
                filing_metadata.append({
                    'ticker': ticker,
                    'company': COMPANIES[ticker],
                    'year': year,
                    'quarter': quarter,
                    'filing_path': filing_path,
                    'filing_id': filing_id
                })
                
                filings_processed += 1
                print(f"Successfully processed filing. Total filings processed so far: {filings_processed}")
                
            except Exception as e:
                print(f"ERROR: Error processing {filing_dir}: {e}")
                traceback.print_exc()
        
        # Process batch with RoBERTa
        if filing_texts:
            print(f"Batch processing {len(filing_texts)} filings for {ticker} with RoBERTa")
            
            # Process in batches with smaller batch size for large files
            batch_results = roberta_analyzer.batch_analyze(filing_texts, batch_size=4)
            
            # Add batch results to sentiment data
            for i, result in enumerate(batch_results):
                if i < len(filing_metadata):  # Safety check
                    sentiment_score, sentiment_category = result
                    metadata = filing_metadata[i]
                    
                    sentiment_data.append({
                        'ticker': metadata['ticker'],
                        'company': metadata['company'],
                        'year': metadata['year'],
                        'quarter': metadata['quarter'],
                        'sentiment_score': sentiment_score,
                        'sentiment': sentiment_category,
                        'filing_path': metadata['filing_path'],
                        'filing_id': metadata['filing_id']
                    })
                    
                    print(f"RoBERTa sentiment for {metadata['ticker']} {metadata['filing_id']} {metadata['year']} Q{metadata['quarter']}: {sentiment_category} ({sentiment_score:.2f})")
    
    # Convert to DataFrame
    if sentiment_data:
        print(f"Creating DataFrame with {len(sentiment_data)} sentiment records")
        df = pd.DataFrame(sentiment_data)
        
        # Save to CSV
        output_file = 'data/processed/earnings_sentiment_roberta.csv'
        df.to_csv(output_file, index=False)
        print(f" Saved RoBERTa sentiment analysis to {output_file}")
        
        return df
    else:
        print("ERROR: No sentiment data created from SEC filings.")
        return None


# Main function to run RoBERTa sentiment analysis
def main():

    print("Starting main function")
    os.makedirs('data/processed', exist_ok=True)
    print(f"Ensured data/processed directory exists")

    # Set confidence threshold - lower values will be more sensitive to sentiment
    sentiment_threshold = 0.5  # Try values between 0.3-0.6
    
    print("\n" + "="*50)
    print("STARTING ROBERTA SENTIMENT ANALYSIS")
    print("="*50)
    print(f"Settings:")
    print(f"- Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"- Analyzing {SAMPLE_RATE*100}% of each filing")
    print("="*50 + "\n")
    
    # Process filings with RoBERTa sentiment analysis
    sentiment_data = process_sec_filings_with_roberta()
    
    if sentiment_data is not None:
        # Display sentiment distribution
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        print("\nSentiment distribution (RoBERTa):")
        print(sentiment_counts)
        
        # Display average sentiment score by company
        avg_by_company = sentiment_data.groupby('ticker')['sentiment_score'].mean()
        print("\nAverage sentiment score by company:")
        print(avg_by_company)
        
        print(f"\nTotal filings processed: {len(sentiment_data)}")
    else:
        print("\nERROR: No sentiment data was generated. Please check logs for details.")
    
    print("\n" + "="*50)
    print("ROBERTA SENTIMENT ANALYSIS COMPLETE")
    print("="*50)
    print(" Main function completed")


if __name__ == "__main__":
    main()