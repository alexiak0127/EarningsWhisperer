# EarningsWhisperer: AI Reads Earnings Reports So You Don’t Have To

## **Project Description**
Earnings reports offer valuable insights into a company's financial health, but investors frequently find it challenging to gauge how the market will react. Predicting stock price movements is challenging because of market efficiency, complex dependencies, and the influence of multiple factors. 
This project integrates **natural language processing (NLP)** and **historical stock data** to build a machine learning model capable of **predicting stock movement (up, down, or stable)** following earnings announcements. By leveraging sentiment analysis of earnings reports and financial indicators, the project aims to provide a data-driven approach to improve investment decision-making.

To ensure feasibility and relevance, the project will focus on **10 major publicly traded tech companies** with significant market influence and high trading volume:
1. Apple (AAPL)
2. Microsoft (MSFT)
3. Google (GOOGL)
4. Amazon (AMZN)
5. Nvidia (NVDA)
6. Meta (META)
7. Tesla (TSLA)
8. Advanced Micro Devices (AMD)
9. Intel (INTC)
10. Salesforce (CRM)

🎥 Check out my mid-term presentation here! https://youtu.be/EHeJyJaSBjA

## **Goals**
- Develop a **stock movement prediction model** based on earnings reports.
- Implement **sentiment analysis** to classify earnings reports as **positive, neutral, or negative**.
- Train a **machine learning model** (Logistic Regression, Random Forest, and Deep Neural Networks) to predict stock movement based on report sentiment.
- Create **visualizations** to showcase stock trends before and after earnings.

## **Data Collection**
### **Sources:**
- **[Yahoo Finance](https://pypi.org/project/yfinance/)** → I use the Yahoo Finance API (yfinance library) to collect historical stock data for the target companies, covering the period from 2021-01-01 to 2024-12-31. I limited the data range to keep the focus on recent market behavior while avoiding COVID-era volatility.
- **[SEC Edgar Database](https://pypi.org/project/sec-edgar-downloader/))** → I utilize the SEC EDGAR database(sec_edgar_downloader) to collect 8-K filings containing earnings announcements. These reports provide the textual data for my sentiment analysis.

### **Data Processing:**
- **Stock Price Data:**
  - Open: Price at the start of the trading session
  - Close: Price at market close
  - High Price: Maximum price reached within a trading session
  - Low Price: Minimum price reached within a trading session
  - Volume: Number of shares traded within a given period
  - Moving Averages: Rolling averages over different time windows (e.g., 5-day, 30-day)
  - Volatility: Standard deviation of stock price movements over time
- **Earnings Report Text:** Extracted from SEC filings.
- **Sentiment Score:** Derived from earnings report text.
- **Market Performance Indicators:** Broader market trends like S&P 500 movements on earnings day.

## **Data Cleaning**
- **Standardization:** Convert all financial text into a structured format.
- **Data Augmentation:** Use various earnings reports to improve generalization.

## **Feature Extraction**
- **Sentiment Score Calculation:** Apply VADER NLP model and TF-IDF vectorization to quantify sentiment.
- **Stock Movement Labeling:** Classify movement as up (1), down (-1), or stable (0) based on the closing price change on the next trading day after the earnings announcement.
- **Time-Series Features:** Include moving averages, volatility, and price trends.
- **Keyword Analysis:** Identify phrases in reports like “exceeds expectations” vs. “missed forecast.”
- **Feature Engineering:** Extract financial terms and sentiment scores from text.

## **Modeling**
### **Model Architecture:**
- **Sentiment Analysis Pipeline:** Convert earnings reports into structured sentiment scores.
- **Machine Learning Classifier:** Train models (Logistic Regression, Random Forest, and more) to predict stock price direction.
- **Deep Learning Motivation:** Given the potentially large and high-dimensional dataset, especially from textual features, deep neural networks may better capture nonlinear patterns and interactions.
- **Baseline Comparison:** Compare predictions against a simple buy-and-hold strategy and market index performance.
- **Hyperparameter Tuning:** Optimize learning rate, tree depth, and regularization parameters.

### **Training Approach:**
- **Supervised Learning:** Train the model using past stock movements and earnings reports.
- **Feature Selection:** Optimize relevant features to improve predictive accuracy.
- **Cross-Validation:** Use k-fold validation to avoid overfitting.

## **Visualization**
- **Stock Price vs. Earnings Sentiment:** Scatter plots, time-series analysis.
- **Model Predictions vs. Actual Stock Movements:** Confusion matrix, performance charts.
- **Sector-Wise Performance:** Heatmap to show model accuracy per industry.
- **Interactive Dashboards:** Real-time stock predictions based on the latest earnings reports. Users can choose specific stocks to analyze.

## **Test Plan / Metrics**
### **Data Split:**
- **80% Training, 20% Testing** to validate model performance.
- **Holdout Test Set:** Use a separate set of unseen earnings reports to evaluate final performance.

### **Evaluation Metrics:**
- **Classification Accuracy:** At least 70%+ accuracy for meaningful predictions.
- **Precision, Recall, and F1-Score:** Ensure balanced recall and precision to avoid false signals.
- **Mean Squared Error (MSE) and R² Score:** Evaluate regression-based approaches for numerical predictions.
- **Market Comparison:** Benchmark model against real stock trends and naive investment strategies.
