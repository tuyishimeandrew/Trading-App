#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Enhanced Trading System Streamlit App with Auto-Refresh Every 5 Minutes

This app:
  - Fetches historical forex data.
  - Computes technical indicators and sentiment scores.
  - Prepares and labels data.
  - Trains multiple machine learning models and selects the best.
  - Runs Monte Carlo simulations for future price estimation.
  - Executes (simulated) trades based on model and simulation agreement.
  - Automatically re-runs the entire analysis every 5 minutes.
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh  # Auto-refresh component

# Auto-refresh the app every 5 minutes (300,000 ms)
st_autorefresh(interval=300000, limit=0, key="trading_analysis_refresh")

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# Machine learning libraries.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Technical indicators.
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands

# Setup logging to "log.txt"
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download NLTK VADER lexicon.
nltk.download('vader_lexicon')

# =============================================================================
# Global Configuration for MT4 (placeholder values)
# =============================================================================
MT4_SERVER = "templer markets mt4-usd"
MT4_LOGIN = "824195"
MT4_PASSWORD = "DYXTY43"
MT4_PORT = 443  # Adjust as needed

# Dictionary to track open trades.
open_trades = {}

# =============================================================================
# 1. Data and News Functions
# =============================================================================
def get_currencies(pair):
    """Extract base and quote currencies from pair symbol."""
    if pair.endswith('=X'):
        return pair[:3], pair[3:6]
    elif '-' in pair:
        parts = pair.split('-')
        return parts[0], parts[1]
    else:
        return pair, 'USD'

# Placeholder for your NewsAPI key (replace with your actual key)
NEWSAPI_API_KEY = "your_news_api_key"

def fetch_news(query):
    """Fetch recent news articles using NewsAPI."""
    try:
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'apiKey': NEWSAPI_API_KEY,
            'pageSize': 5,
            'sortBy': 'publishedAt',
            'language': 'en'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logging.error(f"News API Error: {str(e)}")
        return []

def compute_sentiment_score(pair):
    """Compute an average sentiment score using VADER on news headlines and descriptions."""
    base, quote = get_currencies(pair)
    articles = fetch_news(f"{base} {quote} OR Forex OR Currency")
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for article in articles:
        for field in ['title', 'description']:
            text = article.get(field, '')
            if text:
                sentiments.append(sia.polarity_scores(text)['compound'])
    return sum(sentiments)/len(sentiments) if sentiments else 0.0

def fetch_forex_data(symbol, start_date, end_date):
    """Fetch historical price data from Yahoo Finance."""
    logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        logging.error(f"Error fetching data for {symbol}")
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data['datetime'] = data.index
    data = data[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume'}, inplace=True)
    for col in ['open', 'High', 'Low', 'close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)
    return data

# =============================================================================
# 2. Feature Engineering Functions using TA Indicators
# =============================================================================
def add_ta_indicators(df):
    """Add technical indicators (shifted by one row to avoid lookahead bias)."""
    df = df.copy()
    rsi = RSIIndicator(close=df['close'], window=14, fillna=True)
    df['rsi'] = rsi.rsi().shift(1)
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df['macd'] = macd.macd().shift(1)
    df['macd_signal'] = macd.macd_signal().shift(1)
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['close'], window=14, fillna=True)
    df['adx'] = adx.adx().shift(1)
    cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['close'], window=20, constant=0.015, fillna=True)
    df['cci'] = cci.cci().shift(1)
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['close'], window=14, smooth_window=3, fillna=True)
    df['stoch_k'] = stoch.stoch().shift(1)
    df['stoch_d'] = stoch.stoch_signal().shift(1)
    bb = BollingerBands(close=df['close'], window=20, window_dev=2, fillna=True)
    df['bb_width'] = bb.bollinger_wband().shift(1)
    df['sma_20'] = df['close'].rolling(window=20, min_periods=20).mean().shift(1)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean().shift(1)
    return df

def add_additional_features(df):
    """Add lag features, moving averages, and rate-of-change features."""
    df = df.copy()
    df['close_lag1'] = df['close'].shift(1)
    df['volume_lag1'] = df['Volume'].shift(1)
    df['ma_5'] = df['close'].rolling(window=5, min_periods=5).mean().shift(1)
    df['std_5'] = df['close'].rolling(window=5, min_periods=5).std().shift(1)
    df['roc'] = df['close'].pct_change(periods=5).shift(1)
    df.dropna(inplace=True)
    return df

# =============================================================================
# 3. Data Labeling Function with Enhanced Preparation
# =============================================================================
def prepare_data(df, tp, sl, pip_precision, sentiment_score, max_horizon=50, transaction_cost=0.00005):
    """
    Prepare data by creating target labels and pip differences.
    Modified logic:
      - For a long trade (label = 1): price must reach entry + tp before falling to entry - sl.
      - For a short trade (label = -1): price must reach entry - tp before rising to entry + sl.
      - If both thresholds are met within the same future bar, flag outcome as ambiguous (0).
      - Transaction cost is subtracted from pip differences.
      - Only scans up to max_horizon future bars.
    """
    df = df.copy()
    df = add_additional_features(df)
    df = add_ta_indicators(df)

    adjusted_tp = tp * pip_precision
    adverse = sl * pip_precision

    n = df.shape[0]
    targets = np.zeros(n)
    pip_diff = np.zeros(n)

    closes = df['close'].values
    highs = df['High'].values
    lows = df['Low'].values

    for i in range(n):
        base = closes[i]
        outcome = 0
        for j in range(i+1, min(n, i+1+max_horizon)):
            long_success = highs[j] >= base + adjusted_tp
            short_success = lows[j] <= base - adjusted_tp
            long_adverse = lows[j] <= base - adverse
            short_adverse = highs[j] >= base + adverse

            if long_success and short_success:
                outcome = 0
                break
            elif long_success:
                outcome = 1
                break
            elif short_success:
                outcome = -1
                break
            if long_adverse or short_adverse:
                outcome = 0
                break

        targets[i] = outcome
        if outcome == 1:
            pip_diff[i] = tp - (transaction_cost * tp)
        elif outcome == -1:
            pip_diff[i] = -tp - (transaction_cost * tp)
        else:
            pip_diff[i] = 0

    df['target'] = targets
    df['pip_diff'] = pip_diff
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    features = df.drop(columns=['target', 'datetime', 'pip_diff'])
    target = df['target']
    return features, target, df['pip_diff']

# =============================================================================
# 4. Profit Score Function for Model Evaluation
# =============================================================================
def profit_score(y_true, y_pred, pip_diff_series):
    """
    Calculate average profit per trade:
      - If prediction is correct, add the absolute pip difference.
      - If prediction is incorrect, subtract the absolute pip difference.
      - Trades with prediction 0 (no trade) are ignored.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pip_diff = np.array(pip_diff_series)
    profit = 0.0
    trade_count = 0
    for t, p, pip in zip(y_true, y_pred, pip_diff):
        if p == 0:
            continue
        trade_count += 1
        if p == t:
            profit += abs(pip)
        else:
            profit -= abs(pip)
    return profit / trade_count if trade_count > 0 else 0

profit_scorer = make_scorer(lambda yt, yp: profit_score(yt, yp, pip_diff_series=np.ones(len(yt))),
                            greater_is_better=True)

# =============================================================================
# 5. Monte Carlo Simulation Functions
# =============================================================================
def monte_carlo_simulation(current_price, volatility, n_simulations=1000, n_steps=50, drift=0.0):
    """
    Simulate future price paths using geometric Brownian motion.
    Drift is estimated from historical data.
    """
    dt = 1 / 252
    simulations = []
    for _ in range(n_simulations):
        prices = [current_price]
        for _ in range(n_steps):
            random_factor = np.random.normal(0, 1)
            price = prices[-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * random_factor * np.sqrt(dt))
            prices.append(price)
        simulations.append(prices)
    return np.array(simulations)

def calculate_ordered_probabilities(simulations, current_price, tp, sl, pip_precision):
    """
    Calculate probabilities for upward and downward moves.
    """
    up_target = current_price + (tp * pip_precision)
    up_adverse = current_price - (sl * pip_precision)
    down_target = current_price - (tp * pip_precision)
    down_adverse = current_price + (sl * pip_precision)

    up_count = 0
    down_count = 0
    total = len(simulations)

    for path in simulations:
        up_event = None
        down_event = None
        for price in path[1:]:
            if price >= up_target:
                up_event = True
                break
            if price <= up_adverse:
                up_event = False
                break
        for price in path[1:]:
            if price <= down_target:
                down_event = True
                break
            if price >= down_adverse:
                down_event = False
                break
        if up_event is None:
            up_event = abs(path[-1] - up_target) < abs(path[-1] - up_adverse)
        if down_event is None:
            down_event = abs(path[-1] - down_target) < abs(path[-1] - down_adverse)
        if up_event:
            up_count += 1
        if down_event:
            down_count += 1

    up_probability = up_count / total
    down_probability = down_count / total
    return up_probability, down_probability

# =============================================================================
# 6. Model Training Function with Detailed Metrics
# =============================================================================
def train_model(features, target, pip_diff_series):
    """
    Train classifiers using time-series cross-validation and hyperparameter tuning.
    Logs detailed performance metrics.
    Returns the best-performing model.
    """
    split_index = int(0.8 * len(features))
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]
    pip_diff_train, pip_diff_test = pip_diff_series.iloc[:split_index], pip_diff_series.iloc[split_index:]

    best_score = -np.inf
    best_model = None
    best_name = None

    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, penalty='l2'),
            'params': {
                'clf__C': [0.01, 0.1, 1, 10]
            }
        },
        'SVC': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'clf__C': [0.1, 1, 10],
                'clf__kernel': ['linear', 'rbf']
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'clf__max_depth': [5, 10, None],
                'clf__min_samples_split': [2, 5, 10]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [5, 10, None],
                'clf__min_samples_split': [2, 5]
            }
        }
    }

    cv = TimeSeriesSplit(n_splits=5)

    for name, model_dict in models.items():
        logging.info(f"Training {name}...")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', model_dict['model'])
        ])

        grid_search = GridSearchCV(
            pipeline,
            model_dict['params'],
            cv=cv,
            scoring=profit_scorer,
            n_jobs=-1,
            error_score='raise'
        )
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
            continue

        y_test_pred = grid_search.predict(X_test)
        test_profit = profit_score(y_test, y_test_pred, pip_diff_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        n_trades = 0
        n_wins = 0
        n_losses = 0
        total_winning_pips = 0.0
        total_losing_pips = 0.0
        for t, p, pip in zip(y_test, y_test_pred, pip_diff_test):
            if p == 0:
                continue
            n_trades += 1
            if p == t:
                n_wins += 1
                total_winning_pips += abs(pip)
            else:
                n_losses += 1
                total_losing_pips += abs(pip)
        net_pips = total_winning_pips - total_losing_pips

        logging.info(f"{name} - Best CV Profit Score: {grid_search.best_score_:.4f} | "
                     f"Test Profit Score: {test_profit:.4f} | Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"{name} - Trades: {n_trades}, Wins: {n_wins}, Losses: {n_losses}, "
                     f"Total Winning Pips: {total_winning_pips:.2f}, Total Losing Pips: {total_losing_pips:.2f}, "
                     f"Net Pips: {net_pips:.2f}")

        if test_profit > best_score:
            best_score = test_profit
            best_model = grid_search.best_estimator_
            best_name = name

    logging.info(f"Selected Model: {best_name} with Test Profit Score: {best_score:.4f}")
    return best_model

# =============================================================================
# 7. Trade Execution Function (Simulated)
# =============================================================================
def execute_trade(asset, direction, current_price, tp_pips, sl_pips, pip_precision):
    """
    Execute (simulate) a trade.
    Calculates TP and SL prices and logs trade details.
    Records the trade in the open_trades dictionary.
    """
    if direction == 1:  # Long trade.
         tp_price = current_price + (tp_pips * pip_precision)
         sl_price = current_price - (sl_pips * pip_precision)
    else:  # Short trade.
         tp_price = current_price - (tp_pips * pip_precision)
         sl_price = current_price + (sl_pips * pip_precision)

    logging.info(f"Executing trade for {asset}: Direction: {'Long' if direction==1 else 'Short'}, "
                 f"Entry: {current_price:.5f}, TP: {tp_price:.5f}, SL: {sl_price:.5f}")
    open_trades[asset] = {
         "direction": direction,
         "entry": current_price,
         "tp": tp_price,
         "sl": sl_price,
         "open_time": datetime.now()
    }

# =============================================================================
# 8. Main Trading Analysis Workflow
# =============================================================================
def main_trading_workflow():
    """
    Main workflow:
      - Retrieve historical data for each asset.
      - Compute technical indicators and sentiment.
      - Prepare data with enhanced labeling.
      - Train multiple models with detailed performance metrics; select best model.
      - Run Monte Carlo simulations with drift estimation.
      - Execute trades only when model prediction and simulation signal agree.
    """
    forex_pairs = [
        'GBPJPY=X', 'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X',
        'USDCAD=X', 'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ]
    all_assets = forex_pairs

    pip_precision = {
        'GBPJPY=X': 0.01, 'EURUSD=X': 0.0001, 'USDJPY=X': 0.01,
        'GBPUSD=X': 0.0001, 'AUDUSD=X': 0.0001, 'USDCAD=X': 0.0001,
        'USDCHF=X': 0.0001, 'NZDUSD=X': 0.0001, 'EURGBP=X': 0.0001,
        'EURJPY=X': 0.01
    }

    tp_pips = 50  # Profit target (50 pips)
    sl_pips = 30  # Stop-loss threshold (30 pips)
    signal_probability_threshold = 0.45  # Minimum simulation probability required

    n_simulations = 1000
    n_steps = 50

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')

    for asset in all_assets:
        df = fetch_forex_data(asset, start_date, end_date)
        if df is None or df.empty:
            logging.error(f"No data for {asset}. Skipping.")
            continue

        current_price = df['close'].iloc[-1]

        if asset in open_trades:
            trade = open_trades[asset]
            if trade['direction'] == 1:
                if current_price >= trade['tp']:
                    logging.info(f"Closing long trade for {asset} (TP reached).")
                    del open_trades[asset]
                elif current_price <= trade['sl']:
                    logging.info(f"Closing long trade for {asset} (SL reached).")
                    del open_trades[asset]
                else:
                    logging.info(f"Trade for {asset} still open; skipping new trade.")
                    continue
            else:
                if current_price <= trade['tp']:
                    logging.info(f"Closing short trade for {asset} (TP reached).")
                    del open_trades[asset]
                elif current_price >= trade['sl']:
                    logging.info(f"Closing short trade for {asset} (SL reached).")
                    del open_trades[asset]
                else:
                    logging.info(f"Trade for {asset} still open; skipping new trade.")
                    continue

        current_pip_precision = pip_precision.get(asset, 0.0001)
        sentiment = compute_sentiment_score(asset)

        try:
            features, target, pip_diff_series = prepare_data(df, tp_pips, sl_pips, current_pip_precision, sentiment)
        except Exception as e:
            logging.error(f"Error preparing data for {asset}: {e}")
            continue

        try:
            model = train_model(features, target, pip_diff_series)
        except Exception as e:
            logging.error(f"ML training error for {asset}: {e}")
            continue

        try:
            latest_features = features.iloc[-1].values.reshape(1, -1)
            model_prediction = model.predict(latest_features)[0]
        except Exception as e:
            logging.error(f"Prediction error for {asset}: {e}")
            model_prediction = 0

        try:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            drift = df['log_return'].mean()
        except Exception as e:
            logging.error(f"Error estimating drift for {asset}: {e}")
            drift = 0.0

        try:
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            if volatility == 0 or np.isnan(volatility):
                volatility = 0.01
        except Exception as e:
            logging.error(f"Error computing volatility for {asset}: {e}")
            continue

        simulations = monte_carlo_simulation(current_price, volatility, n_simulations, n_steps, drift)
        up_prob, down_prob = calculate_ordered_probabilities(simulations, current_price, tp_pips, sl_pips, current_pip_precision)

        if up_prob >= down_prob:
            sim_direction = 1
            sim_probability = up_prob
        else:
            sim_direction = -1
            sim_probability = down_prob

        logging.info(f"Asset: {asset}, Model Prediction: {model_prediction}, Simulation Direction: {sim_direction}, "
                     f"Simulation Probability: {sim_probability:.2%}, Sentiment: {sentiment:.2f}")

        if (model_prediction == sim_direction) and (sim_probability >= signal_probability_threshold):
            if asset not in open_trades:
                execute_trade(asset, sim_direction, current_price, tp_pips, sl_pips, current_pip_precision)
            else:
                logging.info(f"Trade for {asset} already open; no new trade executed.")
        else:
            logging.info(f"No trade executed for {asset} due to conflicting signals or insufficient probability.")

# =============================================================================
# 9. Streamlit App Interface
# =============================================================================
def main():
    st.title("Enhanced Trading System")
    st.markdown("""
    This application runs the enhanced trading system with detailed performance metrics.
    
    **Workflow Overview:**
    - Historical forex data is fetched for multiple assets.
    - Technical indicators and sentiment scores are computed.
    - Data is prepared and labeled with target outcomes.
    - Multiple ML models are trained and evaluated.
    - A Monte Carlo simulation estimates future price movements.
    - Trades are executed (simulated) if both model prediction and simulation signals agree.
    
    The system is automatically re-run every 5 minutes.
    """)
    
    if st.button("Run Trading Analysis Now"):
        st.info("Running analysis... This may take a few minutes. Please wait.")
        main_trading_workflow()
        st.success("Analysis completed.")
        
        if open_trades:
            st.subheader("Open Trades")
            st.write(open_trades)
        else:
            st.write("No open trades at the moment.")
        
        try:
            with open("log.txt", "r") as f:
                log_content = f.read()
            st.subheader("Log Output")
            st.text_area("Log", log_content, height=300)
        except Exception as e:
            st.error(f"Error reading log file: {e}")

if __name__ == '__main__':
    main()

