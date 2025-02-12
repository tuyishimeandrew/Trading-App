import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
from numba import njit  # Numba for compilation

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

# Import GARCH calibration functionality.
from arch import arch_model

# Pre-import TA libraries to avoid re-importing them for each call.
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands

# Setup logging to capture training performance and error messages.
logging.basicConfig(
    filename="C:\\Users\\Andrew\\Downloads\\log.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download NLTK data for sentiment analysis.
nltk.download('vader_lexicon')

# =============================================================================
# Global Configuration for MT4 (using the templer server)
# =============================================================================
MT4_SERVER = "templer markets mt4-usd"
MT4_LOGIN = "824195"
MT4_PASSWORD = "DYXTY43"
MT4_PORT = 443  # Adjust as needed

# Dictionary to track open trades; key is asset symbol, value contains trade details.
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

NEWSAPI_API_KEY = "eb91a6486d80448688dcf703a949d4a6"  # Replace with your actual key

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

# -----------------------------------------------------------------------------
# Optimized target and pip difference computation using Numba.
# -----------------------------------------------------------------------------
@njit
def compute_targets_numba(closes, highs, lows, adjusted_tp, adverse, max_horizon, tp, transaction_cost):
    n = len(closes)
    targets = np.zeros(n)
    pip_diff = np.zeros(n)
    for i in range(n):
        base = closes[i]
        outcome = 0
        # Look ahead up to max_horizon bars
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
    return targets, pip_diff

def prepare_data(df, tp, sl, pip_precision, sentiment_score, max_horizon=50, transaction_cost=0.00005):
    """
    Prepare data by creating target labels and pip differences.
    """
    df = df.copy()
    df = add_additional_features(df)
    df = add_ta_indicators(df)

    adjusted_tp = tp * pip_precision
    adverse = sl * pip_precision

    n = df.shape[0]
    closes = df['close'].values
    highs = df['High'].values
    lows = df['Low'].values

    # Call the optimized function
    targets, pip_diff = compute_targets_numba(closes, highs, lows, adjusted_tp, adverse, max_horizon, tp, transaction_cost)

    df['target'] = targets
    df['pip_diff'] = pip_diff
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    features = df.drop(columns=['target', 'datetime', 'pip_diff'])
    target = df['target']
    pip_diff_series = df['pip_diff']
    return features, target, pip_diff_series

# =============================================================================
# 4. Custom Profit Score Function for Model Evaluation (used in backtesting)
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

# =============================================================================
# Backtesting Metrics Functions
# =============================================================================
def calculate_sharpe_ratio(trade_returns, risk_free_rate=0.0):
    if np.std(trade_returns) == 0:
        return 0
    return np.mean(trade_returns - risk_free_rate) / np.std(trade_returns - risk_free_rate) * np.sqrt(252)

def calculate_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - peak
    return drawdown.min()

def calculate_profit_factor(trade_returns):
    wins = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    if losses == 0:
        return float('inf')
    return wins / losses

def backtest_strategy(model, features, target, pip_diff_series):
    predictions = model.predict(features)
    trade_returns = []
    for pred, actual, pip in zip(predictions, target, pip_diff_series):
        if pred == 0:
            continue
        if pred == actual:
            trade_returns.append(abs(pip))
        else:
            trade_returns.append(-abs(pip))
    trade_returns = np.array(trade_returns)
    if len(trade_returns) == 0:
        return {"sharpe_ratio": 0, "max_drawdown": 0, "profit_factor": 0}
    equity_curve = np.cumsum(trade_returns)
    sharpe = calculate_sharpe_ratio(trade_returns)
    mdd = calculate_max_drawdown(equity_curve)
    pf = calculate_profit_factor(trade_returns)
    return {"sharpe_ratio": sharpe, "max_drawdown": mdd, "profit_factor": pf}

# -----------------------------------------------------------------------------
# A helper function to backtest and also return the equity curve.
# -----------------------------------------------------------------------------
def backtest_and_get_curve(model, features, target, pip_diff_series):
    predictions = model.predict(features)
    trade_returns = []
    for pred, actual, pip in zip(predictions, target, pip_diff_series):
        if pred == 0:
            continue
        if pred == actual:
            trade_returns.append(abs(pip))
        else:
            trade_returns.append(-abs(pip))
    trade_returns = np.array(trade_returns)
    if len(trade_returns) == 0:
        return {"metrics": {"sharpe_ratio": 0, "max_drawdown": 0, "profit_factor": 0},
                "equity_curve": np.array([]),
                "trade_returns": trade_returns}
    equity_curve = np.cumsum(trade_returns)
    sharpe = calculate_sharpe_ratio(trade_returns)
    mdd = calculate_max_drawdown(equity_curve)
    pf = calculate_profit_factor(trade_returns)
    metrics = {"sharpe_ratio": sharpe, "max_drawdown": mdd, "profit_factor": pf}
    return {"metrics": metrics, "equity_curve": equity_curve, "trade_returns": trade_returns}

# =============================================================================
# Advanced Calibration Techniques using GARCH
# =============================================================================
def calibrate_parameters(prices):
    if len(prices) < 30:
        return {"drift": 0.0, "initial_vol": 0.01, "drift_crisis": -0.01, "vol_crisis": 0.03}
    log_returns = np.diff(np.log(prices)) * 100  
    try:
        am = arch_model(log_returns, vol='Garch', p=1, o=0, q=1, dist='normal')
        res = am.fit(disp='off')
        initial_vol = res.conditional_volatility[-1] / 100.0 * np.sqrt(252)
        drift_est = np.mean(log_returns) / 100.0 * 252
    except Exception as e:
        logging.error(f"GARCH calibration error: {e}")
        initial_vol = np.std(log_returns) / 100.0 * np.sqrt(252)
        drift_est = np.mean(log_returns) / 100.0 * 252

    drift_crisis = drift_est - 0.02
    vol_crisis = initial_vol * 1.5
    return {"drift": drift_est, "initial_vol": initial_vol, "drift_crisis": drift_crisis, "vol_crisis": vol_crisis}

# -----------------------------------------------------------------------------
# Optimized Hybrid Monte Carlo Simulation using Numba.
# -----------------------------------------------------------------------------
@njit
def monte_carlo_simulation_hybrid_numba(current_price, n_simulations, n_steps, dt,
                                        drift, initial_vol, kappa, theta, sigma_v,
                                        rho, jump_lambda, jump_mu, jump_sigma,
                                        regime_switch_prob, drift_crisis, vol_crisis):
    simulations = np.empty((n_simulations, n_steps+1))
    for sim in range(n_simulations):
        price = current_price
        V = initial_vol**2
        regime = 1
        simulations[sim, 0] = price
        for step in range(1, n_steps+1):
            if np.random.rand() < regime_switch_prob:
                regime = 2 if regime == 1 else 1
            drift_eff = drift if regime == 1 else drift_crisis
            z1 = np.random.randn()
            z_indep = np.random.randn()
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z_indep
            V = np.abs(V + kappa * (theta - V) * dt + sigma_v * np.sqrt(V * dt) * z2)
            jump = 0.0
            if np.random.rand() < jump_lambda * dt:
                jump = jump_mu + jump_sigma * np.random.randn()
            price = price * np.exp((drift_eff - 0.5 * V) * dt + np.sqrt(V * dt) * z1 + jump)
            simulations[sim, step] = price
    return simulations

def monte_carlo_simulation_hybrid(current_price, n_simulations=1000, n_steps=50, dt=1/252,
                                  drift=0.0, initial_vol=0.01, kappa=1.5, theta=0.01, sigma_v=0.1,
                                  rho=0.5, jump_lambda=0.1, jump_mu=-0.05, jump_sigma=0.1,
                                  regime_switch_prob=0.05, drift_crisis=-0.01, vol_crisis=0.03):
    return monte_carlo_simulation_hybrid_numba(current_price, n_simulations, n_steps, dt,
                                               drift, initial_vol, kappa, theta, sigma_v,
                                               rho, jump_lambda, jump_mu, jump_sigma,
                                               regime_switch_prob, drift_crisis, vol_crisis)

def calculate_ordered_probabilities(simulations, current_price, tp, sl, pip_precision):
    up_target = current_price + (tp * pip_precision)
    up_adverse = current_price - (sl * pip_precision)
    down_target = current_price - (tp * pip_precision)
    down_adverse = current_price + (sl * pip_precision)

    up_count = 0
    down_count = 0
    total = simulations.shape[0]

    for i in range(total):
        path = simulations[i]
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
# 6. Trade Execution Functionality (Simulated)
# =============================================================================
def execute_trade(asset, direction, current_price, tp_pips, sl_pips, pip_precision):
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
# 7. Model Training Function (Using profit-based scoring for training)
# =============================================================================
def profit_scorer(y_true, y_pred, pip_diff_series):
    return profit_score(y_true, y_pred, pip_diff_series)

def train_model(features, target, train_pip_diff):
    """
    Train a model using GridSearchCV over multiple classifiers and return the best estimator.
    We use a custom profit-based scorer instead of accuracy.
    """
    pipelines = {
        'lr': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
        ]),
        'svc': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True))
        ]),
        'dt': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', DecisionTreeClassifier())
        ]),
        'rf': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier())
        ])
    }

    param_grid = {
        'lr': {
            'clf__C': [0.1, 1.0, 10.0]
        },
        'svc': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__kernel': ['rbf', 'linear']
        },
        'dt': {
            'clf__max_depth': [3, 5, 7, None]
        },
        'rf': {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [3, 5, 7, None]
        }
    }

    best_score = -np.inf
    best_model = None
    tscv = TimeSeriesSplit(n_splits=5)
    # Create a custom scorer that includes train_pip_diff as an extra argument.
    custom_scorer = make_scorer(profit_scorer, pip_diff_series=train_pip_diff, greater_is_better=True)
    for name, pipeline in pipelines.items():
        grid = GridSearchCV(pipeline, param_grid[name], scoring=custom_scorer, cv=tscv, n_jobs=-1)
        grid.fit(features, target)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
    return best_model

# =============================================================================
# 8. Main Trading Analysis Workflow
# =============================================================================
def main():
    """
    Main workflow:
      - Retrieve historical data for each asset.
      - Compute technical indicators and sentiment.
      - Prepare data with enhanced labeling.
      - Split data into training and test sets (time-based) to avoid overfitting.
      - Train multiple models and select the best one using profit-based scoring.
      - Compute backtesting metrics on the test set.
      - Display the chosen model and its performance.
      - Calibrate drift/volatility using GARCH.
      - Run the hybrid Monte Carlo simulation.
      - Execute trades when model and simulation signals agree.
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
            continue

        current_price = df['close'].iloc[-1]

        # Manage existing open trades.
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

        # Split data into training (80%) and test (20%) sets in a time-aware manner.
        split_index = int(len(features) * 0.8)
        train_features = features.iloc[:split_index]
        train_target = target.iloc[:split_index]
        train_pip_diff = pip_diff_series.iloc[:split_index]
        test_features = features.iloc[split_index:]
        test_target = target.iloc[split_index:]
        test_pip_diff = pip_diff_series.iloc[split_index:]

        try:
            model = train_model(train_features, train_target, train_pip_diff)
        except Exception as e:
            logging.error(f"ML training error for {asset}: {e}")
            continue

        # Evaluate model on the test set.
        try:
            test_predictions = model.predict(test_features)
            test_accuracy = accuracy_score(test_target, test_predictions)
            winning_count = 0
            losing_count = 0
            winning_pips = 0
            losing_pips = 0
            for pred, actual, pip in zip(test_predictions, test_target, test_pip_diff):
                if pred == 0:
                    continue
                if pred == actual:
                    winning_count += 1
                    winning_pips += abs(pip)
                else:
                    losing_count += 1
                    losing_pips += abs(pip)
            # Print chosen model details and evaluation metrics.
            print("\n====================================")
            print(f"Asset: {asset}")
            print("Chosen Model:")
            print(model)
            print(f"Test Accuracy: {test_accuracy:.2%}")
            print(f"Winning Trades: {winning_count} trades, Total Winning Pips: {winning_pips}")
            print(f"Losing Trades: {losing_count} trades, Total Losing Pips: {losing_pips}")
            print("====================================\n")
            logging.info(f"Asset: {asset}, Test Accuracy: {test_accuracy:.2%}, "
                         f"Winning Trades: {winning_count}, Losing Trades: {losing_count}")
        except Exception as e:
            logging.error(f"Error evaluating model on test set for {asset}: {e}")

        try:
            latest_features = test_features.iloc[-1].values.reshape(1, -1)
            model_prediction = model.predict(latest_features)[0]
        except Exception as e:
            logging.error(f"Prediction error for {asset}: {e}")
            model_prediction = 0

        try:
            calibration = calibrate_parameters(df['close'].values)
            drift = calibration["drift"]
            initial_vol = calibration["initial_vol"]
            drift_crisis = calibration["drift_crisis"]
            vol_crisis = calibration["vol_crisis"]
        except Exception as e:
            logging.error(f"Error estimating calibration for {asset}: {e}")
            drift = 0.0
            initial_vol = 0.01
            drift_crisis = -0.01
            vol_crisis = 0.03

        try:
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            if volatility == 0 or np.isnan(volatility):
                volatility = 0.01
        except Exception as e:
            logging.error(f"Error computing volatility for {asset}: {e}")
            continue

        simulations = monte_carlo_simulation_hybrid(
            current_price,
            n_simulations=n_simulations,
            n_steps=n_steps,
            dt=1/252,
            drift=drift,
            initial_vol=initial_vol,
            kappa=1.5,
            theta=0.01,
            sigma_v=0.1,
            rho=0.5,
            jump_lambda=0.1,
            jump_mu=-0.05,
            jump_sigma=0.1,
            regime_switch_prob=0.05,
            drift_crisis=drift_crisis,
            vol_crisis=vol_crisis
        )
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

if __name__ == '__main__':
    # If you want to run the original trading analysis workflow, uncomment the next line:
    # main()
    pass

# =============================================================================
# STREAMLIT BACKTESTING APP
# =============================================================================
import streamlit as st

st.title("Forex Strategy Backtest")

@st.cache(show_spinner=False, allow_output_mutation=True)
def run_backtest():
    results = {}
    forex_pairs = [
        'GBPJPY=X', 'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X',
        'USDCAD=X', 'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ]
    pip_precision = {
        'GBPJPY=X': 0.01, 'EURUSD=X': 0.0001, 'USDJPY=X': 0.01,
        'GBPUSD=X': 0.0001, 'AUDUSD=X': 0.0001, 'USDCAD=X': 0.0001,
        'USDCHF=X': 0.0001, 'NZDUSD=X': 0.0001, 'EURGBP=X': 0.0001,
        'EURJPY=X': 0.01
    }
    tp_pips = 50
    sl_pips = 30

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')

    for asset in forex_pairs:
        df = fetch_forex_data(asset, start_date, end_date)
        if df is None or df.empty:
            continue

        try:
            sentiment = compute_sentiment_score(asset)
        except Exception as e:
            sentiment = 0.0

        current_pip_precision = pip_precision.get(asset, 0.0001)
        try:
            features, target, pip_diff_series = prepare_data(df, tp_pips, sl_pips, current_pip_precision, sentiment)
        except Exception as e:
            logging.error(f"Error preparing data for {asset}: {e}")
            continue

        split_index = int(len(features) * 0.8)
        train_features = features.iloc[:split_index]
        train_target = target.iloc[:split_index]
        train_pip_diff = pip_diff_series.iloc[:split_index]
        test_features = features.iloc[split_index:]
        test_target = target.iloc[split_index:]
        test_pip_diff = pip_diff_series.iloc[split_index:]

        try:
            model = train_model(train_features, train_target, train_pip_diff)
        except Exception as e:
            logging.error(f"ML training error for {asset}: {e}")
            continue

        try:
            test_predictions = model.predict(test_features)
            test_accuracy = accuracy_score(test_target, test_predictions)
        except Exception as e:
            test_accuracy = 0

        backtest_results = backtest_and_get_curve(model, test_features, test_target, test_pip_diff)
        metrics = backtest_results["metrics"]
        equity_curve = backtest_results["equity_curve"]
        trade_returns = backtest_results["trade_returns"]
        profit = profit_score(test_target, test_predictions, test_pip_diff)
        results[asset] = {
            "accuracy": test_accuracy,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "profit_factor": metrics["profit_factor"],
            "profit_score": profit,
            "equity_curve": equity_curve,
            "trade_returns": trade_returns
        }
    return results

if st.button("Run Backtest"):
    st.write("Running backtest... (this may take a moment)")
    results = run_backtest()
    if not results:
        st.write("No backtesting results available.")
    else:
        for asset, res in results.items():
            st.subheader(f"Asset: {asset}")
            st.write(f"Test Accuracy: {res['accuracy']:.2%}")
            st.write(f"Sharpe Ratio: {res['sharpe_ratio']:.2f}")
            st.write(f"Max Drawdown: {res['max_drawdown']:.2f}")
            st.write(f"Profit Factor: {res['profit_factor']:.2f}")
            st.write(f"Profit Score: {res['profit_score']:.2f}")
            if res['equity_curve'].size > 0:
                fig, ax = plt.subplots()
                ax.plot(res["equity_curve"])
                ax.set_title(f"Equity Curve for {asset}")
                ax.set_xlabel("Trades")
                ax.set_ylabel("Equity")
                st.pyplot(fig)
            st.markdown("---")
