import yfinance as yf
import pandas as pd
import numpy as np

class MovingAverageStrategy:
    def __init__(self, ticker, fast_window, slow_window):
        self.ticker = ticker
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.data = None

    def fetch_data(self, start_date, end_date):
        """Step 1: The Data Pipeline"""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        # Calculate daily returns
        self.data['Returns'] = self.data['Close'].pct_change()
        return self.data

    def generate_signals(self):
        """Step 2 & 3: Features, Signals, and Risk Checks"""
        # Features: Calculate Moving Averages
        self.data['Fast_SMA'] = self.data['Close'].rolling(window=self.fast_window).mean()
        self.data['Slow_SMA'] = self.data['Close'].rolling(window=self.slow_window).mean()
        
        # Signal Logic: 1 (Buy) if Fast > Slow, else 0 (Flat)
        self.data['Signal'] = np.where(self.data['Fast_SMA'] > self.data['Slow_SMA'], 1, 0)
        
        # Risk Check (The Kill-Switch): If rolling 5-day volatility is in the top 5%, cut the signal
        self.data['Volatility'] = self.data['Returns'].rolling(window=5).std()
        vol_kill_threshold = self.data['Volatility'].quantile(0.95)
        self.data['Signal'] = np.where(self.data['Volatility'] > vol_kill_threshold, 0, self.data['Signal'])
        
        # CRITICAL: Shift position by 1 day to prevent Look-Ahead Bias
        # You calculate the signal today, but you trade it tomorrow.
        self.data['Position'] = self.data['Signal'].shift(1)

    def run_backtest(self):
        """Step 4: Vectorized Backtest & Metrics"""
        # Calculate strategy returns based on our position
        self.data['Strategy_Return'] = self.data['Position'] * self.data['Returns']
        
        # Drop NaN values created by rolling windows
        clean_data = self.data.dropna()
        
        # Calculate Performance Metrics
        total_return = (clean_data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
        market_return = (clean_data['Returns'] + 1).cumprod().iloc[-1] - 1
        
        # Annualized Sharpe Ratio (Assuming ~252 trading days)
        sharpe_ratio = (clean_data['Strategy_Return'].mean() / clean_data['Strategy_Return'].std()) * np.sqrt(252)
        
        print("\n--- Backtest Results ---")
        print(f"Market Return (Buy & Hold):  {market_return:.2%}")
        print(f"Strategy Return:             {total_return:.2%}")
        print(f"Sharpe Ratio:                {sharpe_ratio:.2f}")
        print("------------------------\n")

if __name__ == "__main__":
    # Initialize the pipeline for the S&P 500 ETF (SPY)
    pipeline = MovingAverageStrategy(ticker="SPY", fast_window=50, slow_window=200)
    
    # Run the system over a 4-year out-of-sample period
    pipeline.fetch_data(start_date="2020-01-01", end_date="2024-01-01")
    pipeline.generate_signals()
    pipeline.run_backtest()