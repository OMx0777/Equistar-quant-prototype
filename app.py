import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="Quant Backtester", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for "Institutional Dark Grey" Style ---
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #2b2b2b;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    /* Text colors to match the dark theme */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #e0e0e0 !important;
    }
    /* Style the primary button */
    .stButton>button {
        background-color: #444444;
        color: white !important;
        border-radius: 8px;
        border: 1px solid #666666;
    }
    .stButton>button:hover {
        background-color: #666666;
        border-color: #888888;
    }
    /* Fix input boxes for dark mode */
    .stTextInput input, .stNumberInput input {
        background-color: #333333 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. The Strategy Class ---
class QuantitativePipeline:
    def __init__(self, ticker, fast_window, slow_window, slippage_bps):
        self.ticker = ticker
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.slippage_rate = slippage_bps / 10000 
        self.data = None

    def fetch_data(self, start_date, end_date):
        self.data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        self.data['Returns'] = self.data['Close'].pct_change()
        return self.data

    def generate_signals(self):
        self.data['Fast_SMA'] = self.data['Close'].rolling(window=self.fast_window).mean()
        self.data['Slow_SMA'] = self.data['Close'].rolling(window=self.slow_window).mean()
        
        self.data['Signal'] = np.where(self.data['Fast_SMA'] > self.data['Slow_SMA'], 1, 0)
        self.data['Volatility'] = self.data['Returns'].rolling(window=5).std()
        vol_kill_threshold = self.data['Volatility'].quantile(0.95)
        self.data['Signal'] = np.where(self.data['Volatility'] > vol_kill_threshold, 0, self.data['Signal'])
        
        self.data['Position'] = self.data['Signal'].shift(1)

    def run_backtest(self):
        clean_data = self.data.dropna().copy()
        clean_data['Trade_Occurred'] = clean_data['Position'].diff().abs()
        clean_data['Strategy_Return'] = (clean_data['Position'] * clean_data['Returns']) - (clean_data['Trade_Occurred'] * self.slippage_rate)
        
        clean_data['Market_Equity'] = (clean_data['Returns'] + 1).cumprod()
        clean_data['Strategy_Equity'] = (clean_data['Strategy_Return'] + 1).cumprod()
        
        clean_data['Peak'] = clean_data['Strategy_Equity'].cummax()
        clean_data['Drawdown'] = (clean_data['Strategy_Equity'] - clean_data['Peak']) / clean_data['Peak']
        
        total_mkt_ret = clean_data['Market_Equity'].iloc[-1] - 1
        total_strat_ret = clean_data['Strategy_Equity'].iloc[-1] - 1
        sharpe = (clean_data['Strategy_Return'].mean() / clean_data['Strategy_Return'].std()) * np.sqrt(252)
        max_dd = clean_data['Drawdown'].min()
        win_rate = len(clean_data[clean_data['Strategy_Return'] > 0]) / len(clean_data[clean_data['Strategy_Return'] != 0])

        return total_mkt_ret, total_strat_ret, sharpe, max_dd, win_rate, clean_data

# --- 2. The Streamlit UI ---

# Logo Section (Updated with modern parameter)
try:
    logo = Image.open("logo.png")
    # Fixed deprecation: using use_container_width instead of use_column_width
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.markdown("*(Upload `logo.png` to your GitHub repo to see it here)*")
    st.sidebar.markdown("---")

st.title("Algorithmic Trading Pipeline Prototype by OM")
st.markdown("**Built for Equistar Technical Screen.** Features include Volatility Kill-Switch, Look-Ahead Bias prevention, and Trade Slippage modeling.")

st.sidebar.header("⚙️ Strategy Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

st.sidebar.markdown("---")
st.sidebar.subheader("Model Weights")
fast_ma = st.sidebar.slider("Fast Moving Average", 10, 100, 50)
slow_ma = st.sidebar.slider("Slow Moving Average", 100, 300, 200)

st.sidebar.markdown("---")
st.sidebar.subheader("Execution Logic")
slippage = st.sidebar.number_input("Slippage per Trade (Basis Points)", min_value=0, max_value=50, value=5)

if st.sidebar.button("Execute Backtest", type="primary"):
    with st.spinner("Compiling data and running vectorized backtest..."):
        pipeline = QuantitativePipeline(ticker, fast_ma, slow_ma, slippage)
        pipeline.fetch_data(start_date, end_date)
        pipeline.generate_signals()
        mkt_ret, strat_ret, sharpe, max_dd, win_rate, final_data = pipeline.run_backtest()
        
        st.subheader("📈 Performance & Risk Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy Return", f"{strat_ret:.2%}", f"{(strat_ret - mkt_ret):.2%} vs Market")
        col2.metric("Sharpe Ratio", f"{sharpe:.2f}", "Risk-Adjusted")
        col3.metric("Max Drawdown", f"{max_dd:.2%}", "Worst Case Loss", delta_color="inverse")
        col4.metric("Trade Win Rate", f"{win_rate:.1%}")

        st.subheader("Equity Curve (Strategy vs. Buy & Hold)")
        fig = go.Figure()
        # Cleaned up chart colors for the dark grey theme
        fig.add_trace(go.Scatter(x=final_data.index, y=final_data['Strategy_Equity'], mode='lines', name='Strategy Equity', line=dict(color='#00e676', width=2)))
        fig.add_trace(go.Scatter(x=final_data.index, y=final_data['Market_Equity'], mode='lines', name='Market Equity (SPY)', line=dict(color='#888888', width=2, dash='dot')))
        
        # Transparent chart background so the dark grey shows through
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified", 
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(color='#e0e0e0'),
            xaxis=dict(gridcolor='#444444'),
            yaxis=dict(gridcolor='#444444')
        )
        st.plotly_chart(fig, use_container_width=True)