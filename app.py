import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import time

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Risk Dashboard")

# --- HELPER FUNCTIONS (The Math) ---

def calculate_var(returns, confidence_level=0.95):
    """
    Calculates Parametric Value at Risk (VaR).
    """
    if returns.empty: return 0, 0
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    var_pct = norm.ppf(1 - confidence_level, mu, sigma)
    
    return var_pct, sigma

def monte_carlo_sim(start_price, mu, sigma, days=30, simulations=100):
    """
    Runs Monte Carlo simulation for future price paths.
    """
    dt = 1
    # Generate random shocks
    shocks = np.random.normal(0, 1, (days, simulations))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * shocks
    
    # Calculate returns and price paths
    daily_returns = np.exp(drift + diffusion)
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = start_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths

# --- DASHBOARD LAYOUT ---

st.title("⚡ Real-Time Quant Risk Dashboard")

# Sidebar for controls
with st.sidebar:
    ticker = st.text_input("Ticker Symbol", value="NVDA") # NVDA is volatile/fun to watch
    st.info("US Markets are OPEN. You will see live updates.")
    confidence = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95)
    sim_days = st.slider("Monte Carlo Forecast (Days)", 10, 252, 30)

# Main structure
col1, col2 = st.columns([2, 1])

# Placeholders for dynamic content
with col1:
    price_chart_placeholder = st.empty()
with col2:
    metrics_placeholder = st.empty()
    risk_chart_placeholder = st.empty()

# --- THE REAL-TIME LOOP ---
st.write("---")
st.subheader(f"Monte Carlo Simulation: {ticker} Future Paths")
mc_chart_placeholder = st.empty()

# Logic to stop the loop if user changes ticker
if 'last_ticker' not in st.session_state:
    st.session_state['last_ticker'] = ticker

while True:
    # 1. Fetch Data (1 Year for stats, 1 Day for live chart)
    # We fetch 1y to calculate Volatility (Sigma) accurately
    hist_data = yf.download(ticker, period='1y', interval='1d', progress=False)
    
    # We fetch intraday for the "Live" chart
    live_data = yf.download(ticker, period='1d', interval='1m', progress=False)
    
    if not hist_data.empty and not live_data.empty:
        
        # --- FIX: ROBUST DATA PARSING ---
        # Handle yfinance returning DataFrames instead of Scalars
        try:
            # Extract Current Price
            if isinstance(live_data['Close'], pd.DataFrame):
                current_price = float(live_data['Close'].iloc[-1].iloc[0])
            else:
                current_price = float(live_data['Close'].iloc[-1])
            
            # Extract Previous Close (for % change)
            if isinstance(hist_data['Close'], pd.DataFrame):
                prev_close = float(hist_data['Close'].iloc[-2].iloc[0])
            else:
                prev_close = float(hist_data['Close'].iloc[-2])
                
        except Exception as e:
            st.error(f"Error parsing data: {e}")
            time.sleep(2)
            continue
        # -------------------------------

        # 2. Calculate Risk Metrics
        # Flatten returns if it's a DataFrame
        returns = hist_data['Close'].pct_change().dropna()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
            
        var_pct, sigma = calculate_var(returns, confidence)
        
        # Annualize volatility
        annual_vol = sigma * np.sqrt(252)
        
        # 3. Update Metrics Column
        with metrics_placeholder.container():
            # Calculate delta
            delta_val = ((current_price - prev_close)/prev_close)*100
            
            st.metric(label="Current Price", value=f"${current_price:.2f}", 
                      delta=f"{delta_val:.2f}%")
            
            st.metric(label="Annualized Volatility", value=f"{annual_vol*100:.1f}%")
            
            # Value at Risk display
            st.error(f"Daily VaR ({confidence*100:.0f}%): {var_pct*100:.2f}%")
            
            # --- FIX: STRING FORMATTING ---
            st.caption(f"Meaning: With {confidence*100:.0f}% confidence, you won't lose more than {abs(var_pct*100):.2f}% in a day.")

        # 4. Update Live Price Chart
        with price_chart_placeholder.container():
            fig_price = go.Figure()
            
            # Ensure y-data is 1D
            y_data = live_data['Close']
            if isinstance(y_data, pd.DataFrame):
                y_data = y_data.iloc[:, 0]
                
            fig_price.add_trace(go.Scatter(x=live_data.index, y=y_data, mode='lines', name='Price'))
            fig_price.update_layout(title=f"Intraday Live Feed: {ticker}", height=350, margin=dict(l=0,r=0,t=30,b=0))
            # We use time.time() to give it a unique ID every second
            st.plotly_chart(fig_price, use_container_width=True, key=f"price_{time.time()}")

        # 5. Run Monte Carlo (Only needs to update occasionally, but we'll do it per loop for effect)
        # Using the last historical mean/std for simulation
        mu = np.mean(returns)
        sim_paths = monte_carlo_sim(current_price, mu, sigma, days=sim_days, simulations=50)
        
        with mc_chart_placeholder.container():
            fig_mc = go.Figure()
            # Plot first 20 paths
            for i in range(20):
                fig_mc.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', opacity=0.4, showlegend=False))
            
            fig_mc.update_layout(title=f"Monte Carlo: {sim_days} Day Forecast", height=300, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_mc, use_container_width=True, key=f"mc_{time.time()}")

    # Refresh Rate
    time.sleep(5) # Wait 5 seconds before next pull
    
    # Check if user changed ticker to break loop (simple reload mechanism)
    if ticker != st.session_state['last_ticker']:
        st.session_state['last_ticker'] = ticker
        st.rerun()