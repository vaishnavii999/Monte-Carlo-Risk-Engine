# 🎲 Monte Carlo Risk Engine

**A Real-Time Quantitative Finance Dashboard built with Python.**

This application models market volatility and forecasts future price paths using stochastic processes. It integrates live market data to calculate Value at Risk (VaR) and perform Monte Carlo simulations on the fly.

## 🚀 Key Features
* **Real-Time Data Pipeline:** Fetches live ticks using `yfinance` to simulate an institutional feed.
* **Stochastic Modeling:** Runs 1,000+ Monte Carlo simulations based on Geometric Brownian Motion (GBM).
* **Risk Analytics:** Calculates Parametric VaR (95% Confidence) and Annualized Volatility.
* **Dynamic Visualization:** Interactive Plotly charts that update without page refreshes.

## 🛠️ Tech Stack
* **Python 3.10+**
* **Streamlit** (Frontend/Server)
* **NumPy & SciPy** (Statistical Math)
* **Plotly** (Interactive Graphing)
## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/vaishnavii999/Monte-Carlo-Risk-Engine.git](https://github.com/vaishnavii999/Monte-Carlo-Risk-Engine.git)

2) Install dependencies:
pip install -r requirements.txt
3) Launch the dashboard:
streamlit run app.py
