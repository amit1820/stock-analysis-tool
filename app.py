import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Smart Stock Analyzer", layout="wide")
st.title("📈 Stock Buy / Hold / Sell Analyzer")

# Input ticker symbol
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY)", "AAPL").upper()

# RSI calculation function
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Scoring weights
metrics_weights = {
    "P/E Ratio": 15,
    "EPS": 10,
    "ROE": 10,
    "Revenue Growth": 10,
    "Debt-to-Equity": 10,
    "MA Cross": 15,
    "RSI": 15,
    "MACD": 15
}

if ticker:
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="6mo")

    # Header info
    st.subheader(f"**{info.get('shortName', ticker)} ({ticker})**")
    st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
    st.write(f"**Market Cap:** {info.get('marketCap', 0):,}")

    # Initialize scores
    long_term_score = 0
    short_term_score = 0
    max_long = metrics_weights['P/E Ratio'] + metrics_weights['EPS'] + metrics_weights['ROE'] + metrics_weights['Revenue Growth'] + metrics_weights['Debt-to-Equity'] + metrics_weights['MA Cross']
    max_short = metrics_weights['RSI'] + metrics_weights['MACD']

    # Fundamental Analysis
    st.markdown("---")
    st.header("🧮 Fundamental & Long-Term Analysis")

    ## P/E Ratio
    pe = info.get('trailingPE')
    if pe:
        st.write(f"**P/E Ratio:** {pe:.2f}")
        if pe < 25:
            long_term_score += metrics_weights['P/E Ratio']
    else:
        st.write("**P/E Ratio:** Data not available")

    ## EPS
    eps = info.get('trailingEps')
    if eps:
        st.write(f"**EPS:** {eps:.2f}")
        if eps > 0:
            long_term_score += metrics_weights['EPS']
    else:
        st.write("**EPS:** Data not available")

    ## ROE
    roe = info.get('returnOnEquity')
    if roe:
        st.write(f"**ROE:** {roe*100:.2f}%")
        if roe > 0.15:
            long_term_score += metrics_weights['ROE']
    else:
        st.write("**ROE:** Data not available")

    ## Revenue Growth
    rev_growth = info.get('revenueGrowth')
    if rev_growth:
        st.write(f"**Revenue Growth (YoY):** {rev_growth*100:.2f}%")
        if rev_growth > 0.10:
            long_term_score += metrics_weights['Revenue Growth']
    else:
        st.write("**Revenue Growth (YoY):** Data not available")

    ## Debt-to-Equity
    de = info.get('debtToEquity')
    if de is not None:
        st.write(f"**Debt-to-Equity Ratio:** {de:.2f}")
        if de < 1:
            long_term_score += metrics_weights['Debt-to-Equity']
    else:
        st.write("**Debt-to-Equity:** Data not available")

    # Moving Averages for long-term trend
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    ma_cross = hist['MA50'].iloc[-1] > hist['MA200'].iloc[-1]
    st.write(f"**Golden Cross (MA50 > MA200):** {'Yes' if ma_cross else 'No'}")
    if ma_cross:
        long_term_score += metrics_weights['MA Cross']

    # Technical Analysis
    st.markdown("---")
    st.header("📈 Short-Term Technical Analysis")

    # RSI and MACD
    hist['RSI'] = calculate_rsi(hist)
    rsi = hist['RSI'].dropna()
    macd_line = hist['Close'].ewm(span=12, adjust=False).mean() - hist['Close'].ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Layout: 2 charts per row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RSI (Last 6mo)")
        fig1, ax1 = plt.subplots(figsize=(4,3))
        rsi.plot(ax=ax1)
        ax1.axhline(70, linestyle='--')
        ax1.axhline(30, linestyle='--')
        st.pyplot(fig1)
        if not rsi.empty and 30 < rsi.iloc[-1] < 70:
            short_term_score += metrics_weights['RSI']

    with col2:
        st.subheader("MACD vs Signal")
        fig2, ax2 = plt.subplots(figsize=(4,3))
        macd_line.plot(ax=ax2, label='MACD')
        signal_line.plot(ax=ax2, label='Signal')
        ax2.legend()
        st.pyplot(fig2)
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            short_term_score += metrics_weights['MACD']

    # Price and MAs charts in next row
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Price Trend (6mo)")
        fig3, ax3 = plt.subplots(figsize=(4,3))
        hist['Close'].plot(ax=ax3)
        st.pyplot(fig3)
    with col4:
        st.subheader("50 & 200 Day MAs")
        fig4, ax4 = plt.subplots(figsize=(4,3))
        hist['Close'].plot(ax=ax4, label='Close')
        hist['MA50'].plot(ax=ax4, label='MA50')
        hist['MA200'].plot(ax=ax4, label='MA200')
        ax4.legend()
        st.pyplot(fig4)

    # Final Recommendations
    st.markdown("---")
    st.header("🧠 Final Recommendations")
    long_pct = int((long_term_score / max_long) * 100)
    short_pct = int((short_term_score / max_short) * 100)

    # Long-term verdict
    if long_pct >= 75:
        lt_rec = "✅ Long-Term: Strong Buy"
    elif long_pct >= 50:
        lt_rec = "➖ Long-Term: Hold"
    else:
        lt_rec = "❌ Long-Term: Sell"

    # Short-term verdict
    if short_pct >= 75:
        st_rec = "✅ Short-Term: Strong Buy"
    elif short_pct >= 50:
        st_rec = "➖ Short-Term: Hold"
    else:
        st_rec = "❌ Short-Term: Sell"

    st.write(f"**Long-Term Score:** {long_pct}% | Recommendation: {lt_rec}")
    st.write(f"**Short-Term Score:** {short_pct}% | Recommendation: {st_rec}")
