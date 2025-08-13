import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Smart Stock Analyzer", layout="wide")
st.title("📈 Stock Buy / Hold / Sell Analyzer")

# ---------------------------
# Configurable analysis windows
# ---------------------------
LONG_PERIOD = "3y"     # long-term analysis window (at least 3 years)
SHORT_PERIOD = "3mo"   # short-term analysis window (at least 2–3 months)

# --- Load ticker/name list for autocomplete (optional) ---
# Place a CSV named 'tickers.csv' in the project folder with columns 'Ticker' and 'Name'.
try:
    tickers_df = pd.read_csv('tickers.csv')
    ticker_options = tickers_df.apply(lambda x: f"{x['Ticker']} - {x['Name']}", axis=1).tolist()
    search_input = st.text_input("Search by Ticker or Company Name:")
    if search_input:
        filtered = [opt for opt in ticker_options if search_input.lower() in opt.lower()]
        if not filtered:
            st.warning("No matches found. Please refine your search or enter a valid ticker.")
        selected = st.selectbox("Select a Stock:", filtered if filtered else [])
    else:
        selected = st.selectbox("Select a Stock:", ticker_options)
    ticker = selected.split(' - ')[0]
except FileNotFoundError:
    st.warning("tickers.csv not found. Falling back to manual input.")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY):", "AAPL").upper()

# --- RSI calculation function ---
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

    # ---------------------------
    # Fetch fundamentals & price history
    # ---------------------------
    info = stock.info
    hist_long = stock.history(period=LONG_PERIOD)   # 3 years
    hist_short = stock.history(period=SHORT_PERIOD) # ~3 months

    # Header
    st.subheader(f"**{info.get('shortName', ticker)} ({ticker})**")
    st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
    st.write(f"**Market Cap:** {info.get('marketCap', 0):,}")

    # Explanations
    st.markdown("---")
    st.header("📚 Metric Explanations")
    st.markdown("""
- **P/E Ratio (Price-to-Earnings):** Share price divided by earnings per share. Lower (<25) may indicate undervaluation.
- **EPS (Earnings Per Share):** Net profit divided by shares outstanding. Positive signals profitability.
- **ROE (Return on Equity):** Net income relative to shareholder equity. >15% shows efficient capital use.
- **Revenue Growth (YoY):** Year-over-year sales increase. >10% indicates healthy expansion.
- **Debt-to-Equity Ratio:** Company’s debt vs. equity. <1 (or <100%) means conservative leverage.
- **MA Cross (Golden Cross):** 50-day MA surpasses 200-day MA – a bullish long-term signal. Calculated on **3-year** price history.
- **RSI (Relative Strength Index):** Momentum oscillator (0–100). 30–70 neutral; >70 overbought; <30 oversold. Calculated on **last 3 months**.
- **MACD (Moving Average Convergence Divergence):** Momentum via EMA differences. Bullish when MACD > signal. Calculated on **last 3 months**.
""")

    # Initialize scores and contributions
    long_term_score = 0
    short_term_score = 0
    contributions = []
    # maxima for normalization
    max_long = sum([metrics_weights[m] for m in ["P/E Ratio","EPS","ROE","Revenue Growth","Debt-to-Equity","MA Cross"]])
    max_short = metrics_weights['RSI'] + metrics_weights['MACD']

    # ---------------------------
    # Fundamental & Long-Term Analysis (3 YEARS)
    # ---------------------------
    st.markdown("---")
    st.header("🧮 Fundamental & Long-Term Analysis (3 Years)")

    # P/E
    pe = info.get('trailingPE')
    pe_pass = bool(pe) and pe < 25
    st.write(f"**P/E Ratio:** {f'{pe:.2f}' if pe else 'N/A'}")
    contributions.append(("P/E Ratio", f"{pe:.2f}" if pe else 'N/A', '<25', metrics_weights['P/E Ratio'] if pe_pass else 0, 'Fundamental'))
    if pe_pass:
        long_term_score += metrics_weights['P/E Ratio']

    # EPS
    eps = info.get('trailingEps')
    eps_pass = bool(eps) and eps > 0
    st.write(f"**EPS:** {f'{eps:.2f}' if eps else 'N/A'}")
    contributions.append(("EPS", f"{eps:.2f}" if eps else 'N/A', '>0', metrics_weights['EPS'] if eps_pass else 0, 'Fundamental'))
    if eps_pass:
        long_term_score += metrics_weights['EPS']

    # ROE
    roe = info.get('returnOnEquity')
    roe_pct = roe*100 if roe else None
    roe_pass = bool(roe) and roe > 0.15
    st.write(f"**ROE:** {f'{roe_pct:.2f}%' if roe_pct is not None else 'N/A'}")
    contributions.append(("ROE (%)", f"{roe_pct:.2f}%" if roe_pct is not None else 'N/A', '>15%', metrics_weights['ROE'] if roe_pass else 0, 'Fundamental'))
    if roe_pass:
        long_term_score += metrics_weights['ROE']

    # Revenue Growth
    rg = info.get('revenueGrowth')
    rg_pct = rg*100 if rg else None
    rg_pass = bool(rg) and rg > 0.10
    st.write(f"**Revenue Growth (YoY):** {f'{rg_pct:.2f}%' if rg_pct is not None else 'N/A'}")
    contributions.append(("Revenue Growth (%)", f"{rg_pct:.2f}%" if rg_pct is not None else 'N/A', '>10%', metrics_weights['Revenue Growth'] if rg_pass else 0, 'Fundamental'))
    if rg_pass:
        long_term_score += metrics_weights['Revenue Growth']

    # Debt/Equity
    de = info.get('debtToEquity')
    de_pass = (de is not None) and (de < 1)
    st.write(f"**Debt-to-Equity:** {de if de is not None else 'N/A'}")
    contributions.append(("Debt-to-Equity", de if de is not None else 'N/A', '<1', metrics_weights['Debt-to-Equity'] if de_pass else 0, 'Fundamental'))
    if de_pass:
        long_term_score += metrics_weights['Debt-to-Equity']

    # Long-term Moving Averages (computed on 3Y data)
    hist_long = hist_long.copy()
    hist_long['MA50'] = hist_long['Close'].rolling(50).mean()
    hist_long['MA200'] = hist_long['Close'].rolling(200).mean()
    ma_cross = bool(len(hist_long) > 200) and (hist_long['MA50'].iloc[-1] > hist_long['MA200'].iloc[-1])
    st.write(f"**Golden Cross (50D > 200D on {LONG_PERIOD} data):** {'Yes' if ma_cross else 'No'}")
    contributions.append(("Golden Cross", 'Yes' if ma_cross else 'No', 'Yes', metrics_weights['MA Cross'] if ma_cross else 0, f'Price ({LONG_PERIOD})'))
    if ma_cross:
        long_term_score += metrics_weights['MA Cross']

    # ---------------------------
    # Short-Term Technical Analysis (LAST 3 MONTHS)
    # ---------------------------
    st.markdown("---")
    st.header("📈 Short-Term Technical Analysis (Last 3 Months)")

    # RSI / MACD on short window
    hist_short = hist_short.copy()
    hist_short['RSI'] = calculate_rsi(hist_short)
    rsi_series = hist_short['RSI'].dropna()
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
    rsi_pass = (rsi_val is not None) and (30 < rsi_val < 70)
    st.write(f"**RSI:** {round(rsi_val,2) if rsi_val is not None else 'N/A'}")
    contributions.append(("RSI", round(rsi_val,2) if rsi_val is not None else 'N/A', '30-70', metrics_weights['RSI'] if rsi_pass else 0, f'Price ({SHORT_PERIOD})'))
    if rsi_pass:
        short_term_score += metrics_weights['RSI']

    macd_line = hist_short['Close'].ewm(span=12, adjust=False).mean() - hist_short['Close'].ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_pass = (len(macd_line.dropna())>0 and len(signal_line.dropna())>0 and macd_line.iloc[-1] > signal_line.iloc[-1])
    st.write(f"**MACD Signal:** {'Bullish' if macd_pass else 'Bearish'}")
    contributions.append(("MACD", 'Bullish' if macd_pass else 'Bearish', 'Bullish', metrics_weights['MACD'] if macd_pass else 0, f'Price ({SHORT_PERIOD})'))
    if macd_pass:
        short_term_score += metrics_weights['MACD']

    # ---------------------------
    # Calculation Details (show windows used)
    # ---------------------------
    st.markdown("---")
    st.header("🔢 Calculation Details & Contributions")
    df = pd.DataFrame(contributions, columns=['Metric','Value','Threshold','Contribution','Window'])
    st.table(df)

    # ---------------------------
    # Charts: small, two per row
    # ---------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Price Trend – Long Term ({LONG_PERIOD})")
        fig1, ax1 = plt.subplots(figsize=(4,3))
        hist_long['Close'].plot(ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.subheader(f"50D & 200D MAs – ({LONG_PERIOD})")
        fig2, ax2 = plt.subplots(figsize=(4,3))
        hist_long['Close'].plot(ax=ax2, label='Close')
        hist_long['MA50'].plot(ax=ax2, label='MA50')
        hist_long['MA200'].plot(ax=ax2, label='MA200')
        ax2.legend()
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader(f"RSI – Short Term ({SHORT_PERIOD})")
        fig3, ax3 = plt.subplots(figsize=(4,3))
        if not rsi_series.empty:
            rsi_series.plot(ax=ax3)
        ax3.axhline(70, linestyle='--')
        ax3.axhline(30, linestyle='--')
        st.pyplot(fig3)
    with col4:
        st.subheader(f"MACD – Short Term ({SHORT_PERIOD})")
        fig4, ax4 = plt.subplots(figsize=(4,3))
        macd_line.plot(ax=ax4, label='MACD')
        signal_line.plot(ax=ax4, label='Signal')
        ax4.legend()
        st.pyplot(fig4)

    # ---------------------------
    # Final Recommendations (separate LT/ ST)
    # ---------------------------
    st.markdown("---")
    st.header("🧠 Final Recommendations")
    long_pct = int((long_term_score / max_long) * 100)
    short_pct = int((short_term_score / max_short) * 100)

    lt_rec = '✅ Long-Term: Strong Buy' if long_pct >= 75 else ('➖ Long-Term: Hold' if long_pct >= 50 else '❌ Long-Term: Sell')
    st_rec_val = '✅ Short-Term: Strong Buy' if short_pct >= 75 else ('➖ Short-Term: Hold' if short_pct >= 50 else '❌ Short-Term: Sell')

    st.write(f"**Long-Term Score:** {long_pct}% → {lt_rec}")
    st.write(f"**Short-Term Score:** {short_pct}% → {st_rec_val}")

    st.caption("Long-term signals use **3 years** of price data; short-term signals use **~3 months**. Fundamentals are based on the latest reported financials.")
