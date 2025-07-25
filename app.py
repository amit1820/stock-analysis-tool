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

    # Metric explanations section
    st.markdown("---")
    st.header("📚 Metric Explanations")
    st.markdown("""
- **P/E Ratio (Price-to-Earnings Ratio):** Compares a company's share price to its earnings per share. Lower values (<25) may indicate undervaluation or better value relative to earnings.
- **EPS (Earnings Per Share):** The portion of a company's profit allocated to each outstanding share. Positive EPS signals profitability.
- **ROE (Return on Equity):** Measures how effectively management is using equity financing to generate profits. ROE >15% indicates strong efficiency.
- **Revenue Growth (YoY):** Year-over-year percentage increase in sales revenue. Growth >10% suggests healthy expansion.
- **Debt-to-Equity Ratio:** Indicates the relative proportion of shareholders' equity and debt used to finance a company's assets. A ratio <1 shows conservative debt levels.
- **MA Cross (Golden Cross):** When the 50-day moving average crosses above the 200-day moving average, it signals a potential long-term bullish trend.
- **RSI (Relative Strength Index):** A momentum oscillator that measures speed and change of price movements. Values between 30 and 70 denote neutral momentum; above 70 is overbought, below 30 is oversold.
- **MACD (Moving Average Convergence Divergence):** Shows the relationship between two moving averages of a security’s price. A bullish signal occurs when the MACD line crosses above the signal line.
""")

    # Initialize scores and record of contributions
    long_term_score = 0
    short_term_score = 0
    contributions = []

    max_long = sum([metrics_weights[m] for m in ["P/E Ratio","EPS","ROE","Revenue Growth","Debt-to-Equity","MA Cross"]])
    max_short = metrics_weights['RSI'] + metrics_weights['MACD']

    # Fundamental & Long-Term Analysis
    st.markdown("---")
    st.header("🧮 Fundamental & Long-Term Analysis")

    # P/E Ratio
    pe = info.get('trailingPE')
    pe_pass = False
    if pe:
        st.write(f"**P/E Ratio:** {pe:.2f}")
        pe_pass = pe < 25
        if pe_pass:
            long_term_score += metrics_weights['P/E Ratio']
        contributions.append(("P/E Ratio", round(pe,2), "<25", metrics_weights['P/E Ratio'] if pe_pass else 0))
    else:
        st.write("**P/E Ratio:** Data not available")
        contributions.append(("P/E Ratio", "N/A", "<25", 0))

    # EPS
    eps = info.get('trailingEps')
    eps_pass = False
    if eps:
        st.write(f"**EPS:** {eps:.2f}")
        eps_pass = eps > 0
        if eps_pass:
            long_term_score += metrics_weights['EPS']
        contributions.append(("EPS", round(eps,2), ">0", metrics_weights['EPS'] if eps_pass else 0))
    else:
        st.write("**EPS:** Data not available")
        contributions.append(("EPS", "N/A", ">0", 0))

    # ROE
    roe = info.get('returnOnEquity')
    roe_pass = False
    if roe:
        roe_pct = roe * 100
        st.write(f"**ROE:** {roe_pct:.2f}%")
        roe_pass = roe > 0.15
        if roe_pass:
            long_term_score += metrics_weights['ROE']
        contributions.append(("ROE (%)", round(roe_pct,2), ">15%", metrics_weights['ROE'] if roe_pass else 0))
    else:
        st.write("**ROE:** Data not available")
        contributions.append(("ROE (%)", "N/A", ">15%", 0))

    # Revenue Growth
    rev_growth = info.get('revenueGrowth')
    rev_pass = False
    if rev_growth:
        rev_pct = rev_growth * 100
        st.write(f"**Revenue Growth (YoY):** {rev_pct:.2f}%")
        rev_pass = rev_growth > 0.10
        if rev_pass:
            long_term_score += metrics_weights['Revenue Growth']
        contributions.append(("Revenue Growth (%)", round(rev_pct,2), ">10%", metrics_weights['Revenue Growth'] if rev_pass else 0))
    else:
        st.write("**Revenue Growth (YoY):** Data not available")
        contributions.append(("Revenue Growth (%)", "N/A", ">10%", 0))

    # Debt-to-Equity
    de = info.get('debtToEquity')
    de_pass = False
    if de is not None:
        st.write(f"**Debt-to-Equity Ratio:** {de:.2f}")
        de_pass = de < 1
        if de_pass:
            long_term_score += metrics_weights['Debt-to-Equity']
        contributions.append(("Debt-to-Equity", round(de,2), "<1", metrics_weights['Debt-to-Equity'] if de_pass else 0))
    else:
        st.write("**Debt-to-Equity:** Data not available")
        contributions.append(("Debt-to-Equity", "N/A", "<1", 0))

    # Golden Cross
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    ma_cross = hist['MA50'].iloc[-1] > hist['MA200'].iloc[-1]
    st.write(f"**Golden Cross (MA50 > MA200):** {'Yes' if ma_cross else 'No'}")
    ma_pass = ma_cross
    if ma_pass:
        long_term_score += metrics_weights['MA Cross']
    contributions.append(("Golden Cross", 'Yes' if ma_cross else 'No', "Yes", metrics_weights['MA Cross'] if ma_pass else 0))

    # Short-Term Technical Analysis
    st.markdown("---")
    st.header("📈 Short-Term Technical Analysis")

    # RSI
    hist['RSI'] = calculate_rsi(hist)
    rsi = hist['RSI'].dropna()
    rsi_val = rsi.iloc[-1] if not rsi.empty else None
    st.write(f"**RSI (Last 6mo):** {round(rsi_val,2) if rsi_val else 'N/A'}")
    rsi_pass = rsi_val and 30 < rsi_val < 70
    if rsi_pass:
        short_term_score += metrics_weights['RSI']
    contributions.append(("RSI", round(rsi_val,2) if rsi_val else 'N/A', "30-70", metrics_weights['RSI'] if rsi_pass else 0))

    # MACD
    macd_line = hist['Close'].ewm(span=12, adjust=False).mean() - hist['Close'].ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_pass = macd_line.iloc[-1] > signal_line.iloc[-1]
    st.write(f"**MACD vs Signal:** {'Bullish' if macd_pass else 'Bearish'}")
    if macd_pass:
        short_term_score += metrics_weights['MACD']
    contributions.append(("MACD", 'Bullish' if macd_pass else 'Bearish', "Bullish", metrics_weights['MACD'] if macd_pass else 0))

    # Calculation Details Table
    st.markdown("---")
    st.header("🔢 Calculation Details")
    df_calc = pd.DataFrame(contributions, columns=['Metric','Value','Threshold','Contribution'])
    st.table(df_calc)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RSI Chart")
        fig1, ax1 = plt.subplots(figsize=(4,3))
        rsi.plot(ax=ax1)
        ax1.axhline(70, linestyle='--')
        ax1.axhline(30, linestyle='--')
        st.pyplot(fig1)
    with col2:
        st.subheader("MACD Chart")
        fig2, ax2 = plt.subplots(figsize=(4,3))
        macd_line.plot(ax=ax2, label='MACD')
        signal_line.plot(ax=ax2, label='Signal')
        ax2.legend()
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Price Trend")
        fig3, ax3 = plt.subplots(figsize=(4,3))
        hist['Close'].plot(ax=ax3)
        st.pyplot(fig3)
    with col4:
        st.subheader("Moving Averages")
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

    lt_rec = "✅ Long-Term: Strong Buy" if long_pct >= 75 else ("➖ Long-Term: Hold" if long_pct >= 50 else "❌ Long-Term: Sell")
    st_rec = "✅ Short-Term: Strong Buy" if short_pct >= 75 else ("➖ Short-Term: Hold" if short_pct >= 50 else "❌ Short-Term: Sell")

    st.write(f"**Long-Term Score:** {long_pct}% | Recommendation: {lt_rec}")
    st.write(f"**Short-Term Score:** {short_pct}% | Recommendation: {st_rec}")
