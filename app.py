import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


st.set_page_config(
    page_title="FractalIntelligence | Institutional Grade Analytics",
    page_icon="💎",
    layout="wide"
)


st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child { background-color: #00ffcc; color: black; border: none; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    .company-box { background-color: #1e2130; padding: 20px; border-radius: 10px; border-left: 5px solid #00ffcc; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)



@st.cache_data
def get_hurst_exponent(time_series, max_lag=20):
    """Calculates the Hurst Exponent (Market Memory)."""
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(show_spinner=False)
def get_asset_details(symbol):
    """Fetches company metadata."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "summary": info.get('longBusinessSummary', 'No description available.'),
            "website": info.get('website', '#')
        }
    except:
        return None

def load_data(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Adj Close'] = df['Close']
    return df


st.title("💎 FractalIntelligence AI")
st.caption("Advanced Market Regime Detection & Predictive Modeling for Institutional Analysis")
st.markdown("---")


col_search, col_date = st.columns([2, 1])
with col_search:
    user_ticker = st.text_input("🔍 ENTER ASSET SYMBOL (e.g., RELIANCE.NS, TSLA, BTC-USD)", "AAPL").upper()
with col_date:
    lookback_years = st.slider("Lookback Period (Years)", 1, 10, 5)

start_date = datetime.now() - timedelta(days=365 * lookback_years)



if user_ticker:
    asset_info = get_asset_details(user_ticker)
    df = load_data(user_ticker, start_date)

    if df is not None and asset_info:
        
        with st.container():
            st.markdown(f"""
            <div class="company-box">
                <h2 style='margin:0;'>{asset_info['name']}</h2>
                <p style='color:#00ffcc;'><b>{asset_info['sector']} | {asset_info['industry']}</b></p>
                <p style='font-size: 0.9rem;'>{asset_info['summary'][:400]}...</p>
            </div>
            """, unsafe_allow_html=True)

      
        df['Returns'] = df['Close'].pct_change()
        df['Hurst'] = df['Close'].rolling(window=100).apply(lambda x: get_hurst_exponent(x.values))
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

      
        features = ['Hurst', 'Volatility', 'Returns']
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
     
        current_prediction = model.predict(X.iloc[[-1]])[0]
        acc = accuracy_score(y_test, model.predict(X_test))

    
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Price", f"${df['Close'].iloc[-1]:.2f}", f"{df['Returns'].iloc[-1]*100:.2f}%")
        
        h_val = df['Hurst'].iloc[-1]
        regime = "TRENDING" if h_val > 0.52 else "MEAN-REVERTING" if h_val < 0.48 else "RANDOM"
        m2.metric("Market DNA (Hurst)", f"{h_val:.3f}", regime)
        
        m3.metric("Model Confidence", f"{acc*100:.1f}%")
        
        pred_text = "BULLISH" if current_prediction == 1 else "BEARISH"
        m4.metric("AI Signal", pred_text, delta="Next 24H")

        tabs = st.tabs(["📊 Technical Chart", "🧠 AI Insights", "📜 Backtest Data"])
        
        with tabs[0]:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
            fig.update_layout(template="plotly_dark", height=600, title=f"{user_ticker} Price History")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Asset Memory Analysis")
                fig_h = px.line(df, y='Hurst', title="Hurst Exponent Over Time")
                fig_h.add_hline(y=0.5, line_dash="dash", line_color="red")
                st.plotly_chart(fig_h, use_container_width=True)
            with c2:
                st.subheader("Decision Drivers")
                imp = pd.DataFrame({'Feature': features, 'Weight': model.feature_importances_})
                st.plotly_chart(px.bar(imp, x='Weight', y='Feature', orientation='h'), use_container_width=True)

        with tabs[2]:
            st.dataframe(df.tail(20), use_container_width=True)
    else:
        st.error("Invalid Ticker or Data Unavailable. Please verify the symbol (e.g., MSFT, RELIANCE.NS).")