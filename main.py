import os
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def as_numeric(series):
    return pd.to_numeric(series, errors='coerce').fillna(0.0)


def find_column(df, candidates):
    df_cols_map = {col.lower().strip().replace('_', ' ').replace('/', ' '): col for col in df.columns}
    for cand in [c.lower().strip() for c in candidates]:
        if cand in df_cols_map:
            return df_cols_map[cand]
    for cand in [c.lower().strip() for c in candidates]:
        for col_lower, col_original in df_cols_map.items():
            if cand in col_lower:
                return col_original
    return None


def find_company_column(df):
    return find_column(df, ['company', 'company name', 'ticker', 'firm', 'organization'])


def smart_parse_period(df):
    period_col = find_column(df, ["period", "date", "year", "fiscal year"])
    if period_col:
        col_data = df[period_col]
        if pd.api.types.is_numeric_dtype(col_data) and (col_data > 1900).all() and (col_data < 2100).all():
            df['Period'] = col_data.astype(str)
        else:
            try:
                parsed = pd.to_datetime(col_data, errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    df['Period'] = parsed.dt.strftime('%Y')
                else:
                    df['Period'] = col_data.astype(str)
            except Exception:
                df['Period'] = col_data.astype(str)
    else:
        df['Period'] = [f"Period {i + 1}" for i in range(len(df))]
    return df


def safe_get_series(df, col_name):
    if col_name and col_name in df.columns:
        return as_numeric(df[col_name])
    return pd.Series(np.zeros(len(df)), index=df.index)


def analyze_financial_data(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    available = {'has_income_statement': False, 'has_balance_sheet': False, 'has_cash_flow': False,
                 'income_columns': {}, 'balance_columns': {}, 'cash_flow_columns': {}}
    income = {'revenue': ['revenue', 'sales', 'turnover'], 'gross_profit': ['gross profit'],
              'net_income': ['net income', 'profit']}
    balance = {'total_assets': ['total assets', 'assets total'],
               'total_liabilities': ['total liabilities', 'liabilities total'],
               'equity': ['shareholder equity', 'equity', 'total equity'], 'current_assets': ['current assets'],
               'current_liabilities': ['current liabilities']}
    cash_flow = {'operating_cash_flow': ['operating cash flow', 'cash flow from operating'],
                 'investing_cash_flow': ['investing cash flow', 'cash flow from investing'],
                 'financing_cash_flow': ['financing cash flow', 'cash flow from financial']}
    for key, cands in income.items():
        if col := find_column(df, cands): available['income_columns'][key] = col; available[
            'has_income_statement'] = True
    for key, cands in balance.items():
        if col := find_column(df, cands): available['balance_columns'][key] = col; available['has_balance_sheet'] = True
    for key, cands in cash_flow.items():
        if col := find_column(df, cands): available['cash_flow_columns'][key] = col; available['has_cash_flow'] = True
    return available


def process_financial_data(df_raw, data_info):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = smart_parse_period(df)
    df = df.sort_values('Period').reset_index(drop=True)
    result = {'Period': df['Period']}
    revenue = safe_get_series(df, data_info['income_columns'].get('revenue'))
    net_income = safe_get_series(df, data_info['income_columns'].get('net_income'))
    gross_profit = safe_get_series(df, data_info['income_columns'].get('gross_profit'))
    total_assets = safe_get_series(df, data_info['balance_columns'].get('total_assets'))
    total_liabilities = safe_get_series(df, data_info['balance_columns'].get('total_liabilities'))
    equity = safe_get_series(df, data_info['balance_columns'].get('equity'))
    current_assets = safe_get_series(df, data_info['balance_columns'].get('current_assets'))
    current_liabilities = safe_get_series(df, data_info['balance_columns'].get('current_liabilities'))
    result.update(
        {'Revenue': revenue, 'Net Income': net_income, 'Gross Profit': gross_profit, 'Total Assets': total_assets,
         'Total Liabilities': total_liabilities, 'Shareholders Equity': equity})
    result['Gross Margin %'] = np.divide(gross_profit, revenue, out=np.zeros_like(gross_profit),
                                         where=revenue != 0) * 100
    result['Revenue Growth %'] = revenue.pct_change().fillna(0) * 100
    result['Current Ratio'] = safe_get_series(df, find_column(df, ['current ratio']))
    result['Debt to Equity'] = safe_get_series(df, find_column(df, ['debt to equity', 'debt equity ratio']))
    result['ROA'] = safe_get_series(df, find_column(df, ['roa', 'return on assets']))
    result['ROE'] = safe_get_series(df, find_column(df, ['roe', 'return on equity']))
    if result['Current Ratio'].sum() == 0 and current_liabilities.sum() > 0:
        result['Current Ratio'] = np.divide(current_assets, current_liabilities, out=np.zeros_like(current_assets),
                                            where=current_liabilities != 0)
    if result['Debt to Equity'].sum() == 0 and equity.sum() > 0:
        result['Debt to Equity'] = np.divide(total_liabilities, equity, out=np.zeros_like(total_liabilities),
                                             where=equity != 0)
    if result['ROA'].sum() == 0 and total_assets.sum() > 0:
        result['ROA'] = np.divide(net_income, total_assets, out=np.zeros_like(net_income),
                                  where=total_assets != 0) * 100
    if result['ROE'].sum() == 0 and equity.sum() > 0:
        result['ROE'] = np.divide(net_income, equity, out=np.zeros_like(net_income), where=equity != 0) * 100
    if data_info['has_cash_flow']:
        result['Operating Cash Flow'] = safe_get_series(df, data_info['cash_flow_columns'].get('operating_cash_flow'))
        result['Investing Cash Flow'] = safe_get_series(df, data_info['cash_flow_columns'].get('investing_cash_flow'))
        result['Financing Cash Flow'] = safe_get_series(df, data_info['cash_flow_columns'].get('financing_cash_flow'))
    return pd.DataFrame(result)


def generate_insights(latest_row):
    insights = []
    if latest_row.empty: return insights

    if pd.notna(latest_row.get('Gross Margin %')):
        gm = latest_row['Gross Margin %']
        if gm > 40:
            insights.append(
                {'category': 'Profitability', 'title': 'Strong Gross Margin', 'value': f'{gm:.1f}%', 'icon': '✅',
                 'severity': 'success',
                 'message': 'Indicates excellent control over production/service costs. This strong efficiency provides a solid foundation for overall profitability.'})
        elif gm > 20:
            insights.append(
                {'category': 'Profitability', 'title': 'Moderate Gross Margin', 'value': f'{gm:.1f}%', 'icon': '📊',
                 'severity': 'info',
                 'message': 'A workable margin that covers costs, but there is potential to boost profitability by optimizing the supply chain or adjusting pricing strategies.'})
        else:
            insights.append(
                {'category': 'Profitability', 'title': 'Low Gross Margin', 'value': f'{gm:.1f}%', 'icon': '⚠️',
                 'severity': 'warning',
                 'message': 'Direct costs are consuming a large portion of revenue. This puts pressure on profitability and requires a review of pricing and supplier costs.'})

    # Growth
    if pd.notna(latest_row.get('Revenue Growth %')):
        rev_growth = latest_row['Revenue Growth %']
        icon, sev = ('📈', 'success') if rev_growth > 0 else ('📉', 'warning')
        insights.append({'category': 'Growth', 'title': 'Revenue Growth', 'value': f"{rev_growth:+.1f}%", 'icon': icon,
                         'severity': sev,
                         'message': f"Top-line revenue {'grew' if rev_growth > 0 else 'declined'} compared to the prior period. Consistent growth indicates strong market demand and competitive positioning."})

    # Liquidity
    if pd.notna(latest_row.get('Current Ratio')):
        cr = latest_row['Current Ratio']
        if cr > 1.5:
            insights.append({'category': 'Liquidity', 'title': 'Strong Liquidity', 'value': f'{cr:.2f}', 'icon': '✅',
                             'severity': 'success',
                             'message': 'A comfortable asset buffer to cover short-term liabilities. This provides a strong safety net for handling unexpected expenses or economic downturns.'})
        elif cr >= 1.0:
            insights.append({'category': 'Liquidity', 'title': 'Adequate Liquidity', 'value': f'{cr:.2f}', 'icon': '📊',
                              'severity': 'info',
                              'message': 'Short-term obligations are currently covered, but the cushion is modest. Monitoring cash flow and working capital is advisable.'})
        else:
            insights.append({'category': 'Liquidity', 'title': 'Liquidity Risk', 'value': f'{cr:.2f}', 'icon': '⚠️',
                             'severity': 'warning',
                             'message': 'Current liabilities exceed current assets. This could pose a challenge in paying off short-term debts, potentially requiring new financing.'})

    if pd.notna(latest_row.get('Debt to Equity')):
        dte = latest_row['Debt to Equity']
        if dte < 1.0:
            insights.append(
                {'category': 'Leverage', 'title': 'Conservative Structure', 'value': f'{dte:.2f}', 'icon': '✅',
                 'severity': 'success',
                 'message': 'The company is primarily funded by equity, indicating low financial risk from borrowing. This provides stability and a strong balance sheet.'})
        elif dte < 2.0:
            insights.append({'category': 'Leverage', 'title': 'Moderate Leverage', 'value': f'{dte:.2f}', 'icon': '📊',
                             'severity': 'info',
                             'message': 'A balanced use of debt and equity to finance assets. It is important to ensure cash flow can comfortably service the debt obligations.'})
        else:
            insights.append({'category': 'Leverage', 'title': 'High Leverage', 'value': f'{dte:.2f}', 'icon': '⚠️',
                             'severity': 'warning',
                             'message': 'A high reliance on debt increases financial risk, making the company more vulnerable to interest rate changes and economic downturns.'})

    return insights


def stream_groq_commentary(prompt: str, container):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: container.error("❌ GROQ_API_KEY not found in .env file."); return
    try:
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}],
                                                temperature=0.7, max_tokens=1024, stream=True)
        full_response = ""
        placeholder = container.empty()
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content: full_response += content; placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    except Exception as e:
        container.error(f"❌ Groq API error: {e}")

st.set_page_config(page_title="Financial Analyzer", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stMetric {border-radius: 10px; padding: 15px; background-color: #262730;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 40px; background-color: transparent; padding: 0;}
    .stTabs [data-baseweb="tab"]:hover {background-color: #31333F;}
    .stTabs [aria-selected="true"] {background-color: #262730; border-radius: 5px;}
    .insight-card {
        background: #262730; 
        border-radius: 10px; 
        padding: 20px; 
        margin-bottom: 25px; /* Increased margin for spacing */
        border-left: 5px solid #4A4A4A;
    }
    .insight-success {border-left-color: #28a745;}
    .insight-warning {border-left-color: #ffc107;}
    .insight-info {border-left-color: #3498db;}
    .insight-title {font-size: 1.1rem; font-weight: 600; margin-bottom: 5px;}
    .insight-value {font-size: 1.5rem; font-weight: bold;} 
    .insight-message {font-size: 0.9rem; color: #b0b3b8;}
    .category-badge {font-size: 0.8rem; color: #b0b3b8; margin-bottom: 10px; display: block;}
</style>
""", unsafe_allow_html=True)
st.title("📊 Intelligent Financial Analyzer")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Financial Data CSV", type=["csv"])
    company_selector_placeholder = st.empty()
    enable_llm = st.checkbox("🤖 Enable AI Commentary", value=True)
    if enable_llm and not os.getenv("GROQ_API_KEY"): st.warning("Set GROQ_API_KEY in your .env file.")
    st.header("🎯 Demo Data")
    if st.button("Use Demo Data", use_container_width=True):
        st.session_state['df_raw'] = pd.read_csv(io.StringIO("""
Year,Company,Revenue,Gross Profit,Net Income,Total Assets,Total Liabilities,Shareholder Equity,Current Ratio,Debt to Equity Ratio,ROE,ROA
2022,MSFT,198270,135620,72738,364840,198298,166542,1.77,0.23,43.7,19.9
2021,MSFT,168088,115856,61271,333779,191791,141988,2.04,0.37,43.2,18.4
2022,GOOG,282836,156385,59972,359228,107613,251615,2.38,0.06,23.4,16.5
2021,GOOG,257637,146707,76033,351984,105739,246245,2.52,0.07,31.7,21.0
2022,AAPL,394328,170782,99803,352755,302083,50672,0.88,2.37,197.0,28.3
2021,AAPL,365817,152836,94680,351002,287912,63090,1.07,1.98,150.1,27.0
2022,AMZN,513983,225011,-2722,462675,340534,122141,0.94,1.97,-1.9,-0.6
2021,AMZN,469822,207623,33364,420549,282365,138184,1.14,1.38,29.4,8.1
        """))
        st.session_state['use_demo'] = True
        st.rerun()

df_raw = st.session_state.get('df_raw')
if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}"); st.stop()
if df_raw is None: st.info("👆 Upload a CSV or use Demo Data to start."); st.stop()

df_analysis = df_raw.copy()
company_col = find_company_column(df_raw)
selected_company = None
if company_col and df_raw[company_col].nunique() > 1:
    companies = sorted(df_raw[company_col].unique())
    default_index = companies.index('GOOG') if 'GOOG' in companies else 0
    selected_company = company_selector_placeholder.selectbox("Select Company", options=companies, index=default_index)
    df_analysis = df_raw[df_raw[company_col] == selected_company].copy()

data_info = analyze_financial_data(df_analysis)
df = process_financial_data(df_analysis, data_info)
latest = df.iloc[-1] if not df.empty else pd.Series()

st.header("📈 Financial Dashboard")

if selected_company:
    st.subheader(f"{selected_company}")

kpis = ['Revenue', 'Net Income', 'Gross Margin %', 'Current Ratio', 'ROE']
kpi_formats = {'Revenue': '${:,.0f}', 'Net Income': '${:,.0f}', 'Gross Margin %': '{:.1f}%', 'Current Ratio': '{:.2f}',
               'ROE': '{:.1f}%'}
kpi_cols = st.columns(len(kpis))
for i, key in enumerate(kpis):
    if key in latest and pd.notna(latest[key]):
        delta_key = 'Revenue Growth %' if key == 'Revenue' else None
        delta_val = latest.get(delta_key)
        delta_str = f"{delta_val:+.1f}%" if delta_key and pd.notna(delta_val) else None
        kpi_cols[i].metric(key.replace('%', '').strip(), kpi_formats[key].format(latest[key]), delta=delta_str)

st.markdown("---")

tabs = st.tabs(["Profitability", "Balance Sheet", "Cash Flow", "Growth & Efficiency"])
with tabs[0]:
    st.subheader("💰 Revenue & Profitability")
    if 'Revenue' in df.columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df['Period'], y=df['Revenue'], name='Revenue'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Period'], y=df['Net Income'], name='Net Income', mode='lines+markers'),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Period'], y=df['Gross Margin %'], name='Gross Margin %', mode='lines+markers',
                                 line=dict(dash='dot')), secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
with tabs[1]:
    st.subheader("🏦 Structure: Assets vs. Liabilities")
    if 'Total Assets' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Period'], y=df['Total Assets'], name='Total Assets'))
        fig.add_trace(go.Scatter(x=df['Period'], y=df['Total Liabilities'], name='Total Liabilities'))
        fig.add_trace(go.Scatter(x=df['Period'], y=df['Shareholders Equity'], name='Equity', mode='lines+markers'))
        st.plotly_chart(fig, use_container_width=True)
with tabs[2]:
    st.subheader("💧 Cash Flow")
    if data_info['has_cash_flow']:
        fig = go.Figure(data=[go.Bar(x=df['Period'], y=df.get('Operating Cash Flow'), name='Operating'),
                              go.Bar(x=df['Period'], y=df.get('Investing Cash Flow'), name='Investing'),
                              go.Bar(x=df['Period'], y=df.get('Financing Cash Flow'), name='Financing')])
        fig.update_layout(barmode='relative', title_text="Cash Flow from Activities")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cash Flow data not found in this dataset.")
with tabs[3]:
    st.subheader("🚀 Growth & Efficiency")
    col1, col2 = st.columns(2)
    if 'Revenue Growth %' in df.columns and len(df) > 1:
        fig = go.Figure(go.Bar(x=df['Period'][1:], y=df['Revenue Growth %'][1:]))
        fig.update_layout(yaxis_ticksuffix='%', title_text="YoY Revenue Growth")
        col1.plotly_chart(fig, use_container_width=True)
    if 'ROE' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Period'], y=df['ROE'], name='ROE'))
        fig.add_trace(go.Scatter(x=df['Period'], y=df['ROA'], name='ROA'))
        fig.update_layout(yaxis_ticksuffix='%', title_text="Efficiency (ROE & ROA)")
        col2.plotly_chart(fig, use_container_width=True)

st.markdown("---")

insight_col, ai_col = st.columns(2)
with insight_col:
    st.header("🎯 Financial Insights")
    insights = generate_insights(latest)
    if not insights:
        st.info("No specific insights to highlight.")
    else:
        for insight in insights:
            st.markdown(f"""<div class="insight-card insight-{insight['severity']}">
                <div class="category-badge">{insight['category']}</div>
                <div class="insight-title">{insight['icon']} {insight['title']}</div>
                <div class="insight-value">{insight['value']}</div>
                <div class="insight-message">{insight['message']}</div>
            </div>""", unsafe_allow_html=True)

with ai_col:
    if enable_llm:
        st.header("🤖 AI-Powered Analysis")
        if os.getenv("GROQ_API_KEY"):
            if st.button("Generate AI Commentary", use_container_width=True, type="primary"):
                prompt = f"You are an expert financial analyst. Analyze this data for {selected_company or 'the company'}. Latest Period: {latest.to_json(indent=2)}. Trend: {df.tail(3).to_string(index=False)}. Provide a concise report with: 1. Executive Summary 2. Key Findings (bullet points) 3. Strategic Recommendations (bullet points)."
                with st.spinner("🧠 AI is analyzing..."):
                    stream_groq_commentary(prompt, st.container())
        else:
            st.warning("Please set your GROQ_API_KEY in the .env file to enable AI analysis.")
