# 커스텀 가능한 한국 주식 수익률 분포 히스토그램 함수
# FinanceDataReader + Pandas + Plotly 활용

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# --- 회사명-티커 매핑 테이블 생성 (국내+해외) ---
@st.cache_data
def get_all_stock_table():
    # KRX
    krx = fdr.StockListing('KRX')
    krx = krx.rename(columns={'Code': 'Code', 'Name': 'Name'})
    krx = krx[['Code', 'Name']].drop_duplicates()
    krx = krx[krx['Code'].str.len() == 6]
    krx['Market'] = 'KRX'

    # 해외 거래소
    all_dfs = [krx]
    for market in ['NASDAQ', 'NYSE', 'AMEX']:
        try:
            df = fdr.StockListing(market)
            # 컬럼명 표준화
            if 'Symbol' in df.columns:
                df = df.rename(columns={'Symbol': 'Code'})
            if 'Name' not in df.columns and 'name' in df.columns:
                df = df.rename(columns={'name': 'Name'})
            if 'Code' in df.columns and 'Name' in df.columns:
                df = df[['Code', 'Name']].drop_duplicates()
                df['Market'] = market
                all_dfs.append(df)
        except Exception as e:
            pass  # 해당 거래소 불러오기 실패시 무시

    all_df = pd.concat(all_dfs, ignore_index=True)
    return all_df

all_stock_table = get_all_stock_table()

# 회사명(티커) 리스트 생성 (시장명도 함께)
company_options = [f"{row.Name} ({row.Code}) [{row.Market}]" for row in all_stock_table.itertuples()]
code_to_name = dict(zip(all_stock_table['Code'], all_stock_table['Name']))
name_to_code = dict(zip(all_stock_table['Name'], all_stock_table['Code']))
code_to_market = dict(zip(all_stock_table['Code'], all_stock_table['Market']))

def get_korean_stock_data(ticker, start_year=1981, end_year=None):
    if end_year is None:
        end_year = datetime.today().year
    start_date = f'{start_year}-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = fdr.DataReader(ticker, start_date, end_date)
    if data.empty:
        return None, None
    yearly_data = data.groupby(data.index.year)['Close'].last()
    returns = yearly_data.pct_change().dropna() * 100
    return yearly_data, returns

def calculate_cagr(prices):
    if len(prices) < 2:
        return np.nan
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    n = len(prices) - 1
    return (end_price / start_price) ** (1 / n) - 1

def plot_return_histogram(returns, period_label, ticker_name, bins, bin_labels, colors):
    total_count = len(returns)
    positive_count = (returns > 0).sum()
    negative_count = (returns <= 0).sum()
    avg_return = returns.mean()
    cagr = calculate_cagr(returns.add(100).div(100).cumprod())
    positive_pct = (positive_count / total_count) * 100
    negative_pct = (negative_count / total_count) * 100
    max_return = returns.max()
    min_return = returns.min()
    max_year = returns.idxmax() if not returns.empty else '-'
    min_year = returns.idxmin() if not returns.empty else '-'

    hist_data = pd.cut(returns, bins=bins, labels=bin_labels, right=False)
    hist_counts = hist_data.value_counts().sort_index()
    hist_percentages = (hist_counts / total_count) * 100

    # 제목 아래에 표시할 통계 텍스트
    subtitle = f"연 평균 정상률 (CAGR): {cagr*100:.2f}%  |  이익 확률: {positive_pct:.1f}%  |  손실 확률: {negative_pct:.1f}%  |  최고: {max_return:.2f}%({max_year})  |  최저: {min_return:.2f}%({min_year})"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_labels,
        y=hist_percentages,
        marker_color=colors,
        text=[f'{pct:.1f}%' if pct > 0 else '' for pct in hist_percentages],
        textposition='outside',
        showlegend=False,
        customdata=[hist_counts.get(label, 0) for label in bin_labels],
        hovertemplate='구간: %{x}<br>비율: %{y:.1f}%<br>횟수: %{customdata}회<extra></extra>',
    ))

    # annotation에서 <br/> 제거 및 확률만 남김
    # 손실 확률
    fig.add_annotation(
        x=1.5, y=max(hist_percentages) * 0.85,
        text=f"손실 확률: {negative_pct:.1f}%",
        showarrow=False,
        font=dict(size=14, color='black'),
        align='center'
    )
    # 이익 확률
    fig.add_annotation(
        x=len(bin_labels) * 0.7, y=max(hist_percentages) * 0.85,
        text=f"이익 확률: {positive_pct:.1f}%",
        showarrow=False,
        font=dict(size=14, color='black'),
        align='center'
    )
    # 제목 아래에 subtitle 표시 (annotation으로)
    fig.add_annotation(
        text=subtitle,
        xref='paper', yref='paper',
        x=0.5, y=1.08, showarrow=False,
        font=dict(size=15, color='black'),
        align='center'
    )

    # 손실/이익 경계선: 0%가 포함된 bin의 왼쪽 경계에 vline
    zero_bin_idx = None
    for i in range(len(bins)-1):
        if bins[i] <= 0 < bins[i+1]:
            zero_bin_idx = i
            break
    if zero_bin_idx is not None:
        fig.add_vline(
            x=zero_bin_idx - 0.5,  # 해당 bin의 왼쪽 경계
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text="손실/이익 경계"
        )

    fig.update_layout(
        title={
            'text': f'{ticker_name} 연 수익률 분포',
            'x': 0.5,
            'font': {'size': 18, 'color': 'black'}
        },
        xaxis_title='연 수익률 구간(%)',
        yaxis_title='발생 빈도(%)',
        template='plotly_white',
        width=1000,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=100, b=100)
    )
    fig.update_xaxes(
        tickangle=0,
        tickfont=dict(size=11),
        showgrid=False,
        showline=True,
        linecolor='black'
    )
    fig.update_yaxes(
        tickfont=dict(size=11),
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        showline=True,
        linecolor='black',
        ticksuffix='%'
    )
    return fig

def get_ticker_and_name(user_input):
    # 입력값이 (시장)까지 포함된 경우
    if '[' in user_input and ']' in user_input:
        # 예: Apple Inc. (AAPL) [NASDAQ]
        code = user_input.split('(')[-1].split(')')[0].strip()
        name = user_input.split('(')[0].strip()
        return code, name
    # 입력값이 6자리 숫자면 티커로 간주
    if user_input in code_to_name:
        return user_input, code_to_name[user_input]
    # 입력값이 회사명이면
    elif user_input in name_to_code:
        return name_to_code[user_input], user_input
    # 회사명(티커) 형태면
    elif '(' in user_input and ')' in user_input:
        name = user_input.split('(')[0].strip()
        code = user_input.split('(')[-1].replace(')','').strip()
        return code, name
    else:
        return user_input, user_input  # fallback

# Streamlit UI
st.title("한국/해외 주식 연 수익률 분포 히스토그램")

st.markdown("""
- **KOSPI 1981~오늘까지 연 수익률 분포**  
- 10% 단위 구간, 손실=회색, 이익=파란색  
- 각 막대 위에 비율(%) 표시, 이익/손실확률, CAGR 표시  
- 아래에서 회사명/티커로 검색해 국내외 주식 동일 분석 가능
""")

# 기본 KOSPI
with st.expander("KOSPI 연 수익률 분포 (1981~오늘)", expanded=True):
    bins = [-100, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_labels = ['~-30', '-30~-20', '-20~-10', '-10~0', '0~10', '10~20', 
                  '20~30', '30~40', '40~50', '50~60', '60~70', '70~80', '80~90', '90~']
    colors = ['#808080' if i < 4 else '#4472C4' for i in range(len(bin_labels))]

    yearly_data, returns = get_korean_stock_data('KS11', 1981)
    if returns is not None:
        fig = plot_return_histogram(returns, '연간', 'KOSPI', bins, bin_labels, colors)
        st.plotly_chart(fig, use_container_width=True)
        # 코스피 연도별 종가 막대그래프
        price_df = yearly_data.reset_index()
        price_df.columns = ['연도', '종가']
        fig_price = go.Figure(go.Bar(x=price_df['연도'], y=price_df['종가'], marker_color='#4472C4'))
        fig_price.update_layout(title="KOSPI 연도별 종가(지수) 막대그래프", xaxis_title='연도', yaxis_title='종가(지수)')
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("KOSPI 데이터를 불러올 수 없습니다.")

# 사용자 입력
st.header("다른 종목/지수 연 수익률 분포 보기")

selected = st.selectbox("회사명 또는 티커를 선택/입력하세요", company_options, index=company_options.index("삼성전자 (005930) [KRX]") if "삼성전자 (005930) [KRX]" in company_options else 0)
user_input = st.text_input("직접 입력(회사명, 티커, 회사명(티커) 모두 가능)", value="삼성전자")

# 유사 검색 결과 보여주기
similar_options = []
do_analysis = False
if user_input.strip() and len(user_input.strip()) >= 2:
    keyword = user_input.strip().lower()
    similar_options = [opt for opt in company_options if keyword in opt.lower()]
    if similar_options:
        st.markdown(f"**유사 검색 결과:**")
        similar_selected = st.selectbox("아래에서 선택하면 바로 분석됩니다", similar_options, key="similar_select")
        # selectbox에서 선택하면 즉시 분석
        ticker, company_name = get_ticker_and_name(similar_selected)
        do_analysis = True
    else:
        do_analysis = False
else:
    do_analysis = False

# selectbox에서 선택하면 자동 분석
if not do_analysis and st.session_state.get('last_selected') != selected:
    ticker, company_name = get_ticker_and_name(selected)
    st.session_state['last_selected'] = selected
    do_analysis = True

# selectbox 우선, 직접입력값이 있으면 덮어씀 (유사 검색에서 선택하지 않은 경우)
if not do_analysis:
    if user_input.strip():
        ticker, company_name = get_ticker_and_name(user_input.strip())
    else:
        ticker, company_name = get_ticker_and_name(selected)

start_year = st.number_input("시작 연도", min_value=1981, max_value=datetime.today().year-1, value=2000)
end_year = st.number_input("종료 연도", min_value=start_year+1, max_value=datetime.today().year, value=datetime.today().year)

if (not similar_options and do_analysis) or (similar_options and do_analysis) or (not similar_options and st.button("분석하기")):
    yearly_data, returns = get_korean_stock_data(ticker, int(start_year), int(end_year))
    if returns is not None:
        fig = plot_return_histogram(returns, '연간', company_name, bins, bin_labels, colors)
        st.plotly_chart(fig, use_container_width=True)
        
        # 상승/하락 연도 계산 및 표기
        up_years = returns[returns > 0].index.tolist()
        down_years = returns[returns <= 0].index.tolist()
        st.markdown(f"**상승 연도 수:** {len(up_years)}  |  **하락 연도 수:** {len(down_years)}")
        st.markdown(f"**상승 연도:** {', '.join(map(str, up_years)) if up_years else '-'}")
        st.markdown(f"**하락 연도:** {', '.join(map(str, down_years)) if down_years else '-'}")

        # 최고/최저 수익률 및 연도
        max_return = returns.max()
        min_return = returns.min()
        max_year = returns.idxmax() if not returns.empty else '-'
        min_year = returns.idxmin() if not returns.empty else '-'
        st.markdown(f"**최고 수익률:** {max_return:.2f}% ({max_year})  |  **최저 수익률:** {min_return:.2f}% ({min_year})")

        # 실제 연도별 종가(지수/주가) 막대그래프 추가
        price_df = yearly_data.reset_index()
        price_df.columns = ['연도', '종가']
        fig_price = go.Figure(go.Bar(x=price_df['연도'], y=price_df['종가'], marker_color='#4472C4'))
        fig_price.update_layout(title=f"{company_name} 연도별 종가(지수/주가) 막대그래프", xaxis_title='연도', yaxis_title='종가(지수/주가)')
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("해당 티커의 데이터를 불러올 수 없습니다.")