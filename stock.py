import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="📊 주식 투자 분석 도구 V2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .comparison-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<div class="main-header">📊 주식 투자 분석 도구 V2</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">✨ 시장지수 비교 기능 추가</p>', unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.title("📋 분석 설정")

# 국내/해외 선택
market_type = st.sidebar.selectbox(
    "🌍 시장 선택",
    ["국내 (한국)", "해외 (미국)"]
)

# 종목 선택/입력
if market_type == "국내 (한국)":
    st.sidebar.subheader("🇰🇷 국내 주식")
    
    # 인기 종목 리스트
    popular_stocks_kr = {
        "삼성전자": "005930",
        "SK하이닉스": "000660", 
        "NAVER": "035420",
        "카카오": "035720",
        "LG화학": "051910",
        "현대차": "005380",
        "KB금융": "105560",
        "셀트리온": "068270",
        "POSCO홀딩스": "005490",
        "LG전자": "066570",
        "삼성바이오로직스": "207940",
        "기아": "000270",
        "SK텔레콤": "017670",
        "LG생활건강": "051900",
        "삼성SDI": "006400"
    }
    
    stock_input_type = st.sidebar.radio(
        "종목 선택 방법",
        ["인기 종목에서 선택", "직접 입력"]
    )
    
    if stock_input_type == "인기 종목에서 선택":
        selected_stock_name = st.sidebar.selectbox(
            "종목 선택",
            list(popular_stocks_kr.keys())
        )
        stock_code = popular_stocks_kr[selected_stock_name]
        stock_symbol = stock_code
    else:
        stock_code = st.sidebar.text_input(
            "종목 코드 입력 (예: 005930)",
            placeholder="6자리 숫자"
        )
        stock_symbol = stock_code
        selected_stock_name = stock_code if stock_code else ""
        
else:  # 해외 (미국)
    st.sidebar.subheader("🇺🇸 해외 주식")
    
    # 인기 종목 리스트
    popular_stocks_us = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Meta": "META",
        "Netflix": "NFLX",
        "AMD": "AMD",
        "Intel": "INTC",
        "Berkshire Hathaway": "BRK-B",
        "Johnson & Johnson": "JNJ",
        "JPMorgan Chase": "JPM",
        "Visa": "V",
        "Procter & Gamble": "PG"
    }
    
    stock_input_type = st.sidebar.radio(
        "종목 선택 방법",
        ["인기 종목에서 선택", "직접 입력"]
    )
    
    if stock_input_type == "인기 종목에서 선택":
        selected_stock_name = st.sidebar.selectbox(
            "종목 선택",
            list(popular_stocks_us.keys())
        )
        stock_symbol = popular_stocks_us[selected_stock_name]
    else:
        stock_symbol = st.sidebar.text_input(
            "종목 심볼 입력 (예: AAPL)",
            placeholder="영문 심볼"
        ).upper()
        selected_stock_name = stock_symbol

# 기간 선택
period_options = {
    "1개월": 30,
    "3개월": 90,
    "6개월": 180,
    "1년": 365,
    "2년": 730,
    "3년": 1095
}

selected_period = st.sidebar.selectbox(
    "📅 분석 기간",
    list(period_options.keys()),
    index=3  # 기본값: 1년
)

# 시장지수 비교 옵션 추가
st.sidebar.markdown("---")
st.sidebar.subheader("📊 시장지수 비교")
compare_with_index = st.sidebar.checkbox("시장지수와 비교", value=True)

if compare_with_index:
    if market_type == "국내 (한국)":
        index_options = {
            "KOSPI": "KS11",
            "KOSDAQ": "KQ11",
            "KRX 100": "KRX100"
        }
        selected_index_name = st.sidebar.selectbox(
            "비교할 지수 선택",
            list(index_options.keys()),
            index=0
        )
        index_symbol = index_options[selected_index_name]
    else:
        index_options = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI"
        }
        selected_index_name = st.sidebar.selectbox(
            "비교할 지수 선택",
            list(index_options.keys()),
            index=0
        )
        index_symbol = index_options[selected_index_name]

# 분석 시작 버튼
analyze_button = st.sidebar.button("🔍 분석 시작", type="primary")

# 투자 지표 계산 함수들
def calculate_returns(prices):
    """수익률 계산"""
    return prices.pct_change().dropna()

def calculate_cumulative_returns(returns):
    """누적 수익률 계산"""
    return (1 + returns).cumprod() - 1

def calculate_volatility(returns, periods=252):
    """변동성 계산 (연환산)"""
    return returns.std() * np.sqrt(periods)

def calculate_sharpe_ratio(returns, risk_free_rate=0.03, periods=252):
    """샤프 비율 계산"""
    annual_return = returns.mean() * periods
    volatility = calculate_volatility(returns, periods)
    return (annual_return - risk_free_rate) / volatility if volatility != 0 else 0

def calculate_max_drawdown(returns):
    """최대 낙폭 계산"""
    cumulative = calculate_cumulative_returns(returns)
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    return drawdown.min()

def calculate_var(returns, confidence_level=0.05):
    """Value at Risk 계산"""
    return np.percentile(returns, confidence_level * 100)

def calculate_beta(stock_returns, market_returns):
    """베타 계산 (시장 대비 민감도)"""
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance != 0 else 0

def calculate_correlation(stock_returns, market_returns):
    """상관계수 계산"""
    return np.corrcoef(stock_returns, market_returns)[0][1]

def get_stock_data(symbol, start_date, end_date, market_type):
    """주식 데이터 가져오기"""
    try:
        if market_type == "국내 (한국)":
            # finance-datareader 사용 (국내)
            df = fdr.DataReader(symbol, start_date, end_date)
            currency = "원"
        else:
            # yfinance 사용 (해외)
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            currency = "달러"
            
        return df, currency
    except Exception as e:
        st.error(f"데이터 가져오기 실패: {str(e)}")
        return None, None

def get_index_data(symbol, start_date, end_date, market_type):
    """시장지수 데이터 가져오기"""
    try:
        if market_type == "국내 (한국)":
            # finance-datareader 사용 (국내 지수)
            df = fdr.DataReader(symbol, start_date, end_date)
        else:
            # yfinance 사용 (해외 지수)
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
        return df
    except Exception as e:
        st.error(f"지수 데이터 가져오기 실패: {str(e)}")
        return None

def normalize_prices(price_series, base_value=100):
    """가격을 기준점 대비 상대적 변화율로 정규화"""
    return (price_series / price_series.iloc[0]) * base_value

# 메인 화면
if analyze_button and stock_symbol:
    try:
        # 로딩 메시지
        with st.spinner(f"📊 {selected_stock_name} 데이터를 불러오는 중..."):
            
            # 데이터 가져오기
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_options[selected_period])
            
            df, currency = get_stock_data(stock_symbol, start_date, end_date, market_type)
            
            # 시장지수 데이터 가져오기
            index_df = None
            if compare_with_index:
                with st.spinner(f"📊 {selected_index_name} 지수 데이터를 불러오는 중..."):
                    index_df = get_index_data(index_symbol, start_date, end_date, market_type)
            
            if df is None or df.empty:
                st.error("❌ 데이터를 찾을 수 없습니다. 종목 코드/심볼을 확인해주세요.")
                st.info("💡 **팁**: 국내 주식은 6자리 숫자 (예: 005930), 해외 주식은 영문 심볼 (예: AAPL)을 입력하세요.")
            else:
                # 데이터 전처리
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # 컬럼명 통일
                if 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                
                # 지수 데이터 전처리
                if index_df is not None and not index_df.empty:
                    index_df.index = pd.to_datetime(index_df.index)
                    index_df = index_df.sort_index()
                    if 'Adj Close' in index_df.columns:
                        index_df['Close'] = index_df['Adj Close']
                
                # 기본 정보 표시
                if compare_with_index and index_df is not None:
                    st.success(f"✅ {selected_stock_name} 및 {selected_index_name} 데이터 로드 완료!")
                else:
                    st.success(f"✅ {selected_stock_name} 데이터 로드 완료! ({len(df)}일간 데이터)")
                
                # 탭 생성
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 주가 차트", "📊 수익률 분석", "📋 주요 지표", "📉 리스크 분석", "💡 투자 분석"])
                
                with tab1:
                    st.subheader("📈 주가 차트")
                    
                    # 차트 옵션
                    chart_type = st.radio(
                        "차트 타입 선택",
                        ["캔들스틱 + 거래량", "가격 비교 (정규화)", "수익률 비교"],
                        horizontal=True
                    )
                    
                    if chart_type == "캔들스틱 + 거래량":
                        # 기존 캔들스틱 차트
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('주가', '거래량'),
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3]
                        )
                        
                        # 캔들스틱
                        fig.add_trace(
                            go.Candlestick(
                                x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name=selected_stock_name,
                                increasing_line_color='red',
                                decreasing_line_color='blue'
                            ),
                            row=1, col=1
                        )
                        
                        # 이동평균선 추가
                        if len(df) >= 20:
                            ma20 = df['Close'].rolling(window=20).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=ma20,
                                    mode='lines',
                                    name='20일 이평선',
                                    line=dict(color='orange', width=1)
                                ),
                                row=1, col=1
                            )
                        
                        if len(df) >= 60:
                            ma60 = df['Close'].rolling(window=60).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=ma60,
                                    mode='lines',
                                    name='60일 이평선',
                                    line=dict(color='purple', width=1)
                                ),
                                row=1, col=1
                            )
                        
                        # 거래량
                        colors = ['red' if close >= open else 'blue' 
                                 for close, open in zip(df['Close'], df['Open'])]
                        
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=df['Volume'],
                                name='거래량',
                                marker_color=colors,
                                opacity=0.7
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            title=f"{selected_stock_name} 주가 차트",
                            height=600,
                            xaxis_rangeslider_visible=False,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "가격 비교 (정규화)" and compare_with_index and index_df is not None:
                        # 정규화된 가격 비교 차트
                        fig = go.Figure()
                        
                        # 주식 가격 정규화
                        stock_normalized = normalize_prices(df['Close'])
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=stock_normalized,
                                mode='lines',
                                name=selected_stock_name,
                                line=dict(color='blue', width=2)
                            )
                        )
                        
                        # 지수 가격 정규화
                        index_normalized = normalize_prices(index_df['Close'])
                        fig.add_trace(
                            go.Scatter(
                                x=index_df.index,
                                y=index_normalized,
                                mode='lines',
                                name=selected_index_name,
                                line=dict(color='red', width=2)
                            )
                        )
                        
                        fig.update_layout(
                            title=f"{selected_stock_name} vs {selected_index_name} 가격 비교 (기준점 100)",
                            xaxis_title="날짜",
                            yaxis_title="정규화된 가격",
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "수익률 비교" and compare_with_index and index_df is not None:
                        # 수익률 비교 차트
                        stock_returns = calculate_returns(df['Close'])
                        stock_cumulative = calculate_cumulative_returns(stock_returns) * 100
                        
                        index_returns = calculate_returns(index_df['Close'])
                        index_cumulative = calculate_cumulative_returns(index_returns) * 100
                        
                        fig = go.Figure()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=stock_cumulative.index,
                                y=stock_cumulative,
                                mode='lines',
                                name=f"{selected_stock_name} 누적수익률",
                                line=dict(color='blue', width=2),
                                fill='tonexty'
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=index_cumulative.index,
                                y=index_cumulative,
                                mode='lines',
                                name=f"{selected_index_name} 누적수익률",
                                line=dict(color='red', width=2)
                            )
                        )
                        
                        fig.update_layout(
                            title=f"{selected_stock_name} vs {selected_index_name} 누적 수익률 비교",
                            xaxis_title="날짜",
                            yaxis_title="누적 수익률 (%)",
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 현재 주가 정보
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "현재가", 
                            f"{current_price:,.0f} {currency}",
                            delta=f"{change:+.0f} ({change_pct:+.2f}%)"
                        )
                    with col2:
                        high_52w = df['High'].max()
                        st.metric("최고가", f"{high_52w:,.0f} {currency}")
                    with col3:
                        low_52w = df['Low'].min()
                        st.metric("최저가", f"{low_52w:,.0f} {currency}")
                    with col4:
                        avg_volume = df['Volume'].mean()
                        st.metric("평균 거래량", f"{avg_volume:,.0f}")
                    
                    # 시장지수와의 비교 정보
                    if compare_with_index and index_df is not None:
                        st.markdown("---")
                        st.markdown("### 📊 시장지수 비교")
                        
                        # 지수 현재 정보
                        index_current = index_df['Close'].iloc[-1]
                        index_prev = index_df['Close'].iloc[-2] if len(index_df) > 1 else index_current
                        index_change = index_current - index_prev
                        index_change_pct = (index_change / index_prev) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                f"{selected_index_name}",
                                f"{index_current:,.2f}",
                                delta=f"{index_change:+.2f} ({index_change_pct:+.2f}%)"
                            )
                        
                        # 베타와 상관계수 계산
                        if len(df) > 30 and len(index_df) > 30:
                            # 공통 날짜 범위로 맞추기
                            common_dates = df.index.intersection(index_df.index)
                            if len(common_dates) > 30:
                                stock_returns_common = calculate_returns(df.loc[common_dates, 'Close'])
                                index_returns_common = calculate_returns(index_df.loc[common_dates, 'Close'])
                                
                                # NaN 제거
                                valid_idx = ~(stock_returns_common.isna() | index_returns_common.isna())
                                stock_returns_clean = stock_returns_common[valid_idx]
                                index_returns_clean = index_returns_common[valid_idx]
                                
                                if len(stock_returns_clean) > 10:
                                    beta = calculate_beta(stock_returns_clean, index_returns_clean)
                                    correlation = calculate_correlation(stock_returns_clean, index_returns_clean)
                                    
                                    with col2:
                                        st.metric("베타 (β)", f"{beta:.2f}")
                                    with col3:
                                        st.metric("상관계수", f"{correlation:.2f}")
                                    with col4:
                                        # 기간 수익률 비교
                                        stock_period_return = (current_price / df['Close'].iloc[0] - 1) * 100
                                        index_period_return = (index_current / index_df['Close'].iloc[0] - 1) * 100
                                        outperformance = stock_period_return - index_period_return
                                        st.metric("시장 대비 초과수익", f"{outperformance:+.2f}%")
                        
                        # 베타 설명
                        st.markdown("""
                        <div class="info-box">
                            <h4>📊 지표 설명</h4>
                            <p><strong>베타(β)</strong>: 시장 대비 주식의 민감도 (1.0=시장과 동일, >1.0=시장보다 변동성 높음)</p>
                            <p><strong>상관계수</strong>: 시장과의 움직임 동조성 (-1~1, 1에 가까울수록 동조)</p>
                            <p><strong>초과수익</strong>: 같은 기간 시장지수 대비 추가 수익률</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 나머지 탭들은 기존과 동일하므로 생략 (수익률 분석, 주요 지표, 리스크 분석, 투자 분석)
                # 여기서는 주요 변경사항인 주가 차트 탭만 수정했습니다.
                
                with tab2:
                    st.subheader("📊 수익률 분석")
                    
                    # 수익률 계산
                    returns = calculate_returns(df['Close'])
                    cumulative_returns = calculate_cumulative_returns(returns)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 누적 수익률 차트 (시장지수 포함)
                        fig_returns = go.Figure()
                        
                        # 주식 누적 수익률
                        fig_returns.add_trace(
                            go.Scatter(
                                x=cumulative_returns.index,
                                y=cumulative_returns * 100,
                                mode='lines',
                                name=f'{selected_stock_name} 누적수익률',
                                line=dict(color='blue', width=2)
                            )
                        )
                        
                        # 시장지수 누적 수익률 (비교 옵션이 켜져있을 때)
                        if compare_with_index and index_df is not None:
                            index_returns = calculate_returns(index_df['Close'])
                            index_cumulative_returns = calculate_cumulative_returns(index_returns)
                            
                            fig_returns.add_trace(
                                go.Scatter(
                                    x=index_cumulative_returns.index,
                                    y=index_cumulative_returns * 100,
                                    mode='lines',
                                    name=f'{selected_index_name} 누적수익률',
                                    line=dict(color='red', width=2)
                                )
                            )
                        
                        fig_returns.update_layout(
                            title="누적 수익률 비교 (%)",
                            xaxis_title="날짜",
                            yaxis_title="수익률 (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_returns, use_container_width=True)
                    
                    with col2:
                        # 수익률 히스토그램
                        fig_hist = px.histogram(
                            returns * 100, 
                            nbins=50,
                            title="일일 수익률 분포",
                            labels={'value': '일일 수익률 (%)', 'count': '빈도'}
                        )
                        fig_hist.update_layout(height=400)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # 수익률 통계 비교
                    st.markdown("### 📈 수익률 통계 비교")
                    
                    if compare_with_index and index_df is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{selected_stock_name}**")
                            daily_return_mean = returns.mean() * 100
                            daily_return_std = returns.std() * 100
                            positive_days = (returns > 0).sum()
                            total_days = len(returns)
                            win_rate = (positive_days / total_days) * 100
                            
                            st.metric("평균 일일 수익률", f"{daily_return_mean:.3f}%")
                            st.metric("일일 수익률 표준편차", f"{daily_return_std:.3f}%")
                            st.metric("상승일 비율", f"{win_rate:.1f}%")
                        
                        with col2:
                            st.markdown(f"**{selected_index_name}**")
                            index_returns = calculate_returns(index_df['Close'])
                            index_daily_return_mean = index_returns.mean() * 100
                            index_daily_return_std = index_returns.std() * 100
                            index_positive_days = (index_returns > 0).sum()
                            index_total_days = len(index_returns)
                            index_win_rate = (index_positive_days / index_total_days) * 100
                            
                            st.metric("평균 일일 수익률", f"{index_daily_return_mean:.3f}%")
                            st.metric("일일 수익률 표준편차", f"{index_daily_return_std:.3f}%")
                            st.metric("상승일 비율", f"{index_win_rate:.1f}%")
                    else:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            daily_return_mean = returns.mean() * 100
                            st.metric("평균 일일 수익률", f"{daily_return_mean:.3f}%")
                        
                        with col2:
                            daily_return_std = returns.std() * 100
                            st.metric("일일 수익률 표준편차", f"{daily_return_std:.3f}%")
                        
                        with col3:
                            positive_days = (returns > 0).sum()
                            total_days = len(returns)
                            win_rate = (positive_days / total_days) * 100
                            st.metric("상승일 비율", f"{win_rate:.1f}%")
                
                with tab3:
                    st.subheader("📋 주요 투자 지표")
                    
                    # 수익률 계산
                    total_return = (current_price / df['Close'].iloc[0] - 1) * 100
                    
                    # 연환산 수익률
                    days = len(df)
                    annual_return = ((current_price / df['Close'].iloc[0]) ** (365/days) - 1) * 100
                    
                    # 변동성 (연환산)
                    volatility = calculate_volatility(returns) * 100
                    
                    # 샤프 비율
                    sharpe_ratio = calculate_sharpe_ratio(returns)
                    
                    # 최대 낙폭 (MDD)
                    max_drawdown = calculate_max_drawdown(returns) * 100
                    
                    # 시장 지수와 비교 (있는 경우)
                    if compare_with_index and index_df is not None:
                        index_returns = calculate_returns(index_df['Close'])
                        index_total_return = (index_current / index_df['Close'].iloc[0] - 1) * 100
                        index_annual_return = ((index_current / index_df['Close'].iloc[0]) ** (365/len(index_df)) - 1) * 100
                        index_volatility = calculate_volatility(index_returns) * 100
                        index_sharpe_ratio = calculate_sharpe_ratio(index_returns)
                        index_max_drawdown = calculate_max_drawdown(index_returns) * 100
                        
                        # 지표 비교 표시
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"### 📈 {selected_stock_name} 지표")
                            st.markdown(f"""
                            <div class="metric-card {'success-metric' if total_return > 0 else 'danger-metric'}">
                                <h4>총 수익률</h4>
                                <h3>{total_return:+.2f}%</h3>
                                <p>{selected_period} 동안의 누적 수익률</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card {'success-metric' if annual_return > 0 else 'danger-metric'}">
                                <h4>연환산 수익률</h4>
                                <h3>{annual_return:+.2f}%</h3>
                                <p>연간 예상 수익률</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            volatility_class = "success-metric" if volatility < 20 else "warning-metric" if volatility < 30 else "danger-metric"
                            volatility_desc = "안정적" if volatility < 20 else "보통" if volatility < 30 else "높음"
                            
                            st.markdown(f"""
                            <div class="metric-card {volatility_class}">
                                <h4>변동성 (연환산)</h4>
                                <h3>{volatility:.2f}%</h3>
                                <p>위험도: {volatility_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"### 📊 {selected_index_name} 지표")
                            st.markdown(f"""
                            <div class="metric-card {'success-metric' if index_total_return > 0 else 'danger-metric'}">
                                <h4>총 수익률</h4>
                                <h3>{index_total_return:+.2f}%</h3>
                                <p>{selected_period} 동안의 누적 수익률</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card {'success-metric' if index_annual_return > 0 else 'danger-metric'}">
                                <h4>연환산 수익률</h4>
                                <h3>{index_annual_return:+.2f}%</h3>
                                <p>연간 예상 수익률</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            index_volatility_class = "success-metric" if index_volatility < 20 else "warning-metric" if index_volatility < 30 else "danger-metric"
                            index_volatility_desc = "안정적" if index_volatility < 20 else "보통" if index_volatility < 30 else "높음"
                            
                            st.markdown(f"""
                            <div class="metric-card {index_volatility_class}">
                                <h4>변동성 (연환산)</h4>
                                <h3>{index_volatility:.2f}%</h3>
                                <p>위험도: {index_volatility_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 상대적 성과 분석
                        st.markdown("### ⚖️ 상대적 성과 분석")
                        outperformance = total_return - index_total_return
                        volatility_diff = volatility - index_volatility
                        sharpe_diff = sharpe_ratio - index_sharpe_ratio
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            perf_class = "success-metric" if outperformance > 0 else "danger-metric"
                            st.markdown(f"""
                            <div class="metric-card {perf_class}">
                                <h4>초과 수익률</h4>
                                <h3>{outperformance:+.2f}%</h3>
                                <p>시장 대비 추가 수익</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            vol_class = "success-metric" if volatility_diff < 0 else "danger-metric"
                            st.markdown(f"""
                            <div class="metric-card {vol_class}">
                                <h4>변동성 차이</h4>
                                <h3>{volatility_diff:+.2f}%</h3>
                                <p>시장 대비 위험도</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            sharpe_class = "success-metric" if sharpe_diff > 0 else "danger-metric"
                            st.markdown(f"""
                            <div class="metric-card {sharpe_class}">
                                <h4>샤프 비율 차이</h4>
                                <h3>{sharpe_diff:+.2f}</h3>
                                <p>위험조정 수익률</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:
                        # 기존 단일 지표 표시
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📈 수익률 지표")
                            st.markdown(f"""
                            <div class="metric-card {'success-metric' if total_return > 0 else 'danger-metric'}">
                                <h4>총 수익률</h4>
                                <h3>{total_return:+.2f}%</h3>
                                <p>{selected_period} 동안의 누적 수익률</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card {'success-metric' if annual_return > 0 else 'danger-metric'}">
                                <h4>연환산 수익률</h4>
                                <h3>{annual_return:+.2f}%</h3>
                                <p>연간 예상 수익률</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### ⚠️ 리스크 지표")
                            volatility_class = "success-metric" if volatility < 20 else "warning-metric" if volatility < 30 else "danger-metric"
                            volatility_desc = "안정적" if volatility < 20 else "보통" if volatility < 30 else "높음"
                            
                            st.markdown(f"""
                            <div class="metric-card {volatility_class}">
                                <h4>변동성 (연환산)</h4>
                                <h3>{volatility:.2f}%</h3>
                                <p>위험도: {volatility_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            mdd_class = "success-metric" if max_drawdown > -10 else "warning-metric" if max_drawdown > -20 else "danger-metric"
                            mdd_desc = "양호" if max_drawdown > -10 else "주의" if max_drawdown > -20 else "위험"
                            
                            st.markdown(f"""
                            <div class="metric-card {mdd_class}">
                                <h4>최대 낙폭 (MDD)</h4>
                                <h3>{max_drawdown:.2f}%</h3>
                                <p>위험도: {mdd_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # 샤프 비율
                    st.markdown("### 🎯 위험 조정 수익률")
                    
                    if compare_with_index and index_df is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sharpe_color = "success-metric" if sharpe_ratio > 1 else "warning-metric" if sharpe_ratio > 0 else "danger-metric"
                            sharpe_grade = "우수" if sharpe_ratio > 1 else "보통" if sharpe_ratio > 0 else "부족"
                            
                            st.markdown(f"""
                            <div class="metric-card {sharpe_color}">
                                <h4>{selected_stock_name} 샤프 비율</h4>
                                <h3>{sharpe_ratio:.2f}</h3>
                                <p>위험 대비 수익률: {sharpe_grade}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            index_sharpe_color = "success-metric" if index_sharpe_ratio > 1 else "warning-metric" if index_sharpe_ratio > 0 else "danger-metric"
                            index_sharpe_grade = "우수" if index_sharpe_ratio > 1 else "보통" if index_sharpe_ratio > 0 else "부족"
                            
                            st.markdown(f"""
                            <div class="metric-card {index_sharpe_color}">
                                <h4>{selected_index_name} 샤프 비율</h4>
                                <h3>{index_sharpe_ratio:.2f}</h3>
                                <p>위험 대비 수익률: {index_sharpe_grade}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        sharpe_color = "success-metric" if sharpe_ratio > 1 else "warning-metric" if sharpe_ratio > 0 else "danger-metric"
                        sharpe_grade = "우수" if sharpe_ratio > 1 else "보통" if sharpe_ratio > 0 else "부족"
                        
                        st.markdown(f"""
                        <div class="metric-card {sharpe_color}">
                            <h4>샤프 비율</h4>
                            <h3>{sharpe_ratio:.2f}</h3>
                            <p>위험 대비 수익률: {sharpe_grade}</p>
                            <small>1.0 이상: 우수, 0~1.0: 보통, 0 미만: 부족</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab4:
                    st.subheader("📉 리스크 분석")
                    
                    # VaR 계산
                    var_95 = calculate_var(returns, 0.05) * 100
                    var_99 = calculate_var(returns, 0.01) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 💰 Value at Risk (VaR)")
                        st.markdown(f"""
                        <div class="metric-card danger-metric">
                            <h4>VaR (95%)</h4>
                            <h3>{var_95:.2f}%</h3>
                            <p>95% 확률로 하루 손실이 이 값을 넘지 않음</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-card danger-metric">
                            <h4>VaR (99%)</h4>
                            <h3>{var_99:.2f}%</h3>
                            <p>99% 확률로 하루 손실이 이 값을 넘지 않음</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 시장지수 VaR과 비교
                        if compare_with_index and index_df is not None:
                            index_var_95 = calculate_var(index_returns, 0.05) * 100
                            index_var_99 = calculate_var(index_returns, 0.01) * 100
                            
                            st.markdown(f"### 📊 {selected_index_name} VaR")
                            st.markdown(f"""
                            <div class="metric-card warning-metric">
                                <h4>시장 VaR (95%)</h4>
                                <h3>{index_var_95:.2f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card warning-metric">
                                <h4>시장 VaR (99%)</h4>
                                <h3>{index_var_99:.2f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 위험도 설명
                        st.markdown("""
                        <div class="info-box">
                            <h4>💡 VaR 해석</h4>
                            <p>• VaR이 -3% 이하면 위험도가 높은 종목</p>
                            <p>• VaR이 -1% 이상이면 비교적 안정적인 종목</p>
                            <p>• 투자 전 본인의 손실 감수 능력을 고려하세요</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # 드로우다운 차트 (시장지수 포함)
                        cumulative_max = cumulative_returns.expanding().max()
                        drawdown = cumulative_returns - cumulative_max
                        
                        fig_dd = go.Figure()
                        
                        # 주식 드로우다운
                        fig_dd.add_trace(
                            go.Scatter(
                                x=drawdown.index,
                                y=drawdown * 100,
                                mode='lines',
                                fill='tonexty',
                                name=f'{selected_stock_name} 드로우다운',
                                line=dict(color='red'),
                                fillcolor='rgba(255,0,0,0.3)'
                            )
                        )
                        
                        # 시장지수 드로우다운 (비교 옵션이 켜져있을 때)
                        if compare_with_index and index_df is not None:
                            index_cumulative_max = index_cumulative_returns.expanding().max()
                            index_drawdown = index_cumulative_returns - index_cumulative_max
                            
                            fig_dd.add_trace(
                                go.Scatter(
                                    x=index_drawdown.index,
                                    y=index_drawdown * 100,
                                    mode='lines',
                                    name=f'{selected_index_name} 드로우다운',
                                    line=dict(color='blue', dash='dash'),
                                )
                            )
                        
                        fig_dd.update_layout(
                            title="드로우다운 비교 차트",
                            xaxis_title="날짜",
                            yaxis_title="드로우다운 (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_dd, use_container_width=True)
                        
                        # 드로우다운 통계
                        max_dd_duration = 0
                        current_dd_duration = 0
                        peak = cumulative_returns.iloc[0]
                        
                        for i, val in enumerate(cumulative_returns):
                            if val >= peak:
                                peak = val
                                current_dd_duration = 0
                            else:
                                current_dd_duration += 1
                                max_dd_duration = max(max_dd_duration, current_dd_duration)
                        
                        st.metric("최대 하락 지속일", f"{max_dd_duration}일")
                        
                        # 시장지수와 MDD 비교
                        if compare_with_index and index_df is not None:
                            st.metric(f"{selected_stock_name} MDD", f"{max_drawdown:.2f}%")
                            st.metric(f"{selected_index_name} MDD", f"{index_max_drawdown:.2f}%")
                
                with tab5:
                    st.subheader("💡 종합 투자 분석")
                    
                    # 종합 점수 계산 (시장 대비 성과 포함)
                    score = 0
                    analysis_points = []
                    
                    # 수익률 평가
                    if total_return > 20:
                        score += 2
                        analysis_points.append("✅ 높은 수익률을 기록하고 있습니다.")
                    elif total_return > 0:
                        score += 1
                        analysis_points.append("📈 플러스 수익률을 유지하고 있습니다.")
                    else:
                        analysis_points.append("📉 현재 손실 상태입니다. 신중한 판단이 필요합니다.")
                    
                    # 시장 대비 성과 평가 (있는 경우)
                    if compare_with_index and index_df is not None:
                        if outperformance > 5:
                            score += 2
                            analysis_points.append("✅ 시장을 크게 상회하는 성과를 보입니다.")
                        elif outperformance > 0:
                            score += 1
                            analysis_points.append("📊 시장보다 우수한 성과를 보입니다.")
                        else:
                            analysis_points.append("📉 시장 대비 저조한 성과를 보입니다.")
                    
                    # 변동성 평가
                    if volatility < 15:
                        score += 2
                        analysis_points.append("✅ 변동성이 낮아 안정적인 투자처입니다.")
                    elif volatility < 25:
                        score += 1
                        analysis_points.append("⚖️ 적당한 변동성을 보이고 있습니다.")
                    else:
                        analysis_points.append("⚠️ 높은 변동성으로 위험도가 큽니다.")
                    
                    # 샤프 비율 평가
                    if sharpe_ratio > 1.5:
                        score += 2
                        analysis_points.append("✅ 위험 대비 수익률이 매우 우수합니다.")
                    elif sharpe_ratio > 0.5:
                        score += 1
                        analysis_points.append("📊 위험 대비 수익률이 양호합니다.")
                    else:
                        analysis_points.append("⚠️ 위험 대비 수익률이 부족합니다.")
                    
                    # MDD 평가
                    if max_drawdown > -10:
                        score += 2
                        analysis_points.append("✅ 낙폭이 적어 심리적 부담이 적습니다.")
                    elif max_drawdown > -20:
                        score += 1
                        analysis_points.append("⚖️ 적당한 수준의 낙폭을 보입니다.")
                    else:
                        analysis_points.append("⚠️ 큰 낙폭으로 심리적 부담이 클 수 있습니다.")
                    
                    # 종합 평가
                    if score >= 8:
                        grade = "A+"
                        grade_color = "success-metric"
                        recommendation = "매우 우수한 투자처로 평가됩니다."
                    elif score >= 6:
                        grade = "A"
                        grade_color = "success-metric"
                        recommendation = "좋은 투자처로 평가됩니다."
                    elif score >= 4:
                        grade = "B"
                        grade_color = "warning-metric"
                        recommendation = "양호한 투자처로 평가됩니다."
                    elif score >= 2:
                        grade = "C"
                        grade_color = "warning-metric"
                        recommendation = "보통 수준의 투자처입니다."
                    else:
                        grade = "D"
                        grade_color = "danger-metric"
                        recommendation = "신중한 접근이 필요한 투자처입니다."
                    
                    # 결과 표시
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card {grade_color}">
                            <h4>종합 평가</h4>
                            <h2>{grade}등급</h2>
                            <p>{recommendation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📝 상세 분석")
                        for point in analysis_points:
                            st.markdown(f"- {point}")
                    
                    # 시장 대비 성과 요약 (있는 경우)
                    if compare_with_index and index_df is not None:
                        st.markdown("---")
                        st.markdown("### 📊 시장 대비 성과 요약")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if outperformance > 0:
                                st.success(f"🎯 **시장 초과수익**: +{outperformance:.2f}%")
                            else:
                                st.error(f"📉 **시장 저조수익**: {outperformance:.2f}%")
                        
                        with col2:
                            if len(stock_returns_clean) > 10:
                                if beta > 1.1:
                                    st.warning(f"⚡ **고베타**: {beta:.2f} (시장보다 변동성 높음)")
                                elif beta < 0.9:
                                    st.info(f"🛡️ **저베타**: {beta:.2f} (시장보다 안정적)")
                                else:
                                    st.success(f"⚖️ **중성베타**: {beta:.2f} (시장과 유사)")
                        
                        with col3:
                            if len(stock_returns_clean) > 10:
                                if correlation > 0.7:
                                    st.info(f"🔗 **높은 상관성**: {correlation:.2f}")
                                elif correlation < 0.3:
                                    st.warning(f"🆓 **낮은 상관성**: {correlation:.2f}")
                                else:
                                    st.success(f"📊 **적당한 상관성**: {correlation:.2f}")
                    
                    # 투자 전략 제안
                    st.markdown("### 💰 맞춤형 투자 전략")
                    
                    strategy_suggestions = []
                    
                    if compare_with_index and index_df is not None:
                        if outperformance > 5 and volatility < 25:
                            strategy_suggestions.append("🎯 **핵심 포트폴리오 편입**: 시장 대비 우수한 성과와 적절한 위험도로 핵심 보유 종목으로 적합합니다.")
                        elif outperformance > 0 and beta > 1.2:
                            strategy_suggestions.append("📈 **상승장 전략**: 시장 상승 시 더 큰 수익을 기대할 수 있는 공격적 종목입니다.")
                        elif beta < 0.8 and max_drawdown > -15:
                            strategy_suggestions.append("🛡️ **방어적 투자**: 시장 하락 시 상대적으로 안전한 방어주 성격을 보입니다.")
                        elif correlation < 0.5:
                            strategy_suggestions.append("🎯 **분산 투자**: 시장과 낮은 상관관계로 포트폴리오 분산 효과를 기대할 수 있습니다.")
                    
                    if total_return > 0 and volatility < 20:
                        strategy_suggestions.append("📈 **장기 보유 전략**: 안정적인 수익과 낮은 변동성으로 장기 투자에 적합합니다.")
                    
                    if volatility > 30:
                        strategy_suggestions.append("🎯 **적립식 투자**: 높은 변동성을 완화하기 위해 분할 매수를 고려해보세요.")
                    
                    if sharpe_ratio > 1:
                        strategy_suggestions.append("⚖️ **포트폴리오 핵심 종목**: 위험 대비 수익률이 우수하여 포트폴리오의 핵심 종목으로 고려할 수 있습니다.")
                    
                    if max_drawdown < -20:
                        strategy_suggestions.append("🛡️ **손절매 설정**: 큰 낙폭 가능성이 있으므로 손절매 라인을 미리 설정하는 것이 좋습니다.")
                    
                    if not strategy_suggestions:
                        strategy_suggestions.append("📊 **균형 잡힌 접근**: 현재 지표들을 종합적으로 고려하여 신중한 투자를 권장합니다.")
                    
                    for suggestion in strategy_suggestions:
                        st.markdown(f"- {suggestion}")
                    
                    # 주의사항
                    st.markdown("---")
                    st.markdown("### ⚠️ 투자 주의사항")
                    st.markdown("""
                    - 이 분석은 **과거 데이터 기반**이며, 미래 수익을 보장하지 않습니다
                    - **분산투자**를 통해 리스크를 분산시키세요
                    - 투자 전 **본인의 투자 성향과 목표**를 명확히 하세요
                    - **정기적인 포트폴리오 리밸런싱**을 고려하세요
                    - 투자는 **여유자금**으로만 하시기 바랍니다
                    - 시장지수와의 비교는 **상대적 성과 파악**에 도움이 되지만, **절대적 투자 기준**은 아닙니다
                    """)
                    
    except Exception as e:
        st.error(f"❌ 오류가 발생했습니다: {str(e)}")
        st.markdown("종목 코드/심볼을 확인하거나 다른 종목을 시도해보세요.")

elif not stock_symbol and analyze_button:
    st.warning("⚠️ 종목을 선택하거나 입력해주세요.")

else:
    # 초기 화면
    st.markdown("""
    ## 🚀 사용 방법
    
    1. **왼쪽 사이드바**에서 시장을 선택하세요 (국내/해외)
    2. **종목**을 선택하거나 직접 입력하세요
    3. **분석 기간**을 선택하세요
    4. **시장지수 비교 옵션**을 설정하세요 ✨
    5. **'분석 시작'** 버튼을 클릭하세요
    
    ## 📊 제공되는 분석 (V2 업데이트)
    
    - **주가 차트**: 캔들스틱, 가격비교(정규화), 수익률비교 차트
    - **시장지수 비교**: KOSPI/KOSDAQ (국내), S&P500/NASDAQ (해외) 
    - **베타/상관계수**: 시장 대비 민감도 및 상관관계 분석
    - **상대적 성과**: 시장 대비 초과수익률 및 위험도 비교
    - **종합 평가**: 시장 대비 성과를 반영한 투자 등급
    
    ## 🆕 V2의 새로운 기능
    
    ### 📈 시장지수 비교
    - **국내**: KOSPI, KOSDAQ, KRX100 지수와 비교
    - **해외**: S&P500, NASDAQ, Dow Jones 지수와 비교
    - **베타 계수**: 시장 대비 주식의 민감도 측정
    - **상관계수**: 시장과의 동조성 분석
    
    ### 📊 다양한 차트 옵션
    - **캔들스틱 + 거래량**: 전통적인 주가 차트
    - **가격 비교 (정규화)**: 동일 기준점에서 상대적 성과 비교
    - **수익률 비교**: 누적 수익률 직접 비교
    
    ### 🎯 향상된 분석
    - **상대적 성과 분석**: 시장 대비 초과수익률
    - **위험 조정 성과**: 시장 대비 샤프 비율
    - **맞춤형 투자 전략**: 시장 특성을 고려한 투자 조언
    
    ## 💡 팁
    
    - **국내 주식**: 6자리 숫자 코드 (예: 005930)
    - **해외 주식**: 영문 심볼 (예: AAPL, TSLA)
    - **시장지수 비교**: 상대적 성과 파악에 매우 유용
    - **베타 해석**: 1.0 초과 시 시장보다 변동성 높음
    
    ## 🎯 추천 테스트 종목
    
    ### 국내 📈
    - **대형주**: 삼성전자(005930) vs KOSPI
    - **기술주**: NAVER(035420) vs KOSDAQ
    - **금융주**: KB금융(105560) vs KOSPI
    
    ### 해외 🌍
    - **기술주**: Apple(AAPL) vs S&P500
    - **성장주**: Tesla(TSLA) vs NASDAQ
    - **가치주**: Berkshire Hathaway(BRK-B) vs S&P500
    
    ## 📈 투자 지표 설명
    
    ### 🎯 베타 (Beta)
    - 시장 대비 주식의 민감도를 측정
    - **1.0**: 시장과 동일한 움직임
    - **1.0 초과**: 시장보다 변동성 높음 (공격적)
    - **1.0 미만**: 시장보다 안정적 (방어적)
    
    ### 🔗 상관계수 (Correlation)
    - 시장과의 움직임 동조성 측정 (-1 ~ 1)
    - **0.7 이상**: 시장과 높은 동조성
    - **0.3 미만**: 시장과 독립적 움직임
    - **음수**: 시장과 반대 방향 움직임
    
    ### 📊 초과수익률 (Alpha)
    - 같은 기간 시장지수 대비 추가 수익률
    - **양수**: 시장 대비 우수한 성과
    - **음수**: 시장 대비 저조한 성과
    """)
    
    # 샘플 데이터 미리보기
    with st.expander("📋 지원 시장지수 및 종목 리스트"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🇰🇷 국내 시장")
            st.markdown("**📊 지원 지수:**")
            st.markdown("- KOSPI (종합주가지수)")
            st.markdown("- KOSDAQ (코스닥지수)")  
            st.markdown("- KRX100 (KRX100지수)")
            
            st.markdown("**📈 인기 종목:**")
            st.markdown("""
            - 삼성전자 (005930)
            - SK하이닉스 (000660)
            - NAVER (035420)
            - 카카오 (035720)
            - LG화학 (051910)
            - 현대차 (005380)
            - KB금융 (105560)
            - 셀트리온 (068270)
            """)
        
        with col2:
            st.markdown("### 🇺🇸 해외 시장")
            st.markdown("**📊 지원 지수:**")
            st.markdown("- S&P 500 (미국 대형주)")
            st.markdown("- NASDAQ (기술주 중심)")
            st.markdown("- Dow Jones (다우존스)")
            
            st.markdown("**📈 인기 종목:**")
            st.markdown("""
            - Apple (AAPL)
            - Microsoft (MSFT)
            - Google (GOOGL)
            - Amazon (AMZN)
            - Tesla (TSLA)
            - NVIDIA (NVDA)
            - Meta (META)
            - Netflix (NFLX)
            """)
    
    # 새 기능 안내
    st.markdown("---")
    st.info("🆕 **V2 업데이트**: 시장지수 비교 기능이 추가되어 더욱 정확한 투자 분석이 가능합니다!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔍 정확한 분석")
        st.markdown("시장 대비 상대적 성과로 더 정확한 투자 판단")
    
    with col2:
        st.markdown("#### 📊 다양한 차트")
        st.markdown("3가지 차트 옵션으로 다각도 분석")
    
    with col3:
        st.markdown("#### 🎯 맞춤 전략")
        st.markdown("베타와 상관계수 기반 투자 전략 제안")
