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
@st.cache_data(ttl=3600)  # 1시간 캐시
def get_all_stock_table():
    try:
        # KRX 데이터 로딩
        krx = fdr.StockListing('KRX')
        krx = krx.rename(columns={'Code': 'Code', 'Name': 'Name'})
        krx = krx[['Code', 'Name']].drop_duplicates()
        krx = krx[krx['Code'].str.len() == 6]
        krx['Market'] = 'KRX'

        all_dfs = [krx]
        
        # 해외 거래소는 선택적으로 로딩
        for market in ['NASDAQ', 'NYSE', 'AMEX']:
            try:
                df = fdr.StockListing(market)
                if 'Symbol' in df.columns:
                    df = df.rename(columns={'Symbol': 'Code'})
                if 'Name' not in df.columns and 'name' in df.columns:
                    df = df.rename(columns={'name': 'Name'})
                if 'Code' in df.columns and 'Name' in df.columns:
                    df = df[['Code', 'Name']].drop_duplicates()
                    df['Market'] = market
                    all_dfs.append(df)
            except Exception as e:
                st.warning(f"{market} 데이터 로딩 실패: {str(e)}")
                continue

        all_df = pd.concat(all_dfs, ignore_index=True)
        return all_df
    
    except Exception as e:
        st.error(f"주식 데이터 로딩 중 오류 발생: {str(e)}")
        # 최소한 빈 DataFrame 반환
        return pd.DataFrame(columns=['Code', 'Name', 'Market'])

# 안전한 데이터 로딩
try:
    all_stock_table = get_all_stock_table()
    if all_stock_table.empty:
        st.error("주식 데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
        st.stop()
        
    company_options = [f"{row.Name} ({row.Code}) [{row.Market}]" for row in all_stock_table.itertuples()]
    code_to_name = dict(zip(all_stock_table['Code'], all_stock_table['Name']))
    name_to_code = dict(zip(all_stock_table['Name'], all_stock_table['Code']))
    code_to_market = dict(zip(all_stock_table['Code'], all_stock_table['Market']))
    
except Exception as e:
    st.error(f"애플리케이션 초기화 오류: {str(e)}")
    st.stop()

def get_korean_stock_data(ticker, start_year=1981, end_year=None):
    try:
        if end_year is None:
            end_year = datetime.today().year
        start_date = f'{start_year}-01-01'
        end_date = datetime.today().strftime('%Y-%m-%d')
        
        data = fdr.DataReader(ticker, start_date, end_date)
        
        if data is None or data.empty:
            return None, None
            
        yearly_data = data.groupby(data.index.year)['Close'].last()
        returns = yearly_data.pct_change().dropna() * 100
        
        return yearly_data, returns
        
    except Exception as e:
        st.error(f"데이터 로딩 오류 ({ticker}): {str(e)}")
        return None, None

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
            line_width=2
        )
        # 경계선 텍스트를 선 위에 배치
        fig.add_annotation(
            x=zero_bin_idx - 0.5,
            y=max(hist_percentages) * 0.5,  # 차트 중간 높이에 배치
            text="손실/이익 경계",
            showarrow=False,
            font=dict(size=12, color='black'),
            textangle=-90,  # 텍스트를 세로로 회전
            align='center',
            bgcolor="white",  # 배경색 추가로 가독성 향상
            bordercolor="black",
            borderwidth=1
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

    with st.spinner('KOSPI 데이터를 불러오는 중입니다...'):
        yearly_data, returns = get_korean_stock_data('KS11', 1981)
    
    if returns is not None:
        fig = plot_return_histogram(returns, '연간', 'KOSPI', bins, bin_labels, colors)
        st.plotly_chart(fig, use_container_width=True)
        # 코스피 연도별 종가 막대그래프
        price_df = yearly_data.reset_index()
        price_df.columns = ['연도', '종가']
        fig_price = go.Figure(go.Bar(x=price_df['연도'], y=price_df['종가'], marker_color='#4472C4'))
        fig_price.update_layout(
            title="KOSPI 연도별 종가(지수) 막대그래프", 
            xaxis_title='연도', 
            yaxis_title='종가(지수)',
            template='plotly_white'
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("KOSPI 데이터를 불러올 수 없습니다.")

# 사용자 입력 부분
st.header("다른 종목/지수 연 수익률 분포 보기")

# 세션 상태 초기화
if 'selected_company' not in st.session_state:
    default_company = "삼성전자 (005930) [KRX]" if "삼성전자 (005930) [KRX]" in company_options else company_options[0]
    st.session_state.selected_company = default_company

if 'text_input_value' not in st.session_state:
    st.session_state.text_input_value = "삼성전자"

if 'last_selectbox_value' not in st.session_state:
    st.session_state.last_selectbox_value = st.session_state.selected_company

if 'last_textinput_value' not in st.session_state:
    st.session_state.last_textinput_value = st.session_state.text_input_value

# selectbox의 현재 인덱스 찾기
try:
    current_index = company_options.index(st.session_state.selected_company)
except (ValueError, IndexError):
    current_index = 0
    st.session_state.selected_company = company_options[0]

# selectbox
selected = st.selectbox(
    "회사명 또는 티커를 선택하세요", 
    company_options, 
    index=current_index,
    key="company_selectbox"
)

# text_input
user_input = st.text_input(
    "직접 입력 (회사명, 티커, 회사명(티커) 모두 가능)", 
    value=st.session_state.text_input_value,
    key="company_textinput"
)

# selectbox 변경 감지 및 text_input 업데이트
if selected != st.session_state.last_selectbox_value:
    st.session_state.last_selectbox_value = selected
    st.session_state.selected_company = selected
    
    # selectbox에서 선택된 값을 파싱해서 회사명만 추출
    if '(' in selected and ')' in selected:
        company_name = selected.split('(')[0].strip()
        st.session_state.text_input_value = company_name
        st.session_state.last_textinput_value = company_name
        st.rerun()

# text_input 변경 감지 및 selectbox 업데이트
if user_input != st.session_state.last_textinput_value:
    st.session_state.last_textinput_value = user_input
    st.session_state.text_input_value = user_input
    
    # text_input 값으로 매칭되는 옵션 찾기
    if user_input.strip():
        keyword = user_input.strip().lower()
        
        # 정확한 매치 우선 검색
        exact_matches = [opt for opt in company_options if keyword in opt.lower()]
        
        if exact_matches:
            # 가장 유사한 항목 선택 (회사명이나 티커가 정확히 일치하는 것 우선)
            best_match = None
            
            # 1순위: 회사명이 정확히 일치
            for opt in exact_matches:
                company_part = opt.split('(')[0].strip().lower()
                if company_part == keyword:
                    best_match = opt
                    break
            
            # 2순위: 티커가 정확히 일치
            if not best_match:
                for opt in exact_matches:
                    if '(' in opt and ')' in opt:
                        ticker_part = opt.split('(')[1].split(')')[0].strip().lower()
                        if ticker_part == keyword:
                            best_match = opt
                            break
            
            # 3순위: 첫 번째 매치
            if not best_match:
                best_match = exact_matches[0]
            
            if best_match != st.session_state.selected_company:
                st.session_state.selected_company = best_match
                st.session_state.last_selectbox_value = best_match
                st.rerun()

# 유사 검색 결과 표시 (text_input에 값이 있을 때만)
similar_options = []
if user_input.strip() and len(user_input.strip()) >= 2:
    keyword = user_input.strip().lower()
    similar_options = [opt for opt in company_options if keyword in opt.lower()]
    
    if similar_options and len(similar_options) > 1:  # 현재 선택된 것 외에 다른 옵션이 있을 때만 표시
        st.markdown(f"**🔍 '{user_input}' 검색 결과 ({len(similar_options)}개):**")
        
        # 최대 10개까지만 표시
        display_options = similar_options[:10]
        
        for i, option in enumerate(display_options):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                if st.button("선택", key=f"select_btn_{i}"):
                    st.session_state.selected_company = option
                    st.session_state.last_selectbox_value = option
                    # 선택된 항목의 회사명을 text_input에 반영
                    company_name = option.split('(')[0].strip()
                    st.session_state.text_input_value = company_name
                    st.session_state.last_textinput_value = company_name
                    st.rerun()
            with col2:
                st.write(option)

# 분석 실행
start_year = st.number_input("시작 연도", min_value=1981, max_value=datetime.today().year-1, value=2000)
end_year = st.number_input("종료 연도", min_value=start_year+1, max_value=datetime.today().year, value=datetime.today().year)

# 현재 선택된 값으로 분석 실행
if st.button("📊 분석하기", type="primary"):
    # 현재 선택된 회사 정보 사용
    current_selection = st.session_state.selected_company
    ticker, company_name = get_ticker_and_name(current_selection)
    
    st.info(f"분석 대상: {company_name} ({ticker})")
    
    with st.spinner('데이터를 불러오는 중입니다...'):
        yearly_data, returns = get_korean_stock_data(ticker, int(start_year), int(end_year))
    
    if returns is not None and not returns.empty:
        fig = plot_return_histogram(returns, '연간', company_name, bins, bin_labels, colors)
        st.plotly_chart(fig, use_container_width=True)
        
        # 상승/하락 연도 계산 및 표기
        up_years = returns[returns > 0].index.tolist()
        down_years = returns[returns <= 0].index.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("상승 연도 수", len(up_years))
            st.caption(f"상승 연도: {', '.join(map(str, up_years)) if up_years else '없음'}")
        with col2:
            st.metric("하락 연도 수", len(down_years))
            st.caption(f"하락 연도: {', '.join(map(str, down_years)) if down_years else '없음'}")

        # 최고/최저 수익률 및 연도
        max_return = returns.max()
        min_return = returns.min()
        max_year = returns.idxmax() if not returns.empty else '-'
        min_year = returns.idxmin() if not returns.empty else '-'
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("최고 수익률", f"{max_return:.2f}%", delta=f"{max_year}년")
        with col4:
            st.metric("최저 수익률", f"{min_return:.2f}%", delta=f"{min_year}년")

        # 실제 연도별 종가(지수/주가) 막대그래프 추가
        price_df = yearly_data.reset_index()
        price_df.columns = ['연도', '종가']
        fig_price = go.Figure(go.Bar(x=price_df['연도'], y=price_df['종가'], marker_color='#4472C4'))
        fig_price.update_layout(
            title=f"{company_name} 연도별 종가 추이", 
            xaxis_title='연도', 
            yaxis_title='종가',
            template='plotly_white'
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # 상세 데이터 테이블 (접기/펼치기)
        with st.expander("📈 연도별 상세 데이터 보기"):
            detail_df = pd.DataFrame({
                '연도': yearly_data.index,
                '종가': yearly_data.values,
                '수익률(%)': ['-'] + [f"{x:.2f}%" for x in returns.values]
            })
            st.dataframe(detail_df, use_container_width=True)
    else:
        st.error(f"❌ '{company_name} ({ticker})' 데이터를 불러올 수 없습니다.")
        st.info("💡 다른 종목을 선택해보시거나, 티커 심볼을 확인해주세요.")
