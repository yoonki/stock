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

@st.cache_data(ttl=3600)
def get_all_stock_table():
    try:
        krx = fdr.StockListing('KRX')
        krx = krx.rename(columns={'Code': 'Code', 'Name': 'Name'})
        krx = krx[['Code', 'Name']].drop_duplicates()
        krx = krx[krx['Code'].str.len() == 6]
        krx['Market'] = 'KRX'

        all_dfs = [krx]
        
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
        return pd.DataFrame(columns=['Code', 'Name', 'Market'])

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

    fig.add_annotation(
        x=1.5, y=max(hist_percentages) * 0.85,
        text=f"손실 확률: {negative_pct:.1f}%",
        showarrow=False,
        font=dict(size=14, color='black'),
        align='center'
    )
    fig.add_annotation(
        x=len(bin_labels) * 0.7, y=max(hist_percentages) * 0.85,
        text=f"이익 확률: {positive_pct:.1f}%",
        showarrow=False,
        font=dict(size=14, color='black'),
        align='center'
    )
    fig.add_annotation(
        text=subtitle,
        xref='paper', yref='paper',
        x=0.5, y=1.08, showarrow=False,
        font=dict(size=15, color='black'),
        align='center'
    )

    zero_bin_idx = None
    for i in range(len(bins)-1):
        if bins[i] <= 0 < bins[i+1]:
            zero_bin_idx = i
            break
    if zero_bin_idx is not None:
        fig.add_vline(
            x=zero_bin_idx - 0.5,
            line_dash="dash",
            line_color="black",
            line_width=2
        )
        fig.add_annotation(
            x=zero_bin_idx - 0.5,
            y=max(hist_percentages) * 0.5,
            text="손실/이익 경계",
            showarrow=False,
            font=dict(size=12, color='black'),
            textangle=-90,
            align='center',
            bgcolor="white",
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
    if '[' in user_input and ']' in user_input:
        code = user_input.split('(')[-1].split(')')[0].strip()
        name = user_input.split('(')[0].strip()
        return code, name
    if user_input in code_to_name:
        return user_input, code_to_name[user_input]
    elif user_input in name_to_code:
        return name_to_code[user_input], user_input
    elif '(' in user_input and ')' in user_input:
        name = user_input.split('(')[0].strip()
        code = user_input.split('(')[-1].replace(')','').strip()
        return code, name
    else:
        return user_input, user_input

st.title("📊 한국/해외 주식 연 수익률 분포 히스토그램")

st.markdown("""
- **KOSPI 1981~오늘까지 연 수익률 분포**  
- 10% 단위 구간, 손실=회색, 이익=파란색  
- 각 막대 위에 비율(%) 표시, 이익/손실확률, CAGR 표시  
- 아래에서 회사명/티커로 검색해 국내외 주식 동일 분석 가능
- **NEW!** 📈 코스피/S&P500 비교 차트 및 상관관계 분석
""")

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

st.header("🔍 다른 종목/지수 연 수익률 분포 보기")

if 'selected_market' not in st.session_state:
    st.session_state.selected_market = 'KRX'

if 'selected_company' not in st.session_state:
    default_company = "삼성전자 (005930) [KRX]" if "삼성전자 (005930) [KRX]" in company_options else company_options[0]
    st.session_state.selected_company = default_company

if 'text_input_value' not in st.session_state:
    st.session_state.text_input_value = "삼성전자"

if 'last_selectbox_value' not in st.session_state:
    st.session_state.last_selectbox_value = st.session_state.selected_company

if 'last_textinput_value' not in st.session_state:
    st.session_state.last_textinput_value = st.session_state.text_input_value

if 'auto_analyze' not in st.session_state:
    st.session_state.auto_analyze = False

market_list = sorted(all_stock_table['Market'].unique())
market_companies = {}

for market in market_list:
    market_data = all_stock_table[all_stock_table['Market'] == market]
    market_companies[market] = [f"{row.Name} ({row.Code})" for row in market_data.itertuples()]

market_icons = {
    'KRX': '🇰🇷',
    'NASDAQ': '🇺🇸',
    'NYSE': '🇺🇸', 
    'AMEX': '🇺🇸'
}

market_descriptions = {
    'KRX': '한국거래소 (Korean Exchange)',
    'NASDAQ': '나스닥 (National Association of Securities Dealers Automated Quotations)',
    'NYSE': '뉴욕증권거래소 (New York Stock Exchange)',
    'AMEX': '아메리칸증권거래소 (American Stock Exchange)'
}

st.subheader("1️⃣ 거래소/시장 선택")

col_market1, col_market2 = st.columns([1, 2])

with col_market1:
    selected_market = st.selectbox(
        "거래소를 선택하세요",
        market_list,
        index=market_list.index(st.session_state.selected_market) if st.session_state.selected_market in market_list else 0,
        format_func=lambda x: f"{market_icons.get(x, '🌍')} {x}",
        key="market_selectbox"
    )

with col_market2:
    if selected_market in market_descriptions:
        st.info(f"📍 **{market_descriptions[selected_market]}**")
        
        market_count = len(market_companies.get(selected_market, []))
        st.caption(f"📊 등록 종목 수: **{market_count:,}개**")

if selected_market != st.session_state.selected_market:
    st.session_state.selected_market = selected_market
    if selected_market in market_companies and market_companies[selected_market]:
        first_company = market_companies[selected_market][0]
        st.session_state.selected_company = f"{first_company} [{selected_market}]"
        st.session_state.last_selectbox_value = st.session_state.selected_company
        company_name = first_company.split(' (')[0]
        st.session_state.text_input_value = company_name
        st.session_state.last_textinput_value = company_name
    st.rerun()

st.subheader(f"2️⃣ {market_icons.get(selected_market, '🌍')} {selected_market} 종목 선택")

current_market_options = market_companies.get(selected_market, [])

if not current_market_options:
    st.warning(f"⚠️ {selected_market} 시장의 데이터를 불러올 수 없습니다.")
    st.stop()

current_company_without_market = st.session_state.selected_company.split(' [')[0] if ' [' in st.session_state.selected_company else st.session_state.selected_company

try:
    current_index = current_market_options.index(current_company_without_market)
except (ValueError, IndexError):
    current_index = 0
    if current_market_options:
        st.session_state.selected_company = f"{current_market_options[0]} [{selected_market}]"

selected = st.selectbox(
    f"회사명 또는 티커를 선택하세요 ({len(current_market_options):,}개 종목)",
    current_market_options,
    index=current_index,
    key="company_selectbox"
)

user_input = st.text_input(
    f"직접 입력 ({selected_market} 시장 내 검색)",
    value=st.session_state.text_input_value,
    key="company_textinput",
    help=f"{selected_market} 시장에서 회사명이나 티커로 검색하세요"
)

selected_with_market = f"{selected} [{selected_market}]"

if selected_with_market != st.session_state.last_selectbox_value:
    st.session_state.last_selectbox_value = selected_with_market
    st.session_state.selected_company = selected_with_market
    
    if '(' in selected and ')' in selected:
        company_name = selected.split('(')[0].strip()
        st.session_state.text_input_value = company_name
        st.session_state.last_textinput_value = company_name
        
        st.session_state.auto_analyze = True
        st.rerun()

if user_input != st.session_state.last_textinput_value:
    st.session_state.last_textinput_value = user_input
    st.session_state.text_input_value = user_input
    
    if user_input.strip():
        keyword = user_input.strip().lower()
        
        exact_matches = [opt for opt in current_market_options if keyword in opt.lower()]
        
        if exact_matches:
            best_match = None
            
            for opt in exact_matches:
                company_part = opt.split('(')[0].strip().lower()
                if company_part == keyword:
                    best_match = opt
                    break
            
            if not best_match:
                for opt in exact_matches:
                    if '(' in opt and ')' in opt:
                        ticker_part = opt.split('(')[1].split(')')[0].strip().lower()
                        if ticker_part == keyword:
                            best_match = opt
                            break
            
            if not best_match:
                best_match = exact_matches[0]
            
            best_match_with_market = f"{best_match} [{selected_market}]"
            if best_match_with_market != st.session_state.selected_company:
                st.session_state.selected_company = best_match_with_market
                st.session_state.last_selectbox_value = best_match_with_market
                st.rerun()

similar_options = []
if user_input.strip() and len(user_input.strip()) >= 2:
    keyword = user_input.strip().lower()
    similar_options = [opt for opt in current_market_options if keyword in opt.lower()]
    
    if similar_options and len(similar_options) > 1:
        st.markdown(f"**🔍 '{user_input}' 검색 결과 ({len(similar_options)}개) - {market_icons.get(selected_market, '🌍')} {selected_market}:**")
        
        display_options = similar_options[:10]
        
        for i, option in enumerate(display_options):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                if st.button("선택", key=f"select_btn_{i}"):
                    option_with_market = f"{option} [{selected_market}]"
                    st.session_state.selected_company = option_with_market
                    st.session_state.last_selectbox_value = option_with_market
                    company_name = option.split('(')[0].strip()
                    st.session_state.text_input_value = company_name
                    st.session_state.last_textinput_value = company_name
                    
                    st.session_state.auto_analyze = True
                    st.rerun()
            with col2:
                st.write(f"{market_icons.get(selected_market, '🌍')} {option}")

col_year1, col_year2 = st.columns(2)
with col_year1:
    start_year = st.number_input("시작 연도", min_value=1981, max_value=datetime.today().year-1, value=2000)
with col_year2:
    end_year = st.number_input("종료 연도", min_value=start_year+1, max_value=datetime.today().year, value=datetime.today().year)

auto_analyze_triggered = st.session_state.get('auto_analyze', False)
manual_analyze_clicked = st.button("📊 분석하기", type="primary")

if auto_analyze_triggered:
    st.session_state.auto_analyze = False

if auto_analyze_triggered or manual_analyze_clicked:
    current_selection = st.session_state.selected_company
    ticker, company_name = get_ticker_and_name(current_selection)
    
    if auto_analyze_triggered:
        st.success(f"🔄 자동 분석: **{company_name}** ({ticker}) 선택됨")
    else:
        st.info(f"🎯 분석 대상: **{company_name}** ({ticker})")
    
    with st.spinner('데이터를 불러오는 중입니다...'):
        yearly_data, returns = get_korean_stock_data(ticker, int(start_year), int(end_year))
    
    if returns is not None and not returns.empty:
        st.subheader("📈 연 수익률 분포")
        fig = plot_return_histogram(returns, '연간', company_name, bins, bin_labels, colors)
        st.plotly_chart(fig, use_container_width=True)
        
        up_years = returns[returns > 0].index.tolist()
        down_years = returns[returns <= 0].index.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 상승 연도 수", len(up_years))
            st.caption(f"상승 연도: {', '.join(map(str, up_years)) if up_years else '없음'}")
        with col2:
            st.metric("📉 하락 연도 수", len(down_years))
            st.caption(f"하락 연도: {', '.join(map(str, down_years)) if down_years else '없음'}")

        max_return = returns.max()
        min_return = returns.min()
        max_year = returns.idxmax() if not returns.empty else '-'
        min_year = returns.idxmin() if not returns.empty else '-'
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("🏆 최고 수익률", f"{max_return:.2f}%", delta=f"{max_year}년")
        with col4:
            st.metric("⚠️ 최저 수익률", f"{min_return:.2f}%", delta=f"{min_year}년")

        st.subheader("📊 연도별 종가 추이 (vs 시장 지수)")
        
        col_index1, col_index2 = st.columns([1, 2])
        
        with col_index1:
            if selected_market == 'KRX':
                default_index = 'KOSPI'
                available_indices = ['KOSPI', 'S&P500']
            else:
                default_index = 'S&P500'
                available_indices = ['S&P500', 'KOSPI']
            
            comparison_index = st.selectbox(
                "비교 지수 선택",
                available_indices,
                index=available_indices.index(default_index),
                format_func=lambda x: f"🇰🇷 {x}" if x == 'KOSPI' else f"🇺🇸 {x}",
                key="comparison_index_select"
            )
        
        with col_index2:
            if comparison_index == 'KOSPI':
                st.info("📈 **KOSPI** - 한국 종합주가지수 (Korea Composite Stock Price Index)")
                st.caption("🏢 한국거래소 상장 주요 기업들의 시가총액 가중평균 지수")
            else:
                st.info("📈 **S&P500** - 미국 스탠더드앤푸어스 500 지수")
                st.caption("🏢 미국 주요 500개 기업의 시가총액 가중평균 지수")
        
        index_tickers = {
            'KOSPI': 'KS11',
            'S&P500': 'SPY'
        }
        
        comparison_ticker = index_tickers[comparison_index]
        
        with st.spinner(f'{comparison_index} 비교 데이터를 불러오는 중입니다...'):
            comparison_yearly_data, _ = get_korean_stock_data(comparison_ticker, int(start_year), int(end_year))
        
        if comparison_yearly_data is not None and not comparison_yearly_data.empty:
            fig_combined = make_subplots(
                specs=[[{"secondary_y": True}]]
            )
            
            price_df = yearly_data.reset_index()
            price_df.columns = ['연도', '종가']
            
            if selected_market == 'KRX':
                price_unit = '원'
                price_format = ':,d'
            else:
                price_unit = '$'
                price_format = ':,.2f'
            
            fig_combined.add_trace(
                go.Bar(
                    x=price_df['연도'], 
                    y=price_df['종가'], 
                    name=f"{company_name}",
                    marker_color='rgba(68, 114, 196, 0.7)',
                    yaxis='y',
                    hovertemplate=f'<b>{company_name}</b><br>연도: %{{x}}<br>종가: %{{y{price_format}}}{price_unit}<extra></extra>'
                ),
                secondary_y=False
            )
            
            comparison_df = comparison_yearly_data.reset_index()
            comparison_df.columns = ['연도', comparison_index]
            
            index_colors = {
                'KOSPI': 'red',
                'S&P500': 'green'
            }
            
            index_color = index_colors.get(comparison_index, 'blue')
            
            fig_combined.add_trace(
                go.Scatter(
                    x=comparison_df['연도'], 
                    y=comparison_df[comparison_index],
                    mode='lines+markers',
                    name=comparison_index,
                    line=dict(color=index_color, width=3),
                    marker=dict(size=6, color=index_color),
                    yaxis='y2',
                    hovertemplate=f'<b>{comparison_index}</b><br>연도: %{{x}}<br>지수: %{{y:,.2f}}<extra></extra>'
                ),
                secondary_y=True
            )
            
            fig_combined.update_xaxes(title_text="연도")
            
            fig_combined.update_yaxes(
                title_text=f"{company_name} 주가 ({price_unit})", 
                secondary_y=False,
                title_font_color="blue",
                tickformat=',d' if selected_market == 'KRX' else ',.2f'
            )
            
            fig_combined.update_yaxes(
                title_text=f"{comparison_index} 지수", 
                secondary_y=True,
                title_font_color=index_color,
                tickformat=',.2f'
            )
            
            fig_combined.update_layout(
                title=f"📊 {company_name} vs {comparison_index} 연도별 추이 비교",
                template='plotly_white',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )
            
            st.plotly_chart(fig_combined, use_container_width=True)
            
            if len(yearly_data) == len(comparison_yearly_data):
                correlation = yearly_data.corr(comparison_yearly_data)
                
                col_corr1, col_corr2 = st.columns(2)
                with col_corr1:
                    st.metric(
                        f"🔗 {comparison_index}와의 상관관계", 
                        f"{correlation:.3f}",
                        help=f"1에 가까울수록 {comparison_index}와 동조화, -1에 가까울수록 반대 움직임"
                    )
                with col_corr2:
                    if correlation > 0.7:
                        corr_desc = f"높은 양의 상관관계 ({comparison_index}와 강하게 동조화) 📈🤝"
                    elif correlation > 0.3:
                        corr_desc = f"보통 양의 상관관계 ({comparison_index}와 어느 정도 동조화) 📈➡️"
                    elif correlation > -0.3:
                        corr_desc = "낮은 상관관계 (독립적인 움직임) 🔄"
                    elif correlation > -0.7:
                        corr_desc = f"보통 음의 상관관계 ({comparison_index}와 반대 경향) 📉⬅️"
                    else:
                        corr_desc = f"높은 음의 상관관계 ({comparison_index}와 강하게 반대) 📉🔄"
                    
                    st.info(f"💡 **해석**: {corr_desc}")
        else:
            price_df = yearly_data.reset_index()
            price_df.columns = ['연도', '종가']
            fig_price = go.Figure(go.Bar(x=price_df['연도'], y=price_df['종가'], marker_color='#4472C4'))
            
            if selected_market == 'KRX':
                currency_unit = '원'
            else:
                currency_unit = '
            
            fig_price.update_layout(
                title=f"{company_name} 연도별 종가 추이", 
                xaxis_title='연도', 
                yaxis_title=f'종가 ({currency_unit})',
                template='plotly_white'
            )
            st.plotly_chart(fig_price, use_container_width=True)
            st.warning(f"⚠️ {comparison_index} 비교 데이터를 불러올 수 없어 개별 차트만 표시됩니다.")
        
        with st.expander("📋 연도별 상세 데이터 보기"):
            detail_df = pd.DataFrame({
                '연도': yearly_data.index,
                '종가': yearly_data.values,
                '수익률(%)': ['-'] + [f"{x:.2f}%" for x in returns.values]
            })
            st.dataframe(detail_df, use_container_width=True)
            
            csv = detail_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV로 다운로드",
                data=csv,
                file_name=f"{company_name}_{start_year}-{end_year}_분석결과.csv",
                mime="text/csv"
            )
    else:
        st.error(f"❌ '{company_name} ({ticker})' 데이터를 불러올 수 없습니다.")
        st.info("💡 다른 종목을 선택해보시거나, 티커 심볼을 확인해주세요.")
        
        st.subheader("🎯 추천 종목")
        recommended = ["삼성전자 (005930) [KRX]", "SK하이닉스 (000660) [KRX]", "NAVER (035420) [KRX]", "카카오 (035720) [KRX]"]
        cols = st.columns(len(recommended))
        
        for i, rec in enumerate(recommended):
            with cols[i]:
                if st.button(rec.split(' (')[0], key=f"rec_{i}"):
                    st.session_state.selected_company = rec
                    st.session_state.last_selectbox_value = rec
                    company_name = rec.split('(')[0].strip()
                    st.session_state.text_input_value = company_name
                    st.session_state.last_textinput_value = company_name
                    st.rerun()

st.markdown("---")
st.markdown("""
### 📌 사용법 가이드
- **거래소 선택**: 먼저 원하는 시장(KRX, NYSE, NASDAQ, AMEX) 선택
- **종목 선택**: 해당 시장 내에서 회사 선택 → 자동 분석 실행
- **직접 입력**: 회사명이나 티커 입력 → 자동으로 해당 항목이 선택됨
- **비교 지수**: KOSPI 또는 S&P500과 비교 분석
- **검색 결과**: 여러 후보가 있을 때 "선택" 버튼으로 바로 선택 가능

### 🎯 주요 기능
- ✅ **시장별 분류**: 거래소별 체계적 종목 탐색
- ✅ **양방향 연동**: selectbox ↔ 직접입력 완전 동기화
- ✅ **수익률 분포**: 연도별 수익률을 구간별로 시각화
- ✅ **글로벌 비교**: KOSPI와 S&P500 중 선택하여 비교 분석
- ✅ **상관관계 분석**: 시장과의 동조화 정도 수치화
- ✅ **상세 데이터**: CSV 다운로드로 추가 분석 가능

### ⚡ 개선사항
- 🌍 **시장별 필터링**: 원하는 거래소 집중 탐색
- 🔄 **실시간 연동**: UI 요소간 즉시 반영
- 📊 **이중 축 차트**: 스케일이 다른 데이터 동시 표시
- 🎨 **개선된 시각화**: 손실/이익 경계선 최적화
- 📈 **글로벌 분석**: 한국/미국 지수 선택적 비교
""")
