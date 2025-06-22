# 📊 코스피 비교 차트 개선 설명 및 예제

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# =============================================================================
# 🔍 개선 내용 설명
# =============================================================================

"""
📈 주요 개선사항:

1. 이중 축(Secondary Y-axis) 사용
   - 왼쪽 축: 개별 주식 가격 (막대그래프)
   - 오른쪽 축: 코스피 지수 (선그래프)
   - 스케일이 다른 데이터를 한 차트에서 비교 가능

2. 시각적 구분
   - 개별 주식: 파란색 막대그래프 (반투명)
   - 코스피: 빨간색 선그래프 (굵은 선 + 마커)

3. 상관관계 분석 추가
   - 피어슨 상관계수 계산
   - 해석 가이드 제공

4. 사용자 경험 개선
   - 통합 호버 정보 (hovermode='x unified')
   - 범례를 상단에 가로로 배치
   - 로딩 스피너 추가
"""

# =============================================================================
# 🎯 실제 구현 예제
# =============================================================================

def create_comparison_chart_example():
    """코스피 비교 차트 생성 예제"""
    
    # 샘플 데이터 생성 (실제로는 FinanceDataReader에서 가져옴)
    years = list(range(2020, 2025))
    
    # 삼성전자 주가 (예시)
    samsung_prices = [58000, 82000, 59000, 71300, 75000]
    
    # 코스피 지수 (예시)
    kospi_values = [2200, 3000, 2400, 2360, 2500]
    
    # 이중 축 차트 생성
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=["삼성전자 vs 코스피 비교"]
    )
    
    # 1. 삼성전자 주가 (막대그래프, 왼쪽 축)
    fig.add_trace(
        go.Bar(
            x=years,
            y=samsung_prices,
            name="삼성전자",
            marker_color='rgba(68, 114, 196, 0.7)',  # 반투명 파란색
            yaxis='y',
            hovertemplate='<b>삼성전자</b><br>연도: %{x}<br>주가: %{y:,}원<extra></extra>'
        ),
        secondary_y=False  # 왼쪽 축 사용
    )
    
    # 2. 코스피 지수 (선그래프, 오른쪽 축)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=kospi_values,
            mode='lines+markers',
            name='KOSPI',
            line=dict(color='red', width=3),
            marker=dict(size=8, color='red', symbol='circle'),
            yaxis='y2',
            hovertemplate='<b>KOSPI</b><br>연도: %{x}<br>지수: %{y:,}<extra></extra>'
        ),
        secondary_y=True  # 오른쪽 축 사용
    )
    
    # 축 설정
    fig.update_xaxes(title_text="연도")
    fig.update_yaxes(
        title_text="삼성전자 주가 (원)", 
        secondary_y=False,
        title_font_color="blue",
        tickformat=',d'  # 천단위 구분자
    )
    fig.update_yaxes(
        title_text="KOSPI 지수", 
        secondary_y=True,
        title_font_color="red",
        tickformat=',d'
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title={
            'text': "📊 삼성전자 vs KOSPI 연도별 추이 비교",
            'x': 0.5,
            'font': {'size': 16}
        },
        template='plotly_white',
        hovermode='x unified',  # 통합 호버
        legend=dict(
            orientation="h",  # 가로 배치
            yanchor="bottom",
            y=1.02,
            xanchor="right", 
            x=1
        ),
        height=500
    )
    
    return fig

# =============================================================================
# 📈 상관관계 분석 함수
# =============================================================================

def analyze_correlation(stock_data, kospi_data):
    """주식과 코스피의 상관관계 분석"""
    
    # 데이터 길이 맞추기
    min_length = min(len(stock_data), len(kospi_data))
    stock_aligned = stock_data[-min_length:]
    kospi_aligned = kospi_data[-min_length:]
    
    # 상관계수 계산
    correlation = np.corrcoef(stock_aligned, kospi_aligned)[0, 1]
    
    # 해석
    if correlation > 0.7:
        interpretation = "높은 양의 상관관계 - 시장과 강하게 동조화"
        emoji = "📈🤝"
    elif correlation > 0.3:
        interpretation = "보통 양의 상관관계 - 시장과 어느 정도 동조화"
        emoji = "📈➡️"
    elif correlation > -0.3:
        interpretation = "낮은 상관관계 - 독립적인 움직임"
        emoji = "🔄"
    elif correlation > -0.7:
        interpretation = "보통 음의 상관관계 - 시장과 반대 경향"
        emoji = "📉⬅️"
    else:
        interpretation = "높은 음의 상관관계 - 시장과 강하게 반대"
        emoji = "📉🔄"
    
    return correlation, interpretation, emoji

# =============================================================================
# 🎨 차트 스타일링 옵션들
# =============================================================================

def create_styled_comparison_chart():
    """다양한 스타일링 옵션을 보여주는 예제"""
    
    # 데이터
    years = list(range(2020, 2025))
    stock_prices = [100, 120, 90, 110, 130]
    market_index = [2000, 2200, 1800, 2100, 2300]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 스타일 옵션 1: 그라데이션 막대
    fig.add_trace(
        go.Bar(
            x=years,
            y=stock_prices,
            name="개별 주식",
            marker=dict(
                color=stock_prices,
                colorscale='Blues',
                opacity=0.8,
                line=dict(color='darkblue', width=1)
            )
        ),
        secondary_y=False
    )
    
    # 스타일 옵션 2: 점선 + 마커 변경
    fig.add_trace(
        go.Scatter(
            x=years,
            y=market_index,
            mode='lines+markers',
            name='시장 지수',
            line=dict(
                color='crimson', 
                width=3, 
                dash='dot'  # 점선
            ),
            marker=dict(
                size=10, 
                color='white',
                line=dict(color='crimson', width=2),
                symbol='diamond'  # 다이아몬드 모양
            )
        ),
        secondary_y=True
    )
    
    # 배경 그리드 스타일링
    fig.update_layout(
        plot_bgcolor='rgba(240,240,240,0.3)',
        paper_bgcolor='white',
        title="🎨 스타일링된 비교 차트",
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    # 격자 스타일
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightblue',
        secondary_y=False
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightcoral',
        secondary_y=True
    )
    
    return fig

# =============================================================================
# 🔧 사용법 가이드
# =============================================================================

"""
💡 사용법 가이드:

1. 이중 축 차트 생성:
   from plotly.subplots import make_subplots
   fig = make_subplots(specs=[[{"secondary_y": True}]])

2. 데이터 추가:
   fig.add_trace(막대그래프_데이터, secondary_y=False)  # 왼쪽 축
   fig.add_trace(선그래프_데이터, secondary_y=True)     # 오른쪽 축

3. 축 설정:
   fig.update_yaxes(title_text="왼쪽 축 제목", secondary_y=False)
   fig.update_yaxes(title_text="오른쪽 축 제목", secondary_y=True)

4. 상관관계 분석:
   correlation = numpy.corrcoef(data1, data2)[0, 1]

🎯 핵심 포인트:
- secondary_y=True/False로 축 구분
- hovermode='x unified'로 통합 호버
- 색상으로 데이터 구분 (파란색=개별주식, 빨간색=시장지수)
- 상관계수로 관계 정도 파악
"""

# 실행 예제
if __name__ == "__main__":
    import streamlit as st
    
    st.title("📊 코스피 비교 차트 개선")
    
    st.header("🔄 개선된 비교 차트")
    st.plotly_chart(create_comparison_chart_example())
    
    st.header("🎨 스타일링 옵션")
    st.plotly_chart(create_styled_comparison_chart())
    
    st.header("📈 상관관계 분석 예제")
    sample_stock = [100, 120, 90, 110, 130]
    sample_kospi = [2000, 2200, 1800, 2100, 2300]
    
    corr, interp, emoji = analyze_correlation(sample_stock, sample_kospi)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("상관계수", f"{corr:.3f}")
    with col2:
        st.info(f"{emoji} {interp}")
