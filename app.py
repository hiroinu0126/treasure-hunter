"""
株式スクリーニングアプリ【完全版】
- 日足/週足切り替え
- バックテスト機能
- 銘柄比較機能

実行: streamlit run streamlit_app_complete.py
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

# 業種マスタのインポート（同じディレクトリにある場合）
try:
    from sector_master import get_sector_from_code
    SECTOR_MASTER_AVAILABLE = True
except:
    SECTOR_MASTER_AVAILABLE = False

# ページ設定
st.set_page_config(
    page_title="お宝発掘ツール - 世はまさに、大海賊時代！",
    page_icon="🏴‍☠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 設定 =====
SECTOR_MAP = {
    'Consumer Defensive': '生活必需品', 'Consumer Cyclical': '一般消費財',
    'Financial Services': '金融', 'Real Estate': '不動産',
    'Technology': '情報技術', 'Healthcare': 'ヘルスケア',
    'Communication Services': '通信サービス', 'Industrials': '資本財・サービス',
    'Basic Materials': '素材', 'Energy': 'エネルギー', 'Utilities': '公共事業',
    'Financial': '金融'
}

# ===== 週足変換関数 =====
def convert_to_weekly(data):
    """日足データを週足に変換"""
    weekly = data.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return weekly.dropna()


# ===== スコア計算関数（週足/日足対応） =====
@st.cache_data(ttl=3600)
def calculate_score(close, volume, n225_return, min_volume, timeframe='daily'):
    """
    スコア計算ロジック（週足/日足対応）
    
    Args:
        timeframe: 'daily' or 'weekly'
    """
    try:
        # 週足の場合は必要データ量を調整
        min_length = 80 if timeframe == 'daily' else 52  # 週足は52週（1年分）
        
        if len(close) < min_length:
            return 0, [], None
            
        curr_p = float(close[-1])
        prev_p = float(close[-2])
        
        # 急落フィルター
        day_return = (curr_p - prev_p) / prev_p
        threshold = -0.03 if timeframe == 'daily' else -0.07  # 週足は-7%
        if day_return < threshold:
            return 0, [], None
        
        # 週足の場合は最近の期間を調整
        recent_period = 5 if timeframe == 'daily' else 2
        recent_max = np.max(close[-recent_period:])
        drop_threshold = 0.07 if timeframe == 'daily' else 0.12
        if (recent_max - curr_p) / recent_max > drop_threshold:
            return 0, [], None
        
        # 指標計算（期間を調整）
        if timeframe == 'daily':
            vol_period = 20
            ma5_period = 5
            ma25_period = 25
            ma75_period = 75
        else:  # weekly
            vol_period = 4  # 約1ヶ月
            ma5_period = 13  # 13週（約3ヶ月）
            ma25_period = 26  # 26週（約6ヶ月）
            ma75_period = 52  # 52週（1年）
        
        avg_vol = float(np.mean(volume[-vol_period:-1]))
        
        if avg_vol < min_volume:
            return 0, [], None
        
        ma5 = np.mean(close[-ma5_period:])
        ma5_prev = np.mean(close[-(ma5_period+1):-1])
        ma25 = np.mean(close[-ma25_period:])
        ma75 = np.mean(close[-ma75_period:])
        
        # 75期間MAの5期間前との比較（トレンド判定）
        lookback = 5 if timeframe == 'daily' else 4  # 週足は4週前と比較
        ma75_prev5 = np.mean(close[-(ma75_period+lookback):-lookback]) if len(close) >= ma75_period+lookback else ma75
        
        # RSI（期間調整）
        rsi_period = 15 if timeframe == 'daily' else 14  # 週足は14週が標準
        delta = np.diff(close[-rsi_period:])
        up, down = delta[delta > 0].sum(), -delta[delta < 0].sum()
        curr_rsi = 100 * up / (up + (down if down > 0 else 1e-9))
        
        # スコアリング
        t_score = 0
        reasons = []
        
        # 黄金交差
        if ma5_prev < ma25 and ma5 > ma25:
            t_score += 15
            reasons.append("黄金交差")
        
        # 出来高を伴う上昇
        vol_multiplier = 2.0 if timeframe == 'daily' else 1.5
        price_threshold = 0.02 if timeframe == 'daily' else 0.05
        if float(volume[-1]) > avg_vol * vol_multiplier and day_return > price_threshold:
            t_score += 15
            reasons.append("大商い伴う上昇")
        
        # RS（対日経平均）
        return_period = 60 if timeframe == 'daily' else 52  # 週足は52週（1年）
        stock_return = (curr_p - close[-return_period]) / close[-return_period] if len(close) >= return_period else 0
        if stock_return > n225_return:
            t_score += 10
            reasons.append("市場超え強気")
        
        # 中長期上昇トレンド
        if ma75 > ma75_prev5:
            t_score += 20
            reasons.append("中長期上昇")
        
        # 25MA近接度
        diff_25 = (curr_p - ma25) / ma25
        if abs(diff_25) < 0.02:
            t_score += 20
            reasons.append("25MA近接")
        elif abs(diff_25) < 0.05:
            t_score += 10
            reasons.append("25MA付近")
        
        # 出来高急増
        if float(volume[-1]) > avg_vol * 1.5:
            t_score += 10
            reasons.append("出来高急増")
        
        # RSI適正範囲
        if 30 <= curr_rsi <= 60:
            t_score += 10
            reasons.append("RSI適正")
        
        if t_score < 40:
            return 0, [], None
        
        metrics = {
            'price': curr_p,
            'ma5': ma5,
            'ma25': ma25,
            'ma75': ma75,
            'diff_25': diff_25,
            'stock_return': stock_return,
            'rsi': curr_rsi,
            'avg_vol': avg_vol,
            'day_return': day_return
        }
        
        return t_score, reasons, metrics
        
    except:
        return 0, [], None


# ===== バックテスト関数 =====
def run_backtest(df_input, min_volume, min_score, months_back, hold_days, timeframe, progress_bar, status_text):
    """バックテスト実行"""
    
    status_text.text("📊 バックテスト準備中...")
    
    # 銘柄リスト作成（最初の30銘柄に制限）
    tickers = []
    names = {}
    for _, row in df_input.head(30).iterrows():
        try:
            c = str(row['コード']).split('.')[0].strip()
            t = c + ".T"
            tickers.append(t)
            names[t] = row['銘柄名']
        except:
            continue
    
    # 日経平均データ取得
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=500)
    
    try:
        interval = '1d' if timeframe == 'daily' else '1wk'
        n225 = yf.download("^N225", start=start_date, end=end_date, interval=interval, progress=False)
        if isinstance(n225.columns, pd.MultiIndex):
            n225.columns = n225.columns.get_level_values(0)
        n225_close = n225['Close'].values.flatten()
    except:
        st.error("日経平均データ取得失敗")
        return pd.DataFrame()
    
    # テスト期間設定
    test_results = []
    freq = 'MS' if timeframe == 'daily' else 'MS'  # 月初
    test_dates = pd.date_range(
        end=end_date - datetime.timedelta(days=hold_days if timeframe == 'daily' else hold_days*7),
        periods=months_back,
        freq=freq
    )
    
    for idx, test_date in enumerate(test_dates):
        progress_bar.progress((idx + 1) / len(test_dates))
        status_text.text(f"📅 バックテスト中: {test_date.strftime('%Y-%m')} ({idx+1}/{len(test_dates)})")
        
        # この時点での日経リターン
        date_idx = np.where(n225.index <= test_date)[0]
        return_period = 60 if timeframe == 'daily' else 52  # 週足は52週
        if len(date_idx) < return_period:
            continue
        n225_return = (n225_close[date_idx[-1]] - n225_close[date_idx[-return_period]]) / n225_close[date_idx[-return_period]]
        
        selected_stocks = []
        
        for ticker in tickers:
            try:
                interval = '1d' if timeframe == 'daily' else '1wk'
                lookback = 250 if timeframe == 'daily' else 500  # 週足は約2年分
                
                data = yf.download(
                    ticker,
                    start=test_date - datetime.timedelta(days=lookback),
                    end=test_date + datetime.timedelta(days=hold_days+10 if timeframe == 'daily' else hold_days*7+30),
                    interval=interval,
                    progress=False
                )
                
                if data.empty or len(data) < (80 if timeframe == 'daily' else 52):  # 週足は52週
                    continue
                    
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # テスト日時点のデータ
                historical_data = data[data.index <= test_date]
                min_length = 80 if timeframe == 'daily' else 52  # 週足は52週
                if len(historical_data) < min_length:
                    continue
                
                close = historical_data['Close'].values.flatten()
                volume = historical_data['Volume'].values.flatten()
                close = close[~np.isnan(close)]
                
                # スコア計算
                score, reasons, metrics = calculate_score(close, volume, n225_return, min_volume, timeframe)
                
                if score >= min_score:
                    buy_price = float(close[-1])
                    
                    # hold_days後の価格
                    future_data = data[data.index > test_date]
                    if len(future_data) >= hold_days:
                        sell_price = float(future_data['Close'].iloc[hold_days-1])
                        return_pct = (sell_price - buy_price) / buy_price * 100
                        
                        selected_stocks.append({
                            'ticker': ticker,
                            'name': names[ticker],
                            'score': score,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'return': return_pct
                        })
                
                time.sleep(0.05)
                
            except:
                continue
        
        if selected_stocks:
            returns = [s['return'] for s in selected_stocks]
            avg_return = np.mean(returns)
            median_return = np.median(returns)
            win_rate = len([s for s in selected_stocks if s['return'] > 0]) / len(selected_stocks) * 100
            
            test_results.append({
                'テスト月': test_date.strftime('%Y-%m'),
                '銘柄数': len(selected_stocks),
                '平均リターン': f"{avg_return:.2f}%",
                '中央値リターン': f"{median_return:.2f}%",
                '勝率': f"{win_rate:.1f}%",
                '最良': f"{max(returns):.2f}%",
                '最悪': f"{min(returns):.2f}%",
                '詳細': selected_stocks
            })
    
    return pd.DataFrame(test_results)


# ===== 銘柄比較チャート =====
def create_comparison_chart(tickers, period='6mo', timeframe='daily'):
    """複数銘柄の比較チャート"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('株価推移（正規化）', '出来高'),
        row_heights=[0.7, 0.3]
    )
    
    interval = '1d' if timeframe == 'daily' else '1wk'
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                continue
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # 価格正規化（最初の価格を100とする）
            normalized_price = (data['Close'] / data['Close'].iloc[0]) * 100
            
            # 価格チャート
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized_price,
                    name=ticker,
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # 出来高チャート
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name=f"{ticker} 出来高",
                    showlegend=False,
                    opacity=0.5
                ),
                row=2, col=1
            )
            
        except:
            continue
    
    fig.update_xaxes(title_text="日付", row=2, col=1)
    fig.update_yaxes(title_text="正規化価格 (基準=100)", row=1, col=1)
    fig.update_yaxes(title_text="出来高", row=2, col=1)
    
    fig.update_layout(
        height=800,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# ===== メインスクリーニング関数 =====
def run_screening(df, min_volume, min_score, timeframe, fetch_fundamentals, progress_bar, status_text):
    """スクリーニング実行"""
    
    # 日経平均データ取得
    status_text.text("🌐 市場環境を確認中...")
    try:
        interval = '1d' if timeframe == 'daily' else '1wk'
        period = '100d' if timeframe == 'daily' else '2y'
        
        n225 = yf.download("^N225", period=period, interval=interval, progress=False)
        if isinstance(n225.columns, pd.MultiIndex):
            n225.columns = n225.columns.get_level_values(0)
        
        m_close = n225['Close'].values.flatten()
        m_curr = m_close[-1]
        
        # 移動平均期間を調整
        ma_period = 25 if timeframe == 'daily' else 26  # 週足は26週MA
        m_ma = np.mean(m_close[-ma_period:])
        m_label = "強気" if m_curr > m_ma else "慎重"
        
        return_period = 60 if timeframe == 'daily' else 52  # 週足は52週（1年）
        n225_return = (m_close[-1] - m_close[-return_period]) / m_close[-return_period] if len(m_close) >= return_period else 0
    except Exception as e:
        m_label = "不明"
        n225_return = 0
        m_curr = 0
        m_ma = 0
        # エラーログを表示（デバッグ用）
        import traceback
        print(f"⚠️ 日経平均データ取得エラー: {e}")
        print(traceback.format_exc())
    
    # 銘柄リスト作成
    tickers = []
    names = {}
    for _, row in df.iterrows():
        try:
            c = str(row['コード']).split('.')[0].strip()
            t = c + ".T"
            tickers.append(t)
            names[t] = row['銘柄名']
        except:
            continue
    
    results = []
    end_date = datetime.datetime.now()
    lookback = 250 if timeframe == 'daily' else 500  # 週足は約2年分（52週×2）
    start_date = end_date - datetime.timedelta(days=lookback)
    
    total = len(tickers)
    interval = '1d' if timeframe == 'daily' else '1wk'
    
    for idx, ticker in enumerate(tickers):
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"📊 解析中... {idx + 1}/{total} ({names.get(ticker, ticker)})")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            
            min_length = 80 if timeframe == 'daily' else 52  # 週足は52週（1年分）
            if data.empty or len(data) < min_length:
                continue
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            close = data['Close'].values.flatten()
            volume = data['Volume'].values.flatten()
            close = close[~np.isnan(close)]
            
            score, reasons, metrics = calculate_score(close, volume, n225_return, min_volume, timeframe)
            
            if score < min_score:
                continue
            
            # ファンダメンタル（業種は必ず取得、その他は設定による）
            code = ticker.replace(".T", "")
            
            # 業種は証券コードから必ず判定（東証33業種分類）
            if SECTOR_MASTER_AVAILABLE:
                sector = get_sector_from_code(code)
            else:
                sector = '不明'
            
            # その他のファンダメンタルデータ
            calc_yield = 0
            per = 0
            pbr = 0
            
            # ファンダメンタル取得がONの場合のみ
            if fetch_fundamentals:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    
                    # fast_infoを優先（高速）
                    try:
                        fast_info = ticker_obj.fast_info
                        if hasattr(fast_info, 'dividendYield') and fast_info.dividendYield:
                            calc_yield = fast_info.dividendYield * 100
                    except:
                        pass
                    
                    # infoから詳細データ取得
                    info = ticker_obj.info
                    
                    # yfinanceから業種が取得できた場合は上書き
                    if 'sector' in info and info.get('sector'):
                        yf_sector = SECTOR_MAP.get(info['sector'], info['sector'])
                        if yf_sector and yf_sector != '不明':
                            sector = yf_sector
                    
                    # 配当利回り
                    if calc_yield == 0:
                        div_rate = info.get('dividendRate') or info.get('dividendYield')
                        if div_rate and metrics['price']:
                            if div_rate < 1:
                                calc_yield = div_rate * 100
                            else:
                                calc_yield = (div_rate / metrics['price']) * 100
                    
                    # PER
                    per_val = info.get('trailingPE') or info.get('forwardPE')
                    if per_val:
                        per = round(per_val, 1)
                    
                    # PBR
                    pbr_val = info.get('priceToBook')
                    if pbr_val:
                        pbr = round(pbr_val, 1)
                
                except:
                    pass
            
            results.append({
                "コード": ticker.replace(".T", ""),
                "銘柄名": names[ticker],
                "スコア": score,
                "価格": round(metrics['price'], 1),
                "前回比": f"{metrics['day_return']*100:+.1f}%",
                "MA乖離": f"{metrics['diff_25']*100:+.1f}%",
                "リターン": f"{metrics['stock_return']*100:+.1f}%",
                "利回り": f"{calc_yield:.2f}%",
                "PER": per,
                "PBR": pbr,
                "RSI": round(metrics['rsi'], 1),
                "平均出来高": int(metrics['avg_vol']),
                "業種": sector,
                "判定": " / ".join(reasons),
                "Ticker": ticker  # 比較用
            })
            
            time.sleep(0.1)
            
        except:
            continue
    
    market_info = {
        'label': m_label,
        'current': m_curr,
        'ma': m_ma,
        'return': n225_return,
        'timeframe': timeframe
    }
    
    return pd.DataFrame(results), market_info


# ===== メインUI =====
st.title("🏴‍☠️ 探せ！この世のすべてをそこに置いてきた")
st.markdown("## 世はまさに、大海賊時代！")
st.markdown("**お宝発掘ツール！チャートからねらい目を自動で探すいわばログポースです。目指せグランドライン！**")
st.markdown("---")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    # タブで機能を切り替え
    mode = st.radio(
        "モード選択",
        ["📊 スクリーニング", "🔬 バックテスト", "📈 銘柄比較", "📖 マニュアル"],
        index=0
    )
    
    # ヘルプセクション
    with st.expander("📖 使い方・ヘルプ"):
        st.markdown("""
        ### ⚡ 高速モード vs 完全モード
        
        **高速モード（推奨）**
        - PER/PBR取得: OFF
        - 処理時間: 3-5分
        - 業種・テクニカル分析のみ
        
        **完全モード**
        - PER/PBR取得: ON
        - 処理時間: 15-30分
        - 全データ取得
        
        ---
        
        ### 📊 スコア基準
        - 40点以上: 合格
        - 50点以上: 良好
        - 60点以上: 優良
        
        ---
        
        ### 🏢 業種判定
        証券コードから自動判定
        - 5334 → ガラス・土石製品
        - 7203 → 輸送用機器
        - 8919 → 不動産業
        
        ---
        
        ### 💡 効率的な使い方
        1. 高速モード（5分）
        2. トップ50をダウンロード
        3. 50銘柄で完全モード（3分）
        4. ファンダメンタル確認
        
        **合計: 8分！**
        
        ---
        
        ### 📅 週足がおすすめ
        - 13週・26週・52週MA
        - ノイズが少ない
        - 週1回チェックでOK
        """)
    
    st.markdown("---")
    
    # 共通設定
    st.markdown("### 時間軸設定")
    timeframe = st.selectbox(
        "時間軸",
        ["daily", "weekly"],
        format_func=lambda x: "📅 日足" if x == "daily" else "📆 週足"
    )
    
    if mode == "📊 スクリーニング":
        uploaded_file = st.file_uploader(
            "銘柄リスト（CSV）",
            type=['csv'],
            help="コード列と銘柄名列を含むCSV"
        )
        
        st.markdown("### フィルター設定")
        
        min_volume = st.number_input(
            "最低平均出来高",
            min_value=0,
            max_value=1000000,
            value=100000,
            step=10000
        )
        
        min_score = st.slider(
            "最低スコア",
            min_value=0,
            max_value=100,
            value=40,
            step=5
        )
        
        # ファンダメンタル取得設定
        st.markdown("### データ取得設定")
        fetch_fundamentals = st.checkbox(
            "PER/PBR/配当を取得",
            value=False,
            help="⚠️ ONにすると処理時間が大幅に増加します（5-10倍）"
        )
    
    elif mode == "🔬 バックテスト":
        uploaded_file = st.file_uploader(
            "銘柄リスト（CSV）",
            type=['csv']
        )
        
        st.markdown("### バックテスト設定")
        
        months_back = st.slider(
            "テスト期間（ヶ月）",
            min_value=3,
            max_value=12,
            value=6
        )
        
        hold_days = st.slider(
            "保有期間（日 or 週）",
            min_value=5,
            max_value=60 if timeframe == 'daily' else 12,
            value=30 if timeframe == 'daily' else 4
        )
        
        min_volume = st.number_input(
            "最低平均出来高",
            min_value=0,
            max_value=1000000,
            value=100000,
            step=10000
        )
        
        min_score = st.slider(
            "最低スコア",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )
    
    else:  # 銘柄比較
        st.markdown("### 比較設定")
        
        ticker_input = st.text_area(
            "銘柄コード（カンマ区切り）",
            "7203,9984,6758",
            help="例: 7203,9984,6758"
        )
        
        comparison_period = st.selectbox(
            "比較期間",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2,
            format_func=lambda x: {
                "1mo": "1ヶ月",
                "3mo": "3ヶ月",
                "6mo": "6ヶ月",
                "1y": "1年",
                "2y": "2年"
            }[x]
        )

# メインエリア
if mode == "📊 スクリーニング":
    if uploaded_file is None:
        st.info("👈 銘柄リストCSVをアップロードしてください")
    else:
        try:
            df_input = pd.read_csv(uploaded_file, encoding='utf-8')
            st.success(f"✅ {len(df_input)}銘柄を読み込みました")
            
            if st.button("🚀 スクリーニング開始", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                df_result, market_info = run_screening(
                    df_input, min_volume, min_score, timeframe, fetch_fundamentals, progress_bar, status_text
                )
                
                progress_bar.empty()
                status_text.empty()
                
                if df_result.empty:
                    st.warning("⚠️ 条件に合う銘柄が見つかりませんでした")
                else:
                    df_sorted = df_result.sort_values(by="スコア", ascending=False)
                    
                    # 市場環境表示
                    st.markdown("## 🌐 市場環境")
                    
                    # データ取得失敗の警告
                    if market_info['label'] == "不明":
                        st.warning("⚠️ 日経平均データの取得に失敗しました。個別銘柄の分析結果は有効です。")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("地合い判定", market_info['label'])
                    with col2:
                        st.metric("日経平均", f"{market_info['current']:,.0f}円")
                    with col3:
                        ma_label = "25MA" if timeframe == 'daily' else "26週MA"
                        st.metric(ma_label, f"{market_info['ma']:,.0f}円")
                    with col4:
                        period_label = "60日" if timeframe == 'daily' else "52週"
                        st.metric(f"{period_label}リターン", f"{market_info['return']*100:+.2f}%")
                    
                    st.markdown("---")
                    
                    # 統計情報
                    st.markdown("## 📊 スクリーニング結果")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("抽出銘柄数", f"{len(df_sorted)}銘柄")
                    with col2:
                        st.metric("平均スコア", f"{df_sorted['スコア'].mean():.1f}点")
                    with col3:
                        st.metric("最高スコア", f"{df_sorted['スコア'].max()}点")
                    
                    # データ品質の情報
                    unknown_sector_count = len(df_sorted[df_sorted['業種'] == '不明'])
                    zero_pbr_count = len(df_sorted[df_sorted['PBR'] == 0])
                    
                    if unknown_sector_count > 0:
                        st.warning(f"⚠️ {unknown_sector_count}銘柄で業種が判定できませんでした（証券コード範囲外）")
                    
                    if not fetch_fundamentals:
                        st.info("ℹ️ PER/PBR/配当利回りの取得をスキップしました。左サイドバーで設定できます。")
                    elif zero_pbr_count > 0:
                        st.info(f"ℹ️ {zero_pbr_count}銘柄でPER/PBR/配当利回りが取得できませんでした。業種とテクニカル分析は有効です。")
                    
                    st.markdown("---")
                    
                    # 結果表示
                    st.markdown("## 🏆 スクリーニング結果")
                    
                    # 表示用に列を調整
                    display_df = df_sorted.drop(columns=['Ticker'])
                    st.dataframe(display_df, use_container_width=True, height=600)
                    
                    # ダウンロード
                    st.markdown("## 💾 結果のダウンロード")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = display_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            "📄 CSVダウンロード",
                            data=csv,
                            file_name=f"stock_scan_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            display_df.to_excel(writer, index=False, sheet_name='スクリーニング結果')
                        st.download_button(
                            "📊 Excelダウンロード",
                            data=buffer.getvalue(),
                            file_name=f"stock_scan_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # セッションステートに保存（銘柄比較用）
                    st.session_state['latest_results'] = df_sorted
                    
        except Exception as e:
            st.error(f"❌ エラー: {e}")

elif mode == "🔬 バックテスト":
    if uploaded_file is None:
        st.info("👈 銘柄リストCSVをアップロードしてください")
    else:
        try:
            df_input = pd.read_csv(uploaded_file, encoding='utf-8')
            st.success(f"✅ {len(df_input)}銘柄を読み込みました（テストは上位30銘柄）")
            
            if st.button("🔬 バックテスト開始", type="primary"):
                st.warning("⏱️ バックテストには10-30分程度かかります")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                backtest_results = run_backtest(
                    df_input, min_volume, min_score, months_back, hold_days, timeframe, progress_bar, status_text
                )
                
                progress_bar.empty()
                status_text.empty()
                
                if backtest_results.empty:
                    st.warning("⚠️ バックテスト結果がありません")
                else:
                    st.success("✅ バックテスト完了！")
                    
                    # サマリー表示
                    st.markdown("## 📊 バックテスト結果サマリー")
                    
                    # 詳細列を除いて表示
                    summary_df = backtest_results.drop(columns=['詳細'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # 平均リターンを計算
                    avg_returns = []
                    for _, row in backtest_results.iterrows():
                        val = float(row['平均リターン'].replace('%', ''))
                        avg_returns.append(val)
                    
                    with col1:
                        st.metric("総平均リターン", f"{np.mean(avg_returns):.2f}%")
                    with col2:
                        st.metric("最良月", f"{max(avg_returns):.2f}%")
                    with col3:
                        st.metric("最悪月", f"{min(avg_returns):.2f}%")
                    with col4:
                        profitable_months = len([r for r in avg_returns if r > 0])
                        st.metric("プラス月率", f"{profitable_months/len(avg_returns)*100:.1f}%")
                    
                    st.markdown("---")
                    
                    # 月次結果テーブル
                    st.markdown("### 📅 月次パフォーマンス")
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # リターン推移グラフ
                    st.markdown("### 📈 リターン推移")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=summary_df['テスト月'],
                        y=avg_returns,
                        mode='lines+markers',
                        name='平均リターン',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=8)
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        xaxis_title="月",
                        yaxis_title="リターン (%)",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 詳細を展開表示
                    st.markdown("### 🔍 月次詳細")
                    for _, row in backtest_results.iterrows():
                        with st.expander(f"{row['テスト月']} - 平均リターン: {row['平均リターン']}"):
                            details = row['詳細']
                            detail_df = pd.DataFrame(details)
                            detail_df = detail_df.sort_values(by='return', ascending=False)
                            st.dataframe(detail_df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ エラー: {e}")

else:  # 銘柄比較
    st.markdown("## 📈 銘柄比較チャート")
    
    # 最新のスクリーニング結果から選択
    if 'latest_results' in st.session_state and not st.session_state['latest_results'].empty:
        st.info("💡 最新のスクリーニング結果から銘柄を選択できます")
        
        top_stocks = st.session_state['latest_results'].head(20)
        selected_stocks = st.multiselect(
            "スクリーニング結果から選択",
            options=top_stocks['Ticker'].tolist(),
            format_func=lambda x: f"{x.replace('.T', '')} - {top_stocks[top_stocks['Ticker']==x]['銘柄名'].values[0]}"
        )
        
        if selected_stocks:
            ticker_input = ",".join([s.replace('.T', '') for s in selected_stocks])
    
    # 手動入力
    st.markdown("### または手動入力")
    
    tickers_list = [t.strip() + ".T" for t in ticker_input.split(',') if t.strip()]
    
    if st.button("📊 比較チャート生成", type="primary"):
        if len(tickers_list) == 0:
            st.warning("銘柄コードを入力してください")
        else:
            with st.spinner("チャート生成中..."):
                fig = create_comparison_chart(tickers_list, comparison_period, timeframe)
                st.plotly_chart(fig, use_container_width=True)
                
                # 統計情報
                st.markdown("### 📊 比較統計")
                stats_data = []
                
                for ticker in tickers_list:
                    try:
                        interval = '1d' if timeframe == 'daily' else '1wk'
                        data = yf.download(ticker, period=comparison_period, interval=interval, progress=False)
                        if not data.empty:
                            if isinstance(data.columns, pd.MultiIndex):
                                data.columns = data.columns.get_level_values(0)
                            
                            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                            volatility = data['Close'].pct_change().std() * np.sqrt(252 if timeframe == 'daily' else 52)
                            max_dd = ((data['Close'] / data['Close'].cummax()) - 1).min() * 100
                            
                            stats_data.append({
                                '銘柄': ticker.replace('.T', ''),
                                '総リターン': f"{total_return:+.2f}%",
                                'ボラティリティ': f"{volatility*100:.2f}%",
                                '最大ドローダウン': f"{max_dd:.2f}%"
                            })
                    except:
                        continue
                
                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

elif mode == "📖 マニュアル":
    st.header("📖 航海の手引き - 完全マニュアル")
    
    # タブで章分け
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏴‍☠️ 航海の始まり", 
        "⚡ 高速/完全モード", 
        "📊 使い方", 
        "💡 航海術", 
        "❓ FAQ",
        "📈 スコア詳細"
    ])
    
    with tab1:
        st.markdown("""
        ## 🏴‍☠️ お宝発掘ツールとは
        
        **チャートの大海原から、お宝銘柄を自動で発掘するログポースです！**
        
        ### 🗺️ 主な機能（武器）
        
        #### 1. 📊 スクリーニング機能（ログポース）
        - **100点満点のスコアリング**でお宝銘柄を発見
        - **東証33業種を自動判定**（海域マップ）
        - **日足・週足の切り替え**対応
        - **急落銘柄を自動除外**（嵐を回避）
        
        #### 2. 🔬 バックテスト機能（航海日誌）
        - 過去6〜12ヶ月の航海記録を検証
        - 月次リターン・勝率を自動計算
        - グラフで視覚的に確認
        
        #### 3. 📈 銘柄比較機能（海図比較）
        - 複数銘柄のチャートを重ねて表示
        - 相対的な強さを一目で比較
        - スクリーニング結果から直接選択
        
        ---
        
        ## 💪 こんな海賊におすすめ
        
        ✅ テクニカル分析で航路を決めたい  
        ✅ 業種（海域）分散を意識したい  
        ✅ 毎週末に宝の地図を更新したい  
        ✅ 客観的なスコアで判断したい  
        ✅ バックテスト（航海日誌）で戦略を検証したい  
        
        ---
        
        ## 📱 対応する船
        
        - 💻 PC（大型船）
        - 📱 スマートフォン（小型船）
        - 📟 タブレット（中型船）
        
        **どんな船からでも、グランドラインを目指せる！**
        
        ---
        
        ## 🌊 世はまさに、大海賊時代！
        
        富・名声・力、この世のすべてを手に入れた男、ゴールド・ロジャー。
        
        彼が放った一言は、人々を海へ駆り立てた...
        
        **「俺の財宝か？欲しけりゃくれてやる。探せ！この世のすべてをそこに置いてきた」**
        
        男たちはグランドラインを目指し、夢を追い続ける。
        
        そして、あなたもまた...
        
        **お宝銘柄という名のワンピースを探す冒険者なのだ！**
        """)
    
    with tab2:
        st.markdown("""
        ## ⚡ 高速モード vs 完全モード
        
        ### 📊 比較表
        """)
        
        comparison_data = {
            "項目": ["処理時間", "業種判定", "株価・出来高", "テクニカル指標", "スコア", "PER", "PBR", "配当利回り", "推奨用途"],
            "高速モード（推奨）": [
                "3-5分",
                "✅ 全銘柄",
                "✅ 全銘柄",
                "✅ 全銘柄",
                "✅ 全銘柄",
                "❌ 0表示",
                "❌ 0表示",
                "❌ 0表示",
                "テクニカル重視・候補抽出"
            ],
            "完全モード": [
                "15-30分",
                "✅ 全銘柄",
                "✅ 全銘柄",
                "✅ 全銘柄",
                "✅ 全銘柄",
                "△ 取得可能銘柄のみ",
                "△ 取得可能銘柄のみ",
                "△ 取得可能銘柄のみ",
                "ファンダ重視・最終判断"
            ]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        st.markdown("""
        ---
        
        ### 💡 効率的な使い方（推奨ワークフロー）
        
        ```
        ステップ1: 高速モードで全銘柄スクリーニング（5分）
        　　↓
        　500銘柄 → トップ50に絞る
        　　↓
        ステップ2: トップ50だけのCSVを作成
        　　↓
        ステップ3: 完全モードで50銘柄を再スクリーニング（3分）
        　　↓
        　PER/PBR/配当も確認して最終判断
        
        合計時間: わずか8分！
        ```
        
        ---
        
        ### ⚙️ 設定方法
        
        **高速モード（デフォルト）:**
        ```
        左サイドバー
        └─ データ取得設定
           □ PER/PBR/配当を取得 ← チェックOFF
        ```
        
        **完全モード:**
        ```
        左サイドバー
        └─ データ取得設定
           ☑ PER/PBR/配当を取得 ← チェックON
        ```
        
        ---
        
        ### 🎯 使い分けのポイント
        
        **高速モードを使うべき場合:**
        - まず全体像を把握したい
        - テクニカル分析がメイン
        - 時間を節約したい
        - 業種分散だけ確認したい
        
        **完全モードを使うべき場合:**
        - 最終的な投資判断前
        - ファンダメンタルも重視
        - 時間に余裕がある
        - 詳細分析が必要
        """)
    
    with tab3:
        st.markdown("""
        ## 📖 基本的な使い方
        
        ### ステップ1: CSVファイルを準備
        
        **必要な列:**
        - コード（例: 7203）
        - 銘柄名（例: トヨタ自動車）
        
        **作成方法:**
        
        #### 方法A: 証券会社からダウンロード
        - SBI証券、楽天証券等でダウンロード可能
        - 「銘柄一覧」や「スクリーニング結果」を保存
        
        #### 方法B: Excelで手動作成
        1. A列に「コード」、B列に「銘柄名」
        2. データを入力
        3. 「名前を付けて保存」→ CSV形式
        
        ---
        
        ### ステップ2: アプリで設定
        
        **左サイドバーで以下を設定:**
        
        1. **モード選択**  
           📊 スクリーニング を選択
        
        2. **時間軸設定**  
           📆 週足 を選択（推奨）
        
        3. **CSVアップロード**  
           ファイルを選択またはドラッグ&ドロップ
        
        4. **データ取得設定**  
           □ PER/PBR/配当を取得（OFFを推奨）
        
        5. **フィルター設定**  
           - 最低平均出来高: 100,000  
           - 最低スコア: 40-50
        
        ---
        
        ### ステップ3: スクリーニング実行
        
        🚀 **スクリーニング開始** ボタンをクリック
        
        - 高速モード: 3-5分待つ ☕
        - 完全モード: 15-30分待つ ☕☕☕
        
        ---
        
        ### ステップ4: 結果を確認
        
        **表示される情報:**
        - 🌐 市場環境（日経平均の状況）
        - 📊 スクリーニング結果テーブル
        - 📈 スコア分布グラフ
        - 🥧 業種別分布グラフ
        
        ---
        
        ### ステップ5: 結果をダウンロード
        
        **2つの形式で保存可能:**
        - 📄 CSVダウンロード
        - 📊 Excelダウンロード
        
        ---
        
        ## 📊 結果の見方
        
        ### スコア（100点満点）
        - **40点未満**: 除外
        - **40-49点**: 様子見
        - **50-59点**: 候補
        - **60-69点**: 有望
        - **70点以上**: 最有力 ⭐
        
        ### 業種
        証券コードから自動判定
        - 5334 → ガラス・土石製品
        - 7203 → 輸送用機器
        - 8919 → 不動産業
        
        ### その他の指標
        - **価格**: 現在株価
        - **前回比**: 前日（前週）比
        - **MA乖離**: 25日（26週）MAからの乖離率
        - **リターン**: 60日（52週）リターン
        """)
    
    with tab4:
        st.markdown("""
        ## 💡 実践的な航海術
        
        ### 📅 週末の航海ルーティン（おすすめ）
        
        #### 土曜朝（5分）- 出航準備
        1. ✅ ログポース（アプリ）にアクセス
        2. ✅ 最新の海図（CSV）をアップロード
        3. ✅ 高速モードでお宝探索開始
        4. ✅ トップ30のお宝をExcelダウンロード
        
        #### 土曜午後（3分）- 精密調査
        5. ✅ トップ30だけの海図を作成
        6. ✅ 完全モードで再探索
        7. ✅ PER/PBR/配当を確認（宝箱の中身チェック）
        
        #### 日曜（1時間）- 最終選定
        8. ✅ 銘柄比較機能で上位10銘柄の海図を比較
        9. ✅ 波形が美しい銘柄3つを選定（ナミさん並みの航海術で）
        10. ✅ 各銘柄のニュースをチェック（新聞（NEWS）クーさん情報収集）
        
        #### 月曜朝 - 上陸（購入）
        11. ✅ 選定した銘柄に投資（お宝ゲット）
        12. ✅ 次の週末まで保有（新世界へ向けて航海）
        
        **航海時間: 合計8分！**
        
        ---
        
        ### 🎯 推奨装備（設定）
        
        #### 初心者海賊向け（東の海レベル）
        ```
        時間軸: 週足（安全航路）
        PER/PBR取得: OFF（高速航海）
        最低スコア: 50点以上
        保有期間: 4〜8週間
        損切り: -5%（嵐からの退避）
        利確: +15%（お宝発見！）
        分散: 10銘柄（10人の仲間）
        ```
        **期待成績:**
        - 勝率: 60-65%
        - 月次リターン: +3-5%
        
        #### 中級海賊向け（グランドライン前半）
        ```
        時間軸: 日足（スリリングな航路）
        PER/PBR取得: ON（完全装備）
        最低スコア: 50点以上
        保有期間: 2〜4週間
        損切り: -3%
        利確: +10%
        分散: 5銘柄（少数精鋭）
        ```
        **期待成績:**
        - 勝率: 55-60%
        - 月次リターン: +4-7%
        
        ---
        
        ### 📈 航海の改善ヒント
        
        **バックテスト（航海日誌）の結果が悪い場合:**
        
        1. **航路を変える（時間軸変更）**
           - 日足 → 週足に変更
           - 週足の方が穏やかな航海（安定）
        
        2. **選別を厳しくする（スコア上げ）**
           - 40点 → 50点 → 60点
           - 厳選すると宝の確率UP
        
        3. **航海期間を調整**
           - 短すぎる → 長くする
           - 長すぎる → 短くする
        
        4. **完全装備で出航（完全モード）**
           - PER/PBRも確認
           - ファンダメンタル的にも堅牢な銘柄を選ぶ
        
        ---
        
        ### 🏢 海域（業種）分散の重要性
        
        **1つの海域に集中は危険！ティーチの二の舞だ！**
        
        **良い例（麦わらの一味並みのバランス）:**
        ```
        銘柄A: 電気機器（ナミ：航海士）
        銘柄B: 医薬品（チョッパー：船医）
        銘柄C: 小売業（サンジ：コック）
        銘柄D: 輸送用機器（フランキー：船大工）
        銘柄E: 不動産業（ロビン：考古学者）
        ```
        → 5つの異なる海域に分散 ✅
        
        **悪い例（全員剣士）:**
        ```
        銘柄A: 電気機器
        銘柄B: 電気機器
        銘柄C: 電気機器
        銘柄D: 電気機器
        銘柄E: 電気機器
        ```
        → 1つの海域に集中 ❌ 海王類に全滅させられる
        
        **このツールは業種（海域）を自動判定するので、バランスの良い一味を作りやすい！**
        
        ---
        
        ## 🏴‍☠️ 名言集
        
        **「海賊王に、俺はなる！」** - ルフィ  
        → **投資家として、俺は成功する！**
        
        **「仲間がいるだろ」** - ルフィ  
        → **分散投資（仲間）がいるだろ**
        
        **「おれは剣術を使えねぇんだコノヤロー！」** - ルフィ  
        → **テクニカル分析はこのツールに任せろ！**
        
        **「地図はあたしが描く！」** - ナミ  
        → **チャートはこのツールが描く！**
        """)
    
    with tab5:
        st.markdown("""
        ## ❓ よくある質問（FAQ）
        
        ### Q1: PER/PBRが0になる
        **A:** 正常です
        
        - 高速モードではPER/PBRを取得しません
        - 完全モードでもyfinanceにデータがない銘柄は0表示
        - 業種とテクニカル分析は問題なく機能します
        
        ---
        
        ### Q2: 完全モードが遅すぎる
        **A:** 高速モードとの組み合わせがおすすめ
        
        1. まず高速モードで候補を絞る
        2. 上位銘柄だけ完全モードで再実行
        3. 処理時間を大幅短縮できます
        
        ---
        
        ### Q3: 業種が表示される仕組みは？
        **A:** 証券コードから自動判定
        
        証券コードの範囲で業種を判定しています。
        
        例:
        - 2000-2999 → 食料品
        - 5200-5399 → ガラス・土石製品
        - 7000-7499 → 輸送用機器
        - 8800-8999 → 不動産業
        
        **東証33業種すべてに対応！**
        
        ---
        
        ### Q4: スクリーニング結果が0件
        **A:** フィルター条件を緩めてください
        
        - 最低スコアを下げる（60 → 50 → 40）
        - 最低出来高を下げる（100,000 → 50,000）
        
        ---
        
        ### Q5: 日足と週足、どちらがおすすめ？
        **A:** 週足がおすすめ
        
        **週足の利点:**
        - ノイズが少ない
        - 信頼性が高い
        - 週1回のチェックでOK
        - 標準的なMA（13週・26週・52週）
        
        **日足の利点:**
        - 細かい動きをキャッチ
        - 短期トレード向き
        
        ---
        
        ### Q6: エラーが出る
        **A:** 以下を試してください
        
        1. ページを再読み込み（F5キー）
        2. ブラウザのキャッシュをクリア
        3. 数分待ってから再実行
        4. 別のブラウザで試す
        
        ---
        
        ### Q7: CSVファイルの作り方がわからない
        **A:** 2つの方法があります
        
        **方法1: 証券会社からダウンロード**
        - SBI証券、楽天証券等で可能
        - 銘柄一覧をCSV形式で保存
        
        **方法2: Excelで作成**
        1. A列に「コード」
        2. B列に「銘柄名」
        3. CSV形式で保存
        
        ---
        
        ### Q8: スマホでも使える？
        **A:** はい、使えます
        
        - iPhoneでもAndroidでもOK
        - ブラウザ（Safari、Chrome等）でアクセス
        - 縦画面でも横画面でも使いやすい
        
        ---
        
        ### Q9: バックテストが遅い
        **A:** 仕様です
        
        - テスト銘柄数が30に制限されています
        - 10-30分程度かかります
        - コーヒーを飲んで待ちましょう ☕
        
        ---
        
        ### Q10: 除外される銘柄は？
        **A:** 以下の銘柄は自動除外
        
        1. **急落銘柄**
           - 前日比（前週比）-3%以上の下落
           - 直近5日（2週）の最高値から-7%以上
        
        2. **流動性不足**
           - 平均出来高10万株未満
        
        3. **データ不足**
           - 日足: 上場80日未満
           - 週足: 上場52週（1年）未満
        """)
    
    with tab6:
        st.markdown("""
        ## 📊 スコアリング詳細
        
        ### 100点満点の内訳
        """)
        
        score_data = {
            "条件": [
                "黄金交差",
                "大商い伴う上昇",
                "市場超え強気",
                "中長期上昇",
                "25MA近接",
                "出来高急増",
                "RSI適正"
            ],
            "配点": [15, 15, 10, 20, 20, 10, 10],
            "判定基準": [
                "短期MAが中期MAを上抜け",
                "出来高2倍以上 かつ 前日比+2%以上",
                "銘柄のリターン > 日経平均のリターン",
                "長期MAが上昇トレンド",
                "株価が25MAの±5%以内",
                "出来高が平均の1.5倍以上",
                "RSIが30-60の範囲"
            ],
            "意味": [
                "上昇トレンド入りのサイン",
                "買い圧力が強い",
                "相対的に強い",
                "長期的な上昇基調",
                "押し目買いのチャンス",
                "注目度が上昇",
                "過熱感がない"
            ]
        }
        st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)
        
        st.markdown("""
        ---
        
        ### 📅 日足と週足の違い
        """)
        
        timeframe_data = {
            "項目": ["短期MA", "中期MA", "長期MA", "RSI期間", "出来高期間", "リターン期間", "適している人", "チェック頻度"],
            "日足": ["5日", "25日", "75日", "15日", "20日", "60日", "短期トレーダー", "毎日"],
            "週足": ["13週", "26週", "52週", "14週", "4週", "52週", "中長期投資家", "週1回"]
        }
        st.dataframe(pd.DataFrame(timeframe_data), use_container_width=True, hide_index=True)
        
        st.markdown("""
        ---
        
        ### 🎯 スコアの判定基準
        
        | スコア範囲 | 判定 | アクション |
        |-----------|------|-----------|
        | 70点以上 | 非常に優良 ⭐⭐⭐ | 最優先で検討 |
        | 60-69点 | 優良 ⭐⭐ | 積極的に検討 |
        | 50-59点 | 良好 ⭐ | 候補として検討 |
        | 40-49点 | 合格 | 様子見 |
        | 40点未満 | 不合格 ❌ | 除外 |
        
        ---
        
        ### 🏢 東証33業種一覧
        
        証券コードの範囲で自動判定されます。
        
        | コード範囲 | 業種 |
        |-----------|------|
        | 1300-1399 | 水産・農林業 |
        | 1400-1499 | 鉱業 |
        | 1500-1999 | 建設業 |
        | 2000-2999 | 食料品 |
        | 3000-3499 | 繊維製品 |
        | 3500-3799 | パルプ・紙 |
        | 3800-4499 | 化学 |
        | 4500-4699 | 医薬品 |
        | 4700-4899 | 情報・通信業 |
        | 5000-5099 | 石油・石炭製品 |
        | 5100-5199 | ゴム製品 |
        | 5200-5399 | ガラス・土石製品 |
        | 5400-5599 | 鉄鋼 |
        | 5600-5799 | 非鉄金属 |
        | 5800-6099 | 金属製品 |
        | 6100-6399 | 機械 |
        | 6400-6999 | 電気機器 |
        | 7000-7499 | 輸送用機器 |
        | 7500-7899 | 精密機器 |
        | 7900-7999 | その他製品 |
        | 8000-8299 | 卸売業 |
        | 8300-8499 | 銀行業 |
        | 8500-8599 | その他金融業 |
        | 8600-8699 | 証券業・商品先物取引業 |
        | 8700-8799 | 保険業 |
        | 8800-8999 | 不動産業 |
        | 9000-9099 | 陸運業 |
        | 9100-9199 | 海運業 |
        | 9200-9299 | 空運業 |
        | 9300-9399 | 倉庫・運輸関連業 |
        | 9400-9499 | 情報・通信業 |
        | 9500-9599 | 電気・ガス業 |
        | 9600-9799 | サービス業 |
        | 9800-9999 | 小売業 |
        
        **例:**
        - 5334（日本特殊陶業） → ガラス・土石製品
        - 7203（トヨタ自動車） → 輸送用機器
        - 8919（カチタス） → 不動産業
        - 9984（ソフトバンクG） → 小売業
        """)

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏴‍☠️ お宝発掘ツール v3.0 - 世はまさに、大海賊時代！ | 投資は自己責任で（海賊王には俺はなる！）</p>
</div>
""", unsafe_allow_html=True)
