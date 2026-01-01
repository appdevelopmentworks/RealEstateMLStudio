"""
ユーティリティモジュール - 共通関数とヘルパー
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os


def load_css():
    """カスタムCSSを読み込む"""
    css = """
    <style>
        /* メインコンテナ */
        .main {
            padding: 1rem;
        }
        
        /* ヘッダースタイル */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        /* カードスタイル */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9rem;
        }
        
        /* ボタンスタイル */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* サイドバー */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* タブスタイル */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* アップローダー */
        .uploadedFile {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* 成功メッセージ */
        .success-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        /* 警告メッセージ */
        .warning-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        /* 情報カード */
        .info-card {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 10px 10px 0;
        }
        
        /* プログレスバー */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* データフレーム */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* セレクトボックス */
        .stSelectbox > div > div {
            border-radius: 10px;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def create_header(title: str, subtitle: str = ""):
    """ヘッダーを作成"""
    header_html = f"""
    <div class="main-header">
        <h1>🏠 {title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def create_metric_card(label: str, value: str, icon: str = "📊"):
    """メトリックカードを作成"""
    card_html = f"""
    <div class="metric-card">
        <div style="font-size: 2rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """
    return card_html


def format_number(value: float, precision: int = 4) -> str:
    """数値をフォーマット"""
    if abs(value) >= 1e6:
        return f"{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.{precision}f}"


def get_data_info(df: pd.DataFrame) -> dict:
    """データフレームの情報を取得"""
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
        'missing_total': df.isnull().sum().sum(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }


def save_to_csv(df: pd.DataFrame, filename: str) -> str:
    """データフレームをCSVに保存"""
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)
    return filepath


def generate_report_filename(prefix: str = "report") -> str:
    """レポートファイル名を生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def display_dataframe_info(df: pd.DataFrame):
    """データフレーム情報を表示"""
    info = get_data_info(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("行数", f"{info['rows']:,}")
    with col2:
        st.metric("列数", f"{info['columns']}")
    with col3:
        st.metric("欠損値", f"{info['missing_total']:,}")
    with col4:
        st.metric("メモリ", f"{info['memory_mb']:.2f} MB")


def show_success_message(message: str):
    """成功メッセージを表示"""
    st.markdown(f"""
    <div class="success-box">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)


def show_warning_message(message: str):
    """警告メッセージを表示"""
    st.markdown(f"""
    <div class="warning-box">
        ⚠️ {message}
    </div>
    """, unsafe_allow_html=True)


def show_info_card(title: str, content: str):
    """情報カードを表示"""
    st.markdown(f"""
    <div class="info-card">
        <strong>{title}</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)


# セッション状態の初期化ヘルパー
def init_session_state():
    """セッション状態を初期化"""
    defaults = {
        'df': None,
        'df_processed': None,
        'preprocessor': None,
        'trainer': None,
        'model': None,
        'model_type': None,
        'target_column': None,
        'feature_columns': None,
        'metrics': None,
        'cv_scores': None,
        'is_trained': False,
        'comparison_results': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
