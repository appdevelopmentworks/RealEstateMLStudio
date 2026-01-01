"""
RealEstateMLStudio - 不動産価格予測MLスタジオ
メインアプリケーション v2.2 - SHAP分析・信頼区間・PDFレポート・What-if分析
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessor import DataPreprocessor, get_data_summary
from src.trainer import ModelTrainer, StackingTrainer, compare_models, get_best_model
from src.visualizer import Visualizer
from src.analysis import SHAPAnalyzer, PredictionInterval, WhatIfAnalyzer, PDFReportGenerator
from src.utils import (
    load_css, create_header, init_session_state, 
    display_dataframe_info, show_success_message, show_warning_message
)

import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="RealEstateMLStudio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSSの読み込み
load_css()

# セッション状態の初期化
init_session_state()

# Visualizerのインスタンス
viz = Visualizer()


def load_sample_data(dataset_name: str) -> pd.DataFrame:
    """サンプルデータをロード"""
    
    if dataset_name == "california":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['MedHouseVal'] = data.target
        
        column_mapping = {
            'MedInc': '世帯収入中央値',
            'HouseAge': '築年数',
            'AveRooms': '平均部屋数',
            'AveBedrms': '平均寝室数',
            'Population': '人口',
            'AveOccup': '平均世帯人数',
            'Latitude': '緯度',
            'Longitude': '経度',
            'MedHouseVal': '住宅価格'
        }
        df = df.rename(columns=column_mapping)
        return df
    
    elif dataset_name == "tokyo_sample":
        np.random.seed(42)
        n_samples = 1000
        
        districts = ['港区', '渋谷区', '新宿区', '世田谷区', '目黒区', 
                    '品川区', '大田区', '杉並区', '中野区', '練馬区']
        
        df = pd.DataFrame({
            '区': np.random.choice(districts, n_samples),
            '築年数': np.random.randint(0, 50, n_samples),
            '面積_m2': np.random.uniform(20, 150, n_samples).round(1),
            '階数': np.random.randint(1, 50, n_samples),
            '駅徒歩分': np.random.randint(1, 20, n_samples),
            '部屋数': np.random.randint(1, 5, n_samples),
            'バルコニー有': np.random.choice([0, 1], n_samples),
            'オートロック': np.random.choice([0, 1], n_samples),
        })
        
        base_price = 3000
        district_premium = {'港区': 2000, '渋谷区': 1800, '新宿区': 1500, 
                          '世田谷区': 1200, '目黒区': 1400, '品川区': 1100,
                          '大田区': 800, '杉並区': 900, '中野区': 850, '練馬区': 700}
        
        df['価格_万円'] = (
            base_price +
            df['区'].map(district_premium) +
            df['面積_m2'] * 50 -
            df['築年数'] * 30 +
            df['階数'] * 20 -
            df['駅徒歩分'] * 50 +
            df['部屋数'] * 200 +
            df['バルコニー有'] * 100 +
            df['オートロック'] * 150 +
            np.random.normal(0, 500, n_samples)
        ).round(0).astype(int)
        
        df['価格_万円'] = df['価格_万円'].clip(lower=1000)
        return df
    
    elif dataset_name == "boston_simple":
        np.random.seed(42)
        n_samples = 500
        
        df = pd.DataFrame({
            'RM': np.random.uniform(4, 9, n_samples).round(2),
            'LSTAT': np.random.uniform(2, 35, n_samples).round(2),
            'PTRATIO': np.random.uniform(12, 22, n_samples).round(1),
            'DIS': np.random.uniform(1, 12, n_samples).round(2),
            'NOX': np.random.uniform(0.4, 0.9, n_samples).round(3),
            'AGE': np.random.uniform(10, 100, n_samples).round(1),
            'TAX': np.random.randint(180, 720, n_samples),
            'CRIM': np.random.exponential(3, n_samples).round(3),
        })
        
        df['PRICE'] = (
            50 +
            df['RM'] * 5 -
            df['LSTAT'] * 0.5 -
            df['PTRATIO'] * 0.8 +
            df['DIS'] * 0.3 -
            df['NOX'] * 15 -
            df['AGE'] * 0.05 -
            df['TAX'] * 0.02 -
            df['CRIM'] * 0.3 +
            np.random.normal(0, 3, n_samples)
        ).round(1).clip(lower=5)
        
        column_mapping = {
            'RM': '部屋数',
            'LSTAT': '低所得者率',
            'PTRATIO': '生徒教師比率',
            'DIS': '都心距離',
            'NOX': '大気汚染指数',
            'AGE': '築年数',
            'TAX': '固定資産税',
            'CRIM': '犯罪率',
            'PRICE': '住宅価格'
        }
        df = df.rename(columns=column_mapping)
        return df
    
    return None


def create_prediction_form(feature_columns: list, df_original: pd.DataFrame, key_prefix: str = "input") -> dict:
    """予測用の入力フォームを作成"""
    
    input_data = {}
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(feature_columns):
        if col in df_original.columns:
            col_data = df_original[col]
            
            if col_data.dtype == 'object' or col_data.nunique() < 10:
                unique_values = col_data.unique().tolist()
                with col1 if i % 2 == 0 else col2:
                    input_data[col] = st.selectbox(
                        f"{col}",
                        options=unique_values,
                        key=f"{key_prefix}_{col}"
                    )
            else:
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                mean_val = float(col_data.mean())
                
                with col1 if i % 2 == 0 else col2:
                    if col_data.dtype in ['int64', 'int32']:
                        input_data[col] = st.number_input(
                            f"{col}",
                            min_value=int(min_val),
                            max_value=int(max_val * 1.5),
                            value=int(mean_val),
                            step=1,
                            key=f"{key_prefix}_{col}"
                        )
                    else:
                        input_data[col] = st.number_input(
                            f"{col}",
                            min_value=min_val * 0.5,
                            max_value=max_val * 1.5,
                            value=mean_val,
                            step=(max_val - min_val) / 100,
                            format="%.2f",
                            key=f"{key_prefix}_{col}"
                        )
        else:
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=0.0,
                    key=f"{key_prefix}_{col}"
                )
    
    return input_data


def main():
    """メインアプリケーション"""
    
    # ヘッダーバナー画像
    if os.path.exists("images/Appbanner.png"):
        st.image("images/Appbanner.png", use_container_width=True)
    else:
        create_header(
            "RealEstateMLStudio",
            "XGBoost / LightGBM / CatBoost / スタッキング による高精度不動産価格予測"
        )
    
    # サイドバー
    with st.sidebar:
        st.image("https://img.icons8.com/3d-fluency/94/home.png", width=80)
        st.title("設定パネル")
        
        st.markdown("---")
        
        # モデル選択
        st.subheader("🤖 モデル設定")
        model_type = st.selectbox(
            "アルゴリズム選択",
            ["XGBoost", "LightGBM", "CatBoost", "スタッキング (アンサンブル)", "全モデル比較"],
            help="使用する機械学習アルゴリズムを選択"
        )
        
        if model_type == "スタッキング (アンサンブル)":
            st.markdown("**スタッキング構成**")
            use_xgb_stack = st.checkbox("XGBoost", value=True)
            use_lgb_stack = st.checkbox("LightGBM", value=True)
            use_cat_stack = st.checkbox("CatBoost", value=True)
        else:
            use_xgb_stack = use_lgb_stack = use_cat_stack = True
        
        include_stacking = False
        if model_type == "全モデル比較":
            include_stacking = st.checkbox("スタッキングも含める", value=False)
        
        st.subheader("⚙️ チューニング設定")
        use_tuning = st.checkbox("ハイパーパラメータチューニング", value=False)
        
        if use_tuning:
            n_trials = st.slider("試行回数", 10, 200, 50, 10)
        else:
            n_trials = 0
        
        use_cv = st.checkbox("交差検証を実行", value=True)
        if use_cv:
            cv_folds = st.slider("Fold数", 3, 10, 5)
        else:
            cv_folds = 5
        
        st.subheader("🔧 前処理設定")
        handle_missing = st.checkbox("欠損値を自動処理", value=True)
        encode_categorical = st.checkbox("カテゴリ変数をエンコード", value=True)
        handle_outliers = st.checkbox("異常値を処理", value=False)
        scale_features = st.checkbox("特徴量をスケーリング", value=False)
        
        if model_type == "CatBoost":
            st.subheader("🐱 CatBoost設定")
            use_native_cat = st.checkbox(
                "カテゴリ変数をネイティブ処理", 
                value=True,
                help="CatBoostのネイティブカテゴリ処理を使用"
            )
            if use_native_cat:
                encode_categorical = False
        else:
            use_native_cat = False
        
        st.markdown("---")
        
        st.subheader("💾 モデル管理")
        if st.session_state.get('is_trained', False):
            if st.button("📥 モデルを保存"):
                save_model()
        
        uploaded_model = st.file_uploader("モデルを読み込み", type=['joblib'])
        if uploaded_model:
            load_saved_model(uploaded_model)
    
    # メインコンテンツ - 6タブに拡張
    tabs = st.tabs([
        "📤 データアップロード", 
        "🔍 データ分析 (EDA)", 
        "🎯 モデル学習",
        "📊 評価結果",
        "🔮 予測実行",
        "🔬 詳細分析"  # 新しいタブ
    ])
    
    # タブ1: データアップロード
    with tabs[0]:
        render_data_upload_tab()
    
    # タブ2: データ分析 (EDA)
    with tabs[1]:
        render_eda_tab()
    
    # タブ3: モデル学習
    with tabs[2]:
        render_training_tab(model_type, use_tuning, n_trials, use_cv, cv_folds,
                           handle_missing, encode_categorical, handle_outliers,
                           scale_features, use_native_cat, use_xgb_stack,
                           use_lgb_stack, use_cat_stack, include_stacking)
    
    # タブ4: 評価結果
    with tabs[3]:
        render_evaluation_tab()
    
    # タブ5: 予測実行
    with tabs[4]:
        render_prediction_tab()
    
    # タブ6: 詳細分析（新機能）
    with tabs[5]:
        render_advanced_analysis_tab()


def render_data_upload_tab():
    """データアップロードタブ"""
    st.header("Step 1: データのアップロード")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <strong>📋 対応フォーマット</strong><br>
            CSV形式のファイルをドラッグ＆ドロップしてください。<br>
            ・教師あり学習用のデータセット（目的変数を含む）<br>
            ・欠損値やカテゴリ変数は自動で処理されます
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "学習データをアップロード",
            type=['csv'],
            help="CSVファイルをドラッグ＆ドロップ"
        )
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 3rem;">📊</div>
            <div class="metric-label">データをアップロードして<br>分析を開始しましょう</div>
        </div>
        """, unsafe_allow_html=True)
    
    # サンプルデータセクション
    st.markdown("---")
    st.subheader("🎮 サンプルデータで試す")
    st.markdown("データがない場合は、以下のサンプルデータセットでアプリの機能を試すことができます。")
    
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        st.markdown("""
        **🌴 カリフォルニア住宅**
        - 20,640件のデータ
        - 8つの特徴量
        - 住宅価格（中央値）を予測
        """)
        if st.button("📥 カリフォルニア住宅データ", key="load_california"):
            df = load_sample_data("california")
            st.session_state['df'] = df
            st.session_state['sample_data_name'] = 'カリフォルニア住宅'
            st.rerun()
    
    with sample_col2:
        st.markdown("""
        **🗼 東京マンション（架空）**
        - 1,000件のデータ
        - 区・面積・築年数など
        - マンション価格を予測
        """)
        if st.button("📥 東京マンションデータ", key="load_tokyo"):
            df = load_sample_data("tokyo_sample")
            st.session_state['df'] = df
            st.session_state['sample_data_name'] = '東京マンション'
            st.rerun()
    
    with sample_col3:
        st.markdown("""
        **🏘️ シンプル住宅データ**
        - 500件のデータ
        - 部屋数・築年数など
        - 住宅価格を予測
        """)
        if st.button("📥 シンプル住宅データ", key="load_boston"):
            df = load_sample_data("boston_simple")
            st.session_state['df'] = df
            st.session_state['sample_data_name'] = 'シンプル住宅'
            st.rerun()
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.session_state['sample_data_name'] = None
    
    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        
        if st.session_state.get('sample_data_name'):
            show_success_message(f"サンプルデータ「{st.session_state['sample_data_name']}」を読み込みました: {len(df):,}行 × {len(df.columns)}列")
        else:
            show_success_message(f"データを読み込みました: {len(df):,}行 × {len(df.columns)}列")
        
        display_dataframe_info(df)
        
        st.markdown("---")
        st.subheader("📋 データプレビュー")
        st.dataframe(df.head(20), use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔢 数値列")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(numeric_cols)
        with col2:
            st.subheader("📝 カテゴリ列")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            st.write(cat_cols if cat_cols else "なし")


def render_eda_tab():
    """EDAタブ"""
    st.header("Step 2: 探索的データ分析 (EDA)")
    
    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        
        st.subheader("📈 基本統計量")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("🔍 欠損値分析")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            '欠損数': missing,
            '欠損率 (%)': missing_pct
        }).sort_values('欠損数', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(missing_df[missing_df['欠損数'] > 0], use_container_width=True)
        with col2:
            if missing.sum() > 0:
                fig = viz.plot_feature_importance(
                    missing[missing > 0].sort_values(ascending=False),
                    top_n=20,
                    title="欠損値の分布"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ 欠損値はありません")
        
        st.subheader("🔗 特徴量相関マトリックス")
        fig_corr = viz.plot_eda_dashboard(df)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("📊 数値変数の分布")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("表示する列を選択", numeric_cols)
            
            import plotly.express as px
            fig = px.histogram(
                df, x=selected_col, 
                nbins=50,
                title=f"{selected_col} の分布",
                marginal="box"
            )
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_warning_message("先にデータをアップロードしてください")


def render_training_tab(model_type, use_tuning, n_trials, use_cv, cv_folds,
                        handle_missing, encode_categorical, handle_outliers,
                        scale_features, use_native_cat, use_xgb_stack,
                        use_lgb_stack, use_cat_stack, include_stacking):
    """モデル学習タブ"""
    st.header("Step 3: モデル学習")
    
    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "🎯 ターゲット列（予測したい列）",
                df.columns.tolist(),
                index=len(df.columns) - 1
            )
            st.session_state['target_column'] = target_column
        
        with col2:
            test_size = st.slider("テストデータの割合", 0.1, 0.4, 0.2, 0.05)
        
        st.markdown("---")
        
        model_info = {
            "XGBoost": "🚀 高精度な勾配ブースティング。バランスの取れた性能。",
            "LightGBM": "⚡ 高速・軽量。大規模データに最適。",
            "CatBoost": "🐱 カテゴリ変数に強い。過学習しにくい。",
            "スタッキング (アンサンブル)": "🏆 複数モデルを組み合わせ最高精度を実現。",
            "全モデル比較": "📊 全モデルを比較して最適なものを選択。"
        }
        
        st.info(f"**選択中のモデル**: {model_type}\n\n{model_info[model_type]}")
        
        st.markdown("---")
        st.subheader("🔧 データ前処理")
        
        if st.button("前処理を実行", key="preprocess_btn"):
            with st.spinner("前処理中..."):
                preprocessor = DataPreprocessor()
                
                actual_encode = encode_categorical and not (model_type == "CatBoost" and use_native_cat)
                
                df_processed = preprocessor.auto_preprocess(
                    df,
                    target_column=target_column,
                    handle_missing=handle_missing,
                    encode_cat=actual_encode,
                    handle_outliers_flag=handle_outliers,
                    scale=scale_features
                )
                
                st.session_state['df_processed'] = df_processed
                st.session_state['preprocessor'] = preprocessor
                st.session_state['df_original'] = df.copy()
                st.session_state['use_native_cat'] = use_native_cat if model_type == "CatBoost" else False
                
                show_success_message("前処理が完了しました！")
                
                st.write("前処理後のデータ:")
                display_dataframe_info(df_processed)
                st.dataframe(df_processed.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🚀 モデル学習")
        
        if st.session_state.get('df_processed') is not None:
            if st.button("🎓 学習開始", type="primary", key="train_btn"):
                train_model(
                    model_type, 
                    use_tuning, 
                    n_trials, 
                    use_cv, 
                    cv_folds, 
                    test_size,
                    use_xgb_stack,
                    use_lgb_stack,
                    use_cat_stack,
                    include_stacking
                )
        else:
            st.info("👆 まず前処理を実行してください")
    else:
        show_warning_message("先にデータをアップロードしてください")


def render_evaluation_tab():
    """評価結果タブ"""
    st.header("Step 4: モデル評価")
    
    if st.session_state.get('is_trained', False):
        trainer = st.session_state['trainer']
        metrics = st.session_state['metrics']
        
        # 評価指標ダッシュボード（常に表示可能）
        st.subheader("📊 評価指標ダッシュボード")
        fig_metrics = viz.plot_metrics_dashboard(metrics, getattr(trainer, 'cv_scores', None))
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # y_testとy_predがある場合のみ詳細グラフを表示
        y_test = getattr(trainer, 'y_test', None)
        y_pred = getattr(trainer, 'y_pred', None)
        
        if y_test is not None and y_pred is not None:
            y_test_vals = y_test.values if hasattr(y_test, 'values') else y_test
            
            st.subheader("🎯 実測値 vs 予測値")
            fig_pred = viz.plot_actual_vs_predicted(y_test_vals, y_pred)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.subheader("📈 残差分析")
            fig_residuals = viz.plot_residuals(y_test_vals, y_pred)
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            st.subheader("📊 予測値と実測値の分布")
            fig_dist = viz.plot_prediction_distribution(y_test_vals, y_pred)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("📝 読み込んだモデルのため、詳細な評価グラフは表示できません。新しいデータで予測を実行してください。")
        
        # 特徴量重要度
        feature_importance = getattr(trainer, 'feature_importance', None)
        if feature_importance is not None:
            st.subheader("🔑 特徴量重要度")
            fig_importance = viz.plot_feature_importance(
                feature_importance,
                top_n=min(20, len(feature_importance))
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # 交差検証結果
        cv_scores = getattr(trainer, 'cv_scores', None)
        if cv_scores:
            st.subheader("🔄 交差検証結果")
            fig_cv = viz.plot_cv_results(cv_scores)
            st.plotly_chart(fig_cv, use_container_width=True)
        
        # モデル比較結果
        if st.session_state.get('comparison_results'):
            st.subheader("⚖️ モデル比較")
            comparison = st.session_state['comparison_results']
            fig_compare = viz.plot_model_comparison(comparison['comparison_df'])
            st.plotly_chart(fig_compare, use_container_width=True)
            
            st.dataframe(comparison['comparison_df'], use_container_width=True)
            
            best = get_best_model(comparison)
            st.success(f"🏆 最良モデル: **{best.upper()}** (R² = {comparison[best]['metrics']['r2']:.4f})")
        
    else:
        show_warning_message("先にモデルを学習してください")


def render_prediction_tab():
    """予測実行タブ"""
    st.header("Step 5: 新しいデータで予測")
    
    if st.session_state.get('is_trained', False):
        
        pred_method = st.radio(
            "予測方法を選択",
            ["📁 CSVファイルから一括予測", "📝 手入力で予測"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if pred_method == "📁 CSVファイルから一括予測":
            st.markdown("""
            <div class="info-card">
                <strong>📋 予測データの準備</strong><br>
                学習データと同じ形式（同じ列名・順序）のCSVファイルをアップロードしてください。
            </div>
            """, unsafe_allow_html=True)
            
            pred_file = st.file_uploader(
                "予測したいデータをアップロード",
                type=['csv'],
                key="pred_uploader"
            )
            
            if pred_file:
                df_pred = pd.read_csv(pred_file)
                
                st.subheader("📋 アップロードされたデータ")
                st.dataframe(df_pred.head(10), use_container_width=True)
                
                if st.button("🔮 予測実行", type="primary", key="batch_predict"):
                    with st.spinner("予測中..."):
                        try:
                            preprocessor = st.session_state.get('preprocessor')
                            if preprocessor and not st.session_state.get('use_native_cat', False):
                                df_pred_processed = preprocessor.transform_new_data(df_pred)
                            else:
                                df_pred_processed = df_pred
                            
                            trainer = st.session_state['trainer']
                            predictions = trainer.predict(df_pred_processed)
                            
                            df_result = df_pred.copy()
                            df_result['予測値'] = predictions
                            
                            show_success_message("予測が完了しました！")
                            
                            st.subheader("📊 予測結果")
                            st.dataframe(df_result, use_container_width=True, height=400)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("平均予測値", f"{predictions.mean():,.2f}")
                            with col2:
                                st.metric("最小予測値", f"{predictions.min():,.2f}")
                            with col3:
                                st.metric("最大予測値", f"{predictions.max():,.2f}")
                            with col4:
                                st.metric("標準偏差", f"{predictions.std():,.2f}")
                            
                            csv = df_result.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                "📥 予測結果をダウンロード",
                                csv,
                                "prediction_results.csv",
                                "text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"予測中にエラーが発生しました: {str(e)}")
        
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 手入力で予測</strong><br>
                物件情報を入力して、価格を予測します。
            </div>
            """, unsafe_allow_html=True)
            
            feature_columns = st.session_state.get('feature_columns', [])
            df_original = st.session_state.get('df_original', st.session_state.get('df'))
            
            if feature_columns and df_original is not None:
                st.subheader("📝 物件情報を入力")
                input_data = create_prediction_form(feature_columns, df_original, "pred")
                
                st.markdown("---")
                
                if st.button("🔮 この物件の価格を予測", type="primary", key="single_predict"):
                    with st.spinner("予測中..."):
                        try:
                            df_input = pd.DataFrame([input_data])
                            
                            preprocessor = st.session_state.get('preprocessor')
                            if preprocessor and not st.session_state.get('use_native_cat', False):
                                df_input_processed = preprocessor.transform_new_data(df_input)
                            else:
                                df_input_processed = df_input
                            
                            if not st.session_state.get('use_native_cat', False):
                                df_input_processed = df_input_processed.select_dtypes(include=[np.number])
                            
                            trainer = st.session_state['trainer']
                            prediction = trainer.predict(df_input_processed)[0]
                            
                            st.markdown("---")
                            st.subheader("🎉 予測結果")
                            
                            target_column = st.session_state.get('target_column', '価格')
                            
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 20px;
                                padding: 40px;
                                text-align: center;
                                color: white;
                                margin: 20px 0;
                                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                            ">
                                <h2 style="margin: 0; font-size: 1.2rem; opacity: 0.9;">予測 {target_column}</h2>
                                <h1 style="margin: 10px 0; font-size: 3.5rem; font-weight: bold;">
                                    {prediction:,.2f}
                                </h1>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.subheader("📋 入力条件")
                            col1, col2 = st.columns(2)
                            items = list(input_data.items())
                            mid = len(items) // 2
                            
                            with col1:
                                for key, value in items[:mid]:
                                    st.write(f"**{key}**: {value}")
                            with col2:
                                for key, value in items[mid:]:
                                    st.write(f"**{key}**: {value}")
                            
                        except Exception as e:
                            st.error(f"予測中にエラーが発生しました: {str(e)}")
            else:
                st.warning("特徴量情報が見つかりません。モデルを再学習してください。")
    else:
        show_warning_message("先にモデルを学習してください")


def render_advanced_analysis_tab():
    """詳細分析タブ（SHAP, 信頼区間, PDF, What-if）"""
    st.header("Step 6: 詳細分析")
    
    if not st.session_state.get('is_trained', False):
        show_warning_message("先にモデルを学習してください")
        return
    
    # サブタブを作成
    analysis_tabs = st.tabs([
        "🔍 SHAP分析",
        "📊 予測信頼区間",
        "📄 PDFレポート",
        "🔄 What-if分析"
    ])
    
    trainer = st.session_state['trainer']
    feature_columns = st.session_state.get('feature_columns', [])
    df_original = st.session_state.get('df_original', st.session_state.get('df'))
    df_processed = st.session_state.get('df_processed')
    target_column = st.session_state.get('target_column', '価格')
    
    # SHAP分析タブ
    with analysis_tabs[0]:
        st.subheader("🔍 SHAP分析 - 予測の説明")
        st.markdown("""
        SHAP（SHapley Additive exPlanations）は、各特徴量が予測にどれだけ貢献しているかを説明します。
        - **正のSHAP値**: 予測価格を上げる要因
        - **負のSHAP値**: 予測価格を下げる要因
        """)
        
        if st.button("🔬 SHAP分析を実行", key="run_shap"):
            with st.spinner("SHAP値を計算中...（少し時間がかかります）"):
                try:
                    # 学習データを取得
                    X_train = df_processed.drop(columns=[target_column])
                    if not st.session_state.get('use_native_cat', False):
                        X_train = X_train.select_dtypes(include=[np.number])
                    
                    # SHAP分析器を作成
                    shap_analyzer = SHAPAnalyzer(trainer.model, X_train)
                    shap_analyzer.calculate_shap_values(max_samples=300)
                    
                    st.session_state['shap_analyzer'] = shap_analyzer
                    
                    # SHAP重要度プロット
                    st.subheader("📊 SHAP特徴量重要度")
                    fig_shap = shap_analyzer.plot_summary(top_n=15)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    show_success_message("SHAP分析が完了しました！")
                    
                except Exception as e:
                    st.error(f"SHAP分析中にエラーが発生しました: {str(e)}")
        
        # 個別予測の説明
        if st.session_state.get('shap_analyzer'):
            st.markdown("---")
            st.subheader("🎯 個別予測の説明")
            st.markdown("物件情報を入力して、予測の内訳を確認できます。")
            
            if feature_columns and df_original is not None:
                input_data_shap = create_prediction_form(feature_columns, df_original, "shap")
                
                if st.button("📊 この物件の予測を説明", key="explain_shap"):
                    with st.spinner("説明を生成中..."):
                        try:
                            df_input = pd.DataFrame([input_data_shap])
                            
                            preprocessor = st.session_state.get('preprocessor')
                            if preprocessor and not st.session_state.get('use_native_cat', False):
                                df_input_processed = preprocessor.transform_new_data(df_input)
                            else:
                                df_input_processed = df_input
                            
                            if not st.session_state.get('use_native_cat', False):
                                df_input_processed = df_input_processed.select_dtypes(include=[np.number])
                            
                            shap_analyzer = st.session_state['shap_analyzer']
                            explanation = shap_analyzer.explain_prediction(df_input_processed)
                            
                            st.session_state['last_shap_explanation'] = explanation
                            st.session_state['last_input_data'] = df_input_processed
                            
                            # 予測結果
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 15px; padding: 20px; text-align: center; color: white; margin: 15px 0;">
                                <h3 style="margin: 0;">予測 {target_column}: {explanation['prediction']:,.2f}</h3>
                                <p style="margin: 5px 0; opacity: 0.8;">基準値: {explanation['base_value']:,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Force Plot
                            fig_force = shap_analyzer.plot_force_single(explanation)
                            st.plotly_chart(fig_force, use_container_width=True)
                            
                            # 貢献度テーブル
                            st.subheader("📋 全特徴量の貢献度")
                            st.dataframe(explanation['contributions'], use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"説明生成中にエラーが発生しました: {str(e)}")
    
    # 予測信頼区間タブ
    with analysis_tabs[1]:
        st.subheader("📊 予測信頼区間")
        st.markdown("""
        予測の不確実性を考慮した信頼区間を表示します。
        - **50%信頼区間**: 予測値の50%がこの範囲に入る
        - **80%信頼区間**: 予測値の80%がこの範囲に入る
        - **95%信頼区間**: 予測値の95%がこの範囲に入る
        """)
        
        if feature_columns and df_original is not None:
            st.subheader("📝 物件情報を入力")
            input_data_interval = create_prediction_form(feature_columns, df_original, "interval")
            
            if st.button("📈 信頼区間付きで予測", key="predict_interval"):
                with st.spinner("信頼区間を計算中..."):
                    try:
                        df_input = pd.DataFrame([input_data_interval])
                        
                        preprocessor = st.session_state.get('preprocessor')
                        if preprocessor and not st.session_state.get('use_native_cat', False):
                            df_input_processed = preprocessor.transform_new_data(df_input)
                        else:
                            df_input_processed = df_input
                        
                        if not st.session_state.get('use_native_cat', False):
                            df_input_processed = df_input_processed.select_dtypes(include=[np.number])
                        
                        # 学習データを取得
                        X_train = df_processed.drop(columns=[target_column])
                        y_train = df_processed[target_column]
                        if not st.session_state.get('use_native_cat', False):
                            X_train = X_train.select_dtypes(include=[np.number])
                        
                        # 信頼区間計算器
                        interval_calc = PredictionInterval(trainer.model, X_train, y_train)
                        interval_calc.fit()
                        
                        result = interval_calc.predict_single_with_interval(df_input_processed)
                        
                        st.session_state['last_interval_result'] = result
                        st.session_state['last_input_data'] = df_input_processed
                        
                        # 予測結果表示
                        prediction = result['prediction']
                        intervals = result['intervals']
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 15px; padding: 25px; text-align: center; color: white; margin: 15px 0;">
                            <h2 style="margin: 0;">予測 {target_column}</h2>
                            <h1 style="margin: 10px 0; font-size: 3rem;">{prediction:,.2f}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 信頼区間表示
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "50%信頼区間",
                                f"{intervals['50%']['lower']:,.1f} 〜 {intervals['50%']['upper']:,.1f}"
                            )
                        with col2:
                            st.metric(
                                "80%信頼区間",
                                f"{intervals['80%']['lower']:,.1f} 〜 {intervals['80%']['upper']:,.1f}"
                            )
                        with col3:
                            st.metric(
                                "95%信頼区間",
                                f"{intervals['95%']['lower']:,.1f} 〜 {intervals['95%']['upper']:,.1f}"
                            )
                        
                        # 信頼区間プロット
                        fig_interval = interval_calc.plot_prediction_interval(result)
                        st.plotly_chart(fig_interval, use_container_width=True)
                        
                        show_success_message("信頼区間の計算が完了しました！")
                        
                    except Exception as e:
                        st.error(f"信頼区間計算中にエラーが発生しました: {str(e)}")
    
    # PDFレポートタブ
    with analysis_tabs[2]:
        st.subheader("📄 PDFレポート出力")
        st.markdown("""
        予測結果をPDFレポートとして出力できます。レポートには以下が含まれます：
        - 予測価格サマリー
        - 入力した物件情報
        - SHAP分析結果（価格算出根拠）
        - 信頼区間
        - モデル性能指標
        """)
        
        # 予測データがあるか確認
        has_prediction = (st.session_state.get('last_shap_explanation') or 
                         st.session_state.get('last_interval_result'))
        
        if not has_prediction:
            st.info("📝 先に「SHAP分析」または「予測信頼区間」タブで予測を実行してください。")
        else:
            st.success("✅ 予測データがあります。レポートを生成できます。")
            
            if st.button("📄 PDFレポートを生成", type="primary", key="generate_pdf"):
                with st.spinner("PDFを生成中..."):
                    try:
                        pdf_gen = PDFReportGenerator()
                        
                        # 予測結果を取得
                        shap_explanation = st.session_state.get('last_shap_explanation')
                        interval_result = st.session_state.get('last_interval_result')
                        input_data = st.session_state.get('last_input_data')
                        metrics = st.session_state.get('metrics')
                        
                        # 予測値を取得
                        if shap_explanation:
                            prediction_result = {'prediction': shap_explanation['prediction']}
                        elif interval_result:
                            prediction_result = {'prediction': interval_result['prediction']}
                        else:
                            prediction_result = {'prediction': 0}
                        
                        # PDF生成
                        pdf_bytes = pdf_gen.generate_report(
                            prediction_result=prediction_result,
                            shap_explanation=shap_explanation,
                            interval_result=interval_result,
                            model_metrics=metrics,
                            input_data=input_data,
                            target_column=target_column
                        )
                        
                        # ダウンロードボタン
                        st.download_button(
                            label="📥 PDFをダウンロード",
                            data=pdf_bytes,
                            file_name=f"prediction_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        
                        show_success_message("PDFレポートが生成されました！")
                        
                    except Exception as e:
                        st.error(f"PDF生成中にエラーが発生しました: {str(e)}")
    
    # What-if分析タブ
    with analysis_tabs[3]:
        st.subheader("🔄 What-if分析（シミュレーション）")
        st.markdown("""
        特徴量の値を変更した場合の予測価格への影響を分析します。
        - **感度分析**: 1つの特徴量を変化させた時の影響
        - **シナリオ比較**: 複数の条件を組み合わせた比較
        """)
        
        if feature_columns and df_original is not None:
            whatif_mode = st.radio(
                "分析モードを選択",
                ["📈 感度分析（1変数）", "📊 シナリオ比較（複数変数）"],
                horizontal=True
            )
            
            st.markdown("---")
            
            if whatif_mode == "📈 感度分析（1変数）":
                st.subheader("📝 ベース条件を入力")
                input_data_whatif = create_prediction_form(feature_columns, df_original, "whatif")
                
                st.markdown("---")
                
                # 分析対象の特徴量を選択
                numeric_features = [col for col in feature_columns 
                                   if df_original[col].dtype in ['int64', 'float64', 'int32', 'float32']]
                
                if numeric_features:
                    selected_feature = st.selectbox(
                        "📊 分析する特徴量を選択",
                        numeric_features
                    )
                    
                    if st.button("🔄 感度分析を実行", key="run_whatif"):
                        with st.spinner("分析中..."):
                            try:
                                df_base = pd.DataFrame([input_data_whatif])
                                
                                preprocessor = st.session_state.get('preprocessor')
                                if preprocessor and not st.session_state.get('use_native_cat', False):
                                    df_base_processed = preprocessor.transform_new_data(df_base)
                                else:
                                    df_base_processed = df_base
                                
                                if not st.session_state.get('use_native_cat', False):
                                    df_base_processed = df_base_processed.select_dtypes(include=[np.number])
                                
                                # What-if分析器
                                X_original = df_original.drop(columns=[target_column], errors='ignore')
                                whatif = WhatIfAnalyzer(trainer.model, feature_columns, X_original)
                                
                                # 感度分析実行
                                sensitivity_df = whatif.analyze_feature_impact(
                                    df_base_processed, 
                                    selected_feature,
                                    n_points=20
                                )
                                
                                # 結果表示
                                st.subheader(f"📊 {selected_feature} の感度分析結果")
                                fig_sensitivity = whatif.plot_feature_sensitivity(sensitivity_df, selected_feature)
                                st.plotly_chart(fig_sensitivity, use_container_width=True)
                                
                                # データテーブル
                                st.subheader("📋 詳細データ")
                                st.dataframe(sensitivity_df, use_container_width=True)
                                
                                show_success_message("感度分析が完了しました！")
                                
                            except Exception as e:
                                st.error(f"分析中にエラーが発生しました: {str(e)}")
                else:
                    st.warning("数値型の特徴量がありません。")
            
            else:  # シナリオ比較
                st.subheader("📝 ベース条件を入力")
                input_data_scenario = create_prediction_form(feature_columns, df_original, "scenario")
                
                st.markdown("---")
                st.subheader("📊 シナリオを定義")
                
                # シナリオ数を選択
                n_scenarios = st.number_input("シナリオ数", min_value=1, max_value=5, value=2)
                
                scenarios = {}
                numeric_features = [col for col in feature_columns 
                                   if df_original[col].dtype in ['int64', 'float64', 'int32', 'float32']]
                
                for i in range(int(n_scenarios)):
                    with st.expander(f"シナリオ {i+1}", expanded=True):
                        scenario_name = st.text_input(f"シナリオ名", value=f"シナリオ{i+1}", key=f"scenario_name_{i}")
                        
                        changes = {}
                        selected_features = st.multiselect(
                            "変更する特徴量",
                            numeric_features,
                            key=f"scenario_features_{i}"
                        )
                        
                        for feat in selected_features:
                            base_val = input_data_scenario.get(feat, df_original[feat].mean())
                            new_val = st.number_input(
                                f"{feat} の新しい値",
                                value=float(base_val),
                                key=f"scenario_{i}_{feat}"
                            )
                            changes[feat] = new_val
                        
                        if changes:
                            scenarios[scenario_name] = changes
                
                if st.button("📊 シナリオ比較を実行", key="run_scenario"):
                    if scenarios:
                        with st.spinner("分析中..."):
                            try:
                                df_base = pd.DataFrame([input_data_scenario])
                                
                                preprocessor = st.session_state.get('preprocessor')
                                if preprocessor and not st.session_state.get('use_native_cat', False):
                                    df_base_processed = preprocessor.transform_new_data(df_base)
                                else:
                                    df_base_processed = df_base
                                
                                if not st.session_state.get('use_native_cat', False):
                                    df_base_processed = df_base_processed.select_dtypes(include=[np.number])
                                
                                # What-if分析器
                                X_original = df_original.drop(columns=[target_column], errors='ignore')
                                whatif = WhatIfAnalyzer(trainer.model, feature_columns, X_original)
                                
                                # シナリオ比較実行
                                scenario_df = whatif.compare_scenarios(df_base_processed, scenarios)
                                
                                # 結果表示
                                st.subheader("📊 シナリオ比較結果")
                                fig_scenario = whatif.plot_scenario_comparison(scenario_df)
                                st.plotly_chart(fig_scenario, use_container_width=True)
                                
                                # データテーブル
                                st.subheader("📋 詳細データ")
                                st.dataframe(scenario_df, use_container_width=True)
                                
                                show_success_message("シナリオ比較が完了しました！")
                                
                            except Exception as e:
                                st.error(f"分析中にエラーが発生しました: {str(e)}")
                    else:
                        st.warning("少なくとも1つのシナリオを定義してください。")


def train_model(model_type, use_tuning, n_trials, use_cv, cv_folds, test_size,
                use_xgb_stack=True, use_lgb_stack=True, use_cat_stack=True,
                include_stacking=False):
    """モデル学習を実行"""
    
    df_processed = st.session_state['df_processed']
    target_column = st.session_state['target_column']
    
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    
    use_native_cat = st.session_state.get('use_native_cat', False)
    if not use_native_cat:
        X = X.select_dtypes(include=[np.number])
    
    st.session_state['feature_columns'] = X.columns.tolist()
    
    cat_features = None
    if use_native_cat:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            cat_features = [X.columns.get_loc(col) for col in cat_cols]
    
    if model_type == "全モデル比較":
        with st.spinner("全モデルを比較中..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            status_text.text("モデルを比較中...")
            progress_bar.progress(30)
            
            X_train_num = X_train.select_dtypes(include=[np.number])
            X_test_num = X_test.select_dtypes(include=[np.number])
            
            comparison_results = compare_models(
                X_train_num, X_test_num, y_train, y_test,
                include_stacking=include_stacking
            )
            
            progress_bar.progress(100)
            status_text.text("完了！")
            
            best_model_type = get_best_model(comparison_results)
            
            trainer = ModelTrainer()
            trainer.model = comparison_results[best_model_type]['model']
            trainer.model_type = best_model_type
            trainer.feature_importance = comparison_results[best_model_type]['feature_importance']
            trainer.y_pred = comparison_results[best_model_type]['predictions']
            trainer.y_test = y_test
            
            st.session_state['trainer'] = trainer
            st.session_state['metrics'] = comparison_results[best_model_type]['metrics']
            st.session_state['is_trained'] = True
            st.session_state['comparison_results'] = comparison_results
            
            show_success_message(f"比較完了！最良モデル: {best_model_type.upper()}")
            st.dataframe(comparison_results['comparison_df'], use_container_width=True)
    
    elif model_type == "スタッキング (アンサンブル)":
        with st.spinner("スタッキングモデルを学習中..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            X_train_num = X_train.select_dtypes(include=[np.number])
            X_test_num = X_test.select_dtypes(include=[np.number])
            
            status_text.text("スタッキングモデルを構築中...")
            progress_bar.progress(30)
            
            stacking_trainer = StackingTrainer()
            stacking_trainer.train(
                X_train_num, y_train,
                use_xgboost=use_xgb_stack,
                use_lightgbm=use_lgb_stack,
                use_catboost=use_cat_stack,
                cv_folds=cv_folds
            )
            
            progress_bar.progress(80)
            status_text.text("評価中...")
            
            metrics = stacking_trainer.evaluate(X_test_num, y_test)
            
            progress_bar.progress(100)
            status_text.text("完了！")
            
            st.session_state['trainer'] = stacking_trainer
            st.session_state['metrics'] = metrics
            st.session_state['is_trained'] = True
            st.session_state['model_type'] = 'stacking'
            
            show_success_message("スタッキングモデルの学習が完了しました！")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col3:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col4:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            
    else:
        trainer = ModelTrainer()
        
        model_map = {
            'XGBoost': 'xgboost',
            'LightGBM': 'lightgbm',
            'CatBoost': 'catboost'
        }
        selected_model = model_map[model_type]
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=test_size)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if use_tuning:
            status_text.text(f"ハイパーパラメータチューニング中... ({n_trials}試行)")
            
            X_train_num = X_train.select_dtypes(include=[np.number])
            
            def update_progress(progress):
                progress_bar.progress(int(progress * 50))
            
            tuning_results = trainer.tune_hyperparameters(
                X_train_num, y_train,
                model_type=selected_model,
                n_trials=n_trials,
                cv_folds=cv_folds,
                progress_callback=update_progress
            )
            
            st.write("最適パラメータ:", tuning_results['best_params'])
            progress_bar.progress(50)
        else:
            progress_bar.progress(25)
        
        status_text.text("モデルを学習中...")
        
        if selected_model != 'catboost' or not use_native_cat:
            X_train_fit = X_train.select_dtypes(include=[np.number])
            X_test_fit = X_test.select_dtypes(include=[np.number])
            trainer.train(X_train_fit, y_train, model_type=selected_model, use_default_params=not use_tuning)
        else:
            X_train_fit = X_train
            X_test_fit = X_test
            trainer.train(X_train_fit, y_train, model_type=selected_model, 
                         use_default_params=not use_tuning, cat_features=cat_features)
        
        progress_bar.progress(70)
        
        if use_cv:
            status_text.text("交差検証を実行中...")
            X_cv = X.select_dtypes(include=[np.number]) if selected_model != 'catboost' or not use_native_cat else X
            cv_scores = trainer.cross_validate(X_cv, y, cv_folds=cv_folds)
            st.write(f"CV R² Score: {cv_scores['r2_mean']:.4f} (±{cv_scores['r2_std']:.4f})")
        
        progress_bar.progress(90)
        
        status_text.text("モデルを評価中...")
        metrics = trainer.evaluate(X_test_fit, y_test)
        trainer.y_test = y_test
        
        progress_bar.progress(100)
        status_text.text("完了！")
        
        st.session_state['trainer'] = trainer
        st.session_state['metrics'] = metrics
        st.session_state['is_trained'] = True
        st.session_state['model_type'] = selected_model
        
        show_success_message(f"{model_type} モデルの学習が完了しました！")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with col3:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        with col4:
            st.metric("MAPE", f"{metrics['mape']:.2f}%")


def save_model():
    """モデルを保存"""
    trainer = st.session_state['trainer']
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = st.session_state.get('model_type', 'model')
    filename = f"model_{model_type}_{timestamp}.joblib"
    filepath = os.path.join('models', filename)
    
    try:
        trainer.save_model(filepath)
        show_success_message(f"モデルを保存しました: {filename}")
    except Exception as e:
        st.error(f"保存に失敗しました: {str(e)}")


def load_saved_model(uploaded_model):
    """保存されたモデルを読み込み"""
    try:
        import joblib
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
            tmp.write(uploaded_model.getvalue())
            tmp_path = tmp.name
        
        trainer = ModelTrainer()
        model_data = trainer.load_model(tmp_path)
        
        st.session_state['trainer'] = trainer
        st.session_state['metrics'] = model_data['metrics']
        st.session_state['is_trained'] = True
        st.session_state['model_type'] = model_data['model_type']
        
        show_success_message(f"モデルを読み込みました: {model_data['model_type']}")
        
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"読み込みに失敗しました: {str(e)}")


if __name__ == "__main__":
    main()
