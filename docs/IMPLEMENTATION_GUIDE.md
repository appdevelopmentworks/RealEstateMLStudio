# 実装ガイド（バイブコーディング用）

**プロジェクト名**: RealEstateMLStudio  
**バージョン**: 2.3  
**最終更新**: 2026-01-01

---

このドキュメントは、AIアシスタント（Claude等）がこのプロジェクトをゼロから再実装するための詳細ガイドです。

---

## 1. プロジェクト概要

### 1.1 一言で説明

「不動産価格を機械学習で予測し、SHAP分析で説明可能にするStreamlitアプリ」

### 1.2 コア機能

1. **データ処理**: CSV読み込み → 前処理 → 特徴量/ターゲット分離
2. **モデル学習**: XGBoost/LightGBM/CatBoost/スタッキング
3. **評価可視化**: R², RMSE等の指標をPlotlyダッシュボードで表示
4. **予測実行**: 新データに対する価格予測
5. **説明可能AI**: SHAP分析による予測根拠の可視化
6. **ユーティリティ**: 類似物件検索、データ品質チェック等

---

## 2. 実装手順

### Phase 1: プロジェクト構造作成

```bash
mkdir RealEstateMLStudio
cd RealEstateMLStudio
mkdir src data models reports images docs
touch app.py requirements.txt README.md
touch src/__init__.py src/preprocessor.py src/trainer.py src/visualizer.py src/analysis.py src/utils.py
```

### Phase 2: 依存パッケージ（requirements.txt）

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.4.0
shap>=0.43.0
plotly>=5.18.0
seaborn>=0.13.0
matplotlib>=3.8.0
japanize-matplotlib>=1.1.3
Pillow>=10.0.0
joblib>=1.3.0
openpyxl>=3.1.0
fpdf2>=2.7.0
kaleido>=0.2.1
```

### Phase 3: 各モジュール実装

以下の順序で実装を進める：

1. `src/utils.py` - 共通ユーティリティ
2. `src/preprocessor.py` - データ前処理
3. `src/trainer.py` - モデル学習
4. `src/visualizer.py` - 可視化
5. `src/analysis.py` - 高度分析
6. `app.py` - メインアプリ

---

## 3. モジュール別実装ガイド

### 3.1 utils.py

**目的**: 共通関数、CSS、ヘルパー

```python
# 必須インポート
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os

# 実装すべき関数
def load_css():
    """グラデーション背景、カード、ボタンスタイルのCSS"""
    css = """
    <style>
        .main-header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 15px; color: white; text-align: center;
        }
        .metric-card { ... }
        .success-box { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); ... }
        .warning-box { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); ... }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def init_session_state():
    """セッション変数の初期化"""
    defaults = {
        'df': None, 'df_processed': None, 'preprocessor': None,
        'trainer': None, 'model': None, 'is_trained': False, ...
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_success_message(message: str):
    st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)

def show_warning_message(message: str):
    st.markdown(f'<div class="warning-box">⚠️ {message}</div>', unsafe_allow_html=True)

def display_dataframe_info(df):
    """4列でrows, columns, missing, memoryを表示"""
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("行数", f"{len(df):,}")
    ...
```

### 3.2 preprocessor.py

**目的**: データ前処理パイプライン

```python
# 必須インポート
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.imputers = {}
        self.numeric_columns = None
        self.categorical_columns = None
    
    def auto_preprocess(self, df, target_column=None, handle_missing=True, 
                        encode_cat=True, handle_outliers_flag=False, scale=False):
        """
        自動前処理の主要ロジック:
        1. ターゲット列を一時分離
        2. 数値/カテゴリ列を自動識別
        3. 欠損値処理（数値→中央値、カテゴリ→最頻値）
        4. カテゴリ変数をLabelEncoder
        5. 異常値処理（IQR法でclip）
        6. スケーリング（StandardScaler）
        7. ターゲット列を戻す
        """
        df_processed = df.copy()
        
        # ターゲット分離
        target = None
        if target_column and target_column in df_processed.columns:
            target = df_processed[target_column].copy()
            df_processed = df_processed.drop(columns=[target_column])
        
        # 列タイプ識別
        self.numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # 各処理を実行...
        
        # ターゲット復元
        if target is not None:
            df_processed[target_column] = target.values
        
        return df_processed
    
    def transform_new_data(self, df):
        """学習時と同じ変換を適用（予測時用）"""
        df_processed = df.copy()
        # imputers, label_encoders, scalerを順に適用
        return df_processed
```

### 3.3 trainer.py

**目的**: MLモデルの学習・評価

```python
# 必須インポート
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.best_params = None
        self.feature_importance = None
        self.metrics = {}
        self.y_pred = None
        self.y_test = None
    
    def tune_hyperparameters(self, X, y, model_type='xgboost', n_trials=50, cv_folds=5):
        """
        Optunaによるチューニング:
        1. TPESamplerでstudy作成
        2. 目的関数で各パラメータをtrial.suggest_*で探索
        3. cross_val_scoreでMSEを評価
        4. 最適パラメータをself.best_paramsに保存
        """
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    ...
                }
                model = XGBRegressor(**params)
            ...
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
            return -scores.mean()
        
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        return {'best_params': self.best_params, 'best_score': study.best_value}
    
    def train(self, X_train, y_train, model_type='xgboost', params=None):
        """
        モデル学習:
        1. model_typeに応じてモデルインスタンス作成
        2. params指定なければbest_paramsまたはデフォルト使用
        3. fit実行
        4. feature_importanceを保存
        """
        if model_type == 'xgboost':
            self.model = XGBRegressor(**params)
        elif model_type == 'lightgbm':
            self.model = LGBMRegressor(**params)
        elif model_type == 'catboost':
            self.model = CatBoostRegressor(**params, verbose=False)
        
        self.model.fit(X_train, y_train)
        self.feature_importance = pd.Series(
            self.model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        return self.model
    
    def evaluate(self, X_test, y_test):
        """RMSE, MAE, R², MAPEを計算"""
        self.y_pred = self.model.predict(X_test)
        self.y_test = y_test
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, self.y_pred)),
            'mae': mean_absolute_error(y_test, self.y_pred),
            'r2': r2_score(y_test, self.y_pred),
            'mape': mean_absolute_percentage_error(y_test, self.y_pred) * 100
        }
        return self.metrics

class StackingTrainer:
    """XGBoost + LightGBM + CatBoost をElasticNetでスタッキング"""
    def train(self, X_train, y_train, use_xgboost=True, use_lightgbm=True, use_catboost=True):
        estimators = []
        if use_xgboost:
            estimators.append(('xgboost', XGBRegressor(...)))
        if use_lightgbm:
            estimators.append(('lightgbm', LGBMRegressor(...)))
        if use_catboost:
            estimators.append(('catboost', CatBoostRegressor(...)))
        
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=ElasticNet(alpha=0.1, l1_ratio=0.5),
            cv=5, n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        return self.model

def compare_models(X_train, X_test, y_train, y_test, include_stacking=False):
    """3モデル（+スタッキング）を比較してDataFrame返却"""
    results = {}
    # 各モデルを学習・評価
    # comparison_df = pd.DataFrame({...}, index=['RMSE', 'MAE', 'R²', 'MAPE'])
    return results
```

### 3.4 visualizer.py

**目的**: Plotlyによるグラフ生成

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    ...
}

class Visualizer:
    def __init__(self):
        self.theme = 'plotly_white'
    
    def plot_actual_vs_predicted(self, y_true, y_pred, title="実測値 vs 予測値"):
        """
        散布図:
        - 残差の大きさでカラースケール
        - 45度線（完全予測線）
        - 回帰線
        """
        fig = go.Figure()
        residuals = y_true - y_pred
        
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode='markers',
            marker=dict(color=np.abs(residuals), colorscale='RdYlGn_r', showscale=True),
            hovertemplate="実測: %{x}<br>予測: %{y}"
        ))
        
        # 45度線
        fig.add_trace(go.Scatter(
            x=[y_true.min(), y_true.max()],
            y=[y_true.min(), y_true.max()],
            mode='lines', line=dict(color='red', dash='dash'),
            name='完全予測線'
        ))
        
        fig.update_layout(template=self.theme, ...)
        return fig
    
    def plot_metrics_dashboard(self, metrics, cv_scores=None):
        """
        4つのゲージチャート:
        - RMSE（低いほど良い）
        - R²（1に近いほど良い、色付きステップ）
        - MAE
        - MAPE（%表示）
        """
        fig = make_subplots(rows=2, cols=2, specs=[[{"type": "indicator"}]*2]*2)
        
        # RMSE
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['rmse'],
            title={"text": "RMSE<br><span style='font-size:11px;color:gray'>低いほど良い</span>"},
            gauge=dict(axis=dict(range=[0, metrics['rmse']*2]), ...)
        ), row=1, col=1)
        
        # R² (ステップカラー)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['r2'],
            gauge=dict(
                axis=dict(range=[0, 1]),
                steps=[
                    dict(range=[0, 0.5], color="#ffebee"),
                    dict(range=[0.5, 0.7], color="#fff3e0"),
                    dict(range=[0.7, 0.9], color="#e8f5e9"),
                    dict(range=[0.9, 1], color="#c8e6c9"),
                ]
            )
        ), row=1, col=2)
        
        ...
        return fig
    
    def plot_feature_importance(self, feature_importance, top_n=20):
        """水平バーチャート、Viridisカラースケール"""
        fi = feature_importance.head(top_n).sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation='h',
            marker=dict(color=px.colors.sample_colorscale('Viridis', len(fi)))
        ))
        return fig
```

### 3.5 analysis.py

**目的**: SHAP分析、信頼区間、What-if分析、PDF生成、ユーティリティ

```python
import shap
import numpy as np
import pandas as pd
from scipy import stats
from fpdf import FPDF

class SHAPAnalyzer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
    
    def calculate_shap_values(self, X=None, max_samples=500):
        """
        モデルタイプを判定して適切なExplainerを使用:
        - XGBoost/LightGBM/CatBoost → TreeExplainer
        - Stacking → KernelExplainer
        - その他 → Permutation Explainer（フォールバック）
        """
        if X is None:
            X = self.X_train
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)
        
        model_type = type(self.model).__name__
        
        try:
            if 'XGB' in model_type or 'LGBM' in model_type or 'CatBoost' in model_type:
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X)
            else:
                # KernelExplainerまたはPermutationExplainerにフォールバック
                ...
        except Exception:
            # 最終フォールバック
            ...
        
        self.X_sample = X
        return self.shap_values
    
    def explain_prediction(self, X_single):
        """
        単一予測の説明:
        - base_value（期待値）
        - 各特徴量のSHAP値
        - 貢献度テーブル（正負、影響の方向）
        """
        shap_values_single = self.explainer.shap_values(X_single)[0]
        base_value = self.explainer.expected_value
        
        contributions = pd.DataFrame({
            '特徴量': X_single.columns,
            '値': X_single.values[0],
            'SHAP値': shap_values_single,
            '影響': ['↑ 価格上昇' if v > 0 else '↓ 価格下落' for v in shap_values_single]
        }).sort_values('SHAP値', key=abs, ascending=False)
        
        return {
            'base_value': base_value,
            'shap_values': shap_values_single,
            'contributions': contributions,
            'prediction': base_value + shap_values_single.sum()
        }

class PredictionInterval:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.residual_std = None
    
    def fit(self):
        """残差の標準偏差を計算"""
        y_pred = self.model.predict(self.X_train)
        residuals = self.y_train.values - y_pred
        self.residual_std = np.std(residuals)
        return self
    
    def predict_single_with_interval(self, X_single, confidence_levels=[0.5, 0.8, 0.95]):
        """
        信頼区間計算:
        - 各信頼水準に対応するz値を取得
        - margin = z * residual_std
        - lower/upper = prediction ± margin
        """
        prediction = self.model.predict(X_single)[0]
        intervals = {}
        for conf in confidence_levels:
            z = stats.norm.ppf(1 - (1 - conf) / 2)
            margin = z * self.residual_std
            intervals[f'{int(conf*100)}%'] = {
                'lower': prediction - margin,
                'upper': prediction + margin,
                'margin': margin
            }
        return {'prediction': prediction, 'intervals': intervals}

class WhatIfAnalyzer:
    def analyze_feature_impact(self, X_base, feature, n_points=20):
        """
        感度分析:
        1. 特徴量の範囲を取得
        2. n_points個の値を生成
        3. 各値でX_baseを変更して予測
        4. 変化量・変化率を計算
        """
        values = np.linspace(self.X_original[feature].min(), 
                            self.X_original[feature].max(), n_points)
        base_pred = self.model.predict(X_base)[0]
        
        results = []
        for val in values:
            X_modified = X_base.copy()
            X_modified[feature] = val
            new_pred = self.model.predict(X_modified)[0]
            results.append({
                feature: val,
                '予測価格': new_pred,
                '変化量': new_pred - base_pred,
                '変化率(%)': (new_pred - base_pred) / base_pred * 100
            })
        return pd.DataFrame(results)

class PDFReportGenerator:
    def generate_report(self, prediction_result, shap_explanation=None, ...):
        """
        fpdf2でPDF生成:
        1. ヘッダー（タイトル、日時）
        2. 予測サマリー
        3. 入力データテーブル
        4. SHAP分析結果（正負の要因）
        5. モデル性能
        6. 免責事項
        """
        class PDF(FPDF):
            def header(self): ...
            def footer(self): ...
        
        pdf = PDF()
        pdf.add_page()
        # 各セクションを追加...
        return bytes(pdf.output())

class SimilarPropertyFinder:
    """
    類似物件検索:
    - StandardScalerで正規化
    - euclidean_distances または cosine_similarity
    - 上位N件を返却
    """

class DataQualityChecker:
    """
    データ品質チェック:
    - 重複行、欠損値、異常値（IQR）
    - 型の問題、高相関ペア
    - 品質スコア計算（100 - issues * 2）
    """

class FeatureEngineer:
    """
    特徴量自動生成:
    - 交互作用: col1 * col2
    - 多項式: col^2, sqrt(col)
    - 比率: col1 / col2
    - ビニング: pd.qcut
    """

class PredictionHistory:
    """
    予測履歴管理:
    - リストで履歴保持
    - DataFrame変換
    - CSV/JSONエクスポート
    - 統計計算
    """
```

### 3.6 app.py

**目的**: メインStreamlitアプリケーション

```python
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessor import DataPreprocessor
from src.trainer import ModelTrainer, StackingTrainer, compare_models
from src.visualizer import Visualizer
from src.analysis import SHAPAnalyzer, PredictionInterval, WhatIfAnalyzer, PDFReportGenerator, ...
from src.utils import load_css, init_session_state, show_success_message, ...

# ページ設定
st.set_page_config(
    page_title="RealEstateMLStudio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()
init_session_state()
viz = Visualizer()

def main():
    # ヘッダー画像 or テキストヘッダー
    if os.path.exists("images/Appbanner.png"):
        st.image("images/Appbanner.png", use_container_width=True)
    else:
        create_header("RealEstateMLStudio", "...")
    
    # サイドバー
    with st.sidebar:
        st.title("設定パネル")
        model_type = st.selectbox("アルゴリズム選択", 
            ["XGBoost", "LightGBM", "CatBoost", "スタッキング", "全モデル比較"])
        use_tuning = st.checkbox("ハイパーパラメータチューニング")
        if use_tuning:
            n_trials = st.slider("試行回数", 10, 200, 50)
        ...
    
    # 7タブ構成
    tabs = st.tabs([
        "📤 データアップロード", "🔍 EDA", "🎯 モデル学習",
        "📊 評価結果", "🔮 予測実行", "🔬 詳細分析", "🛠️ ユーティリティ"
    ])
    
    with tabs[0]:
        render_data_upload_tab()
    with tabs[1]:
        render_eda_tab()
    # ...

def render_data_upload_tab():
    st.header("Step 1: データのアップロード")
    
    uploaded_file = st.file_uploader("学習データをアップロード", type=['csv'])
    
    # サンプルデータボタン
    if st.button("カリフォルニア住宅データ"):
        df = load_sample_data("california")
        st.session_state['df'] = df
        st.rerun()
    
    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        display_dataframe_info(df)
        st.dataframe(df.head(20))

def render_training_tab(model_type, use_tuning, ...):
    st.header("Step 3: モデル学習")
    
    df = st.session_state.get('df')
    target_column = st.selectbox("ターゲット列", df.columns.tolist())
    
    if st.button("前処理を実行"):
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.auto_preprocess(df, target_column, ...)
        st.session_state['df_processed'] = df_processed
        st.session_state['preprocessor'] = preprocessor
    
    if st.button("学習開始"):
        train_model(model_type, use_tuning, ...)

def train_model(model_type, use_tuning, ...):
    df_processed = st.session_state['df_processed']
    target_column = st.session_state['target_column']
    
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    X = X.select_dtypes(include=[np.number])
    
    st.session_state['feature_columns'] = X.columns.tolist()
    
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    if use_tuning:
        trainer.tune_hyperparameters(X_train, y_train, model_type, n_trials)
    
    trainer.train(X_train, y_train, model_type)
    metrics = trainer.evaluate(X_test, y_test)
    
    st.session_state['trainer'] = trainer
    st.session_state['metrics'] = metrics
    st.session_state['is_trained'] = True

def render_advanced_analysis_tab():
    """SHAP, 信頼区間, PDF, What-ifの4サブタブ"""
    tabs = st.tabs(["SHAP分析", "予測信頼区間", "PDFレポート", "What-if分析"])
    
    with tabs[0]:  # SHAP
        if st.button("SHAP分析を実行"):
            shap_analyzer = SHAPAnalyzer(trainer.model, X_train)
            shap_analyzer.calculate_shap_values()
            fig = shap_analyzer.plot_summary()
            st.plotly_chart(fig)
    
    with tabs[1]:  # 信頼区間
        # 入力フォーム
        # predict_single_with_interval
        # plot_prediction_interval
    
    # ...

def render_utility_tab():
    """類似物件, 品質チェック, 特徴量生成, 履歴の4サブタブ"""
    tabs = st.tabs(["類似物件検索", "データ品質チェック", "特徴量自動生成", "予測履歴管理"])
    # 各タブの実装

if __name__ == "__main__":
    main()
```

---

## 4. 重要な実装パターン

### 4.1 セッション状態の活用

```python
# 初期化
if 'key' not in st.session_state:
    st.session_state['key'] = default_value

# 保存
st.session_state['trainer'] = trainer

# 取得
trainer = st.session_state.get('trainer')

# 存在チェック
if st.session_state.get('is_trained', False):
    ...
```

### 4.2 エラーハンドリング

```python
try:
    result = some_operation()
except Exception as e:
    st.error(f"エラーが発生しました: {str(e)}")
```

### 4.3 プログレス表示

```python
progress_bar = st.progress(0)
status_text = st.empty()

for i in range(100):
    progress_bar.progress(i)
    status_text.text(f"処理中... {i}%")

progress_bar.progress(100)
status_text.text("完了！")
```

### 4.4 動的フォーム生成

```python
def create_prediction_form(feature_columns, df_original, key_prefix):
    input_data = {}
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(feature_columns):
        if df_original[col].dtype == 'object':
            # カテゴリ → selectbox
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.selectbox(col, df_original[col].unique())
        else:
            # 数値 → number_input
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.number_input(col, value=df_original[col].mean())
    
    return input_data
```

---

## 5. テスト・デバッグ

### 5.1 動作確認手順

```bash
# 起動
streamlit run app.py

# ブラウザで確認
# 1. サンプルデータ読み込み
# 2. EDAタブで統計確認
# 3. XGBoostでモデル学習
# 4. 評価結果確認
# 5. 手入力で予測
# 6. SHAP分析実行
```

### 5.2 よくあるエラーと対処

| エラー | 原因 | 対処 |
|--------|------|------|
| KeyError: 'df' | データ未読み込み | セッション状態チェック追加 |
| SHAP計算失敗 | モデル非対応 | フォールバックExplainer使用 |
| 予測時エラー | 列不一致 | transform_new_data使用 |

---

## 6. 拡張のヒント

### 6.1 新モデル追加

1. `trainer.py`の`_objective_xxx`関数を追加
2. `train`メソッドに分岐追加
3. サイドバーの選択肢に追加

### 6.2 新分析機能追加

1. `analysis.py`に新クラス作成
2. `app.py`の詳細分析タブにサブタブ追加
3. セッション状態に結果を保存

### 6.3 UI改善

1. `utils.py`のCSSを編集
2. カラーパレットを`visualizer.py`で調整
3. レイアウトを`st.columns`で調整
