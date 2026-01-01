# API リファレンス

**プロジェクト名**: RealEstateMLStudio  
**バージョン**: 2.3  
**最終更新**: 2026-01-01

---

## 目次

1. [preprocessor.py](#1-preprocessorpy)
2. [trainer.py](#2-trainerpy)
3. [visualizer.py](#3-visualizerpy)
4. [analysis.py](#4-analysispy)
5. [utils.py](#5-utilspy)
6. [app.py](#6-apppy)

---

## 1. preprocessor.py

### DataPreprocessor

データ前処理を行うクラス。

#### コンストラクタ

```python
DataPreprocessor()
```

#### 属性

| 属性 | 型 | 説明 |
|------|-----|------|
| label_encoders | dict | カラム名をキーとしたLabelEncoderの辞書 |
| scaler | StandardScaler/MinMaxScaler | スケーラーインスタンス |
| imputers | dict | カラム名をキーとしたImputerの辞書 |
| original_columns | list | 元のカラム名リスト |
| numeric_columns | list | 数値カラム名リスト |
| categorical_columns | list | カテゴリカラム名リスト |

#### メソッド

##### analyze_data

```python
def analyze_data(self, df: pd.DataFrame) -> dict
```

データの基本分析を行う。

**引数:**
- `df`: 分析対象のDataFrame

**戻り値:**
```python
{
    'shape': (rows, columns),
    'columns': [...],
    'dtypes': {...},
    'missing_values': {...},
    'missing_percentage': {...},
    'numeric_columns': [...],
    'categorical_columns': [...],
    'duplicates': int,
    'memory_usage': float  # MB
}
```

##### detect_outliers_iqr

```python
def detect_outliers_iqr(self, df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series
```

IQR法による異常値検出。

**引数:**
- `df`: 対象DataFrame
- `column`: 対象カラム名
- `threshold`: IQR倍率（デフォルト: 1.5）

**戻り値:** 異常値フラグのSeries（True=異常値）

##### detect_outliers_zscore

```python
def detect_outliers_zscore(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series
```

Z-score法による異常値検出。

**引数:**
- `df`: 対象DataFrame
- `column`: 対象カラム名
- `threshold`: Zスコア閾値（デフォルト: 3.0）

**戻り値:** 異常値フラグのSeries

##### handle_missing_values

```python
def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame
```

欠損値の処理。

**引数:**
- `df`: 対象DataFrame
- `strategy`: 処理戦略
  - `'auto'`: 数値→中央値、カテゴリ→最頻値
  - `'drop'`: 欠損行を削除
  - `'mean'`: 平均値で補完
  - `'median'`: 中央値で補完
  - `'mode'`: 最頻値で補完

**戻り値:** 処理後のDataFrame

##### encode_categorical

```python
def encode_categorical(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame
```

カテゴリ変数のラベルエンコーディング。

**引数:**
- `df`: 対象DataFrame
- `columns`: エンコード対象カラム（Noneで自動検出）

**戻り値:** エンコード後のDataFrame

##### scale_features

```python
def scale_features(self, df: pd.DataFrame, method: str = 'standard', columns: list = None) -> pd.DataFrame
```

特徴量のスケーリング。

**引数:**
- `df`: 対象DataFrame
- `method`: スケーリング方法
  - `'standard'`: StandardScaler
  - `'minmax'`: MinMaxScaler
- `columns`: 対象カラム（Noneで数値列全て）

**戻り値:** スケーリング後のDataFrame

##### handle_outliers

```python
def handle_outliers(self, df: pd.DataFrame, columns: list = None, method: str = 'iqr', action: str = 'clip') -> pd.DataFrame
```

異常値の処理。

**引数:**
- `df`: 対象DataFrame
- `columns`: 対象カラム（Noneで数値列全て）
- `method`: 検出方法（`'iqr'` or `'zscore'`）
- `action`: 処理方法
  - `'remove'`: 行を削除
  - `'clip'`: 上下限でクリップ

**戻り値:** 処理後のDataFrame

##### auto_preprocess

```python
def auto_preprocess(self, df: pd.DataFrame, target_column: str = None,
                    handle_missing: bool = True,
                    encode_cat: bool = True,
                    handle_outliers_flag: bool = False,
                    scale: bool = False) -> pd.DataFrame
```

自動前処理パイプライン。

**引数:**
- `df`: 対象DataFrame
- `target_column`: ターゲット列名（前処理から除外）
- `handle_missing`: 欠損値処理フラグ
- `encode_cat`: エンコーディングフラグ
- `handle_outliers_flag`: 異常値処理フラグ
- `scale`: スケーリングフラグ

**戻り値:** 前処理後のDataFrame

##### transform_new_data

```python
def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame
```

学習時と同じ変換を新しいデータに適用。

**引数:**
- `df`: 変換対象のDataFrame

**戻り値:** 変換後のDataFrame

---

### 関数

##### get_data_summary

```python
def get_data_summary(df: pd.DataFrame) -> dict
```

データの要約統計を取得。

---

## 2. trainer.py

### ModelTrainer

モデルトレーニングクラス。

#### コンストラクタ

```python
ModelTrainer()
```

#### 属性

| 属性 | 型 | 説明 |
|------|-----|------|
| model | object | 学習済みモデル |
| model_type | str | モデルタイプ（xgboost/lightgbm/catboost） |
| best_params | dict | 最適パラメータ |
| cv_scores | dict | 交差検証結果 |
| feature_importance | pd.Series | 特徴量重要度 |
| metrics | dict | 評価指標 |
| X_train, X_test | pd.DataFrame | 学習/テストデータ |
| y_train, y_test | pd.Series | 学習/テストラベル |
| y_pred | np.ndarray | 予測値 |
| cat_features | list | CatBoost用カテゴリ特徴量インデックス |

#### メソッド

##### prepare_data

```python
def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple
```

データの分割。

**戻り値:** `(X_train, X_test, y_train, y_test)`

##### tune_hyperparameters

```python
def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                        model_type: str = 'xgboost',
                        n_trials: int = 50,
                        cv_folds: int = 5,
                        progress_callback=None) -> dict
```

Optunaによるハイパーパラメータチューニング。

**引数:**
- `X`: 特徴量DataFrame
- `y`: ターゲットSeries
- `model_type`: モデルタイプ
- `n_trials`: 試行回数
- `cv_folds`: 交差検証Fold数
- `progress_callback`: 進捗コールバック関数

**戻り値:**
```python
{
    'best_params': {...},
    'best_score': float,
    'n_trials': int,
    'optimization_history': DataFrame
}
```

##### train

```python
def train(self, X_train: pd.DataFrame, y_train: pd.Series,
          model_type: str = 'xgboost',
          params: dict = None,
          use_default_params: bool = False,
          cat_features: list = None) -> object
```

モデルの学習。

**引数:**
- `X_train`: 学習用特徴量
- `y_train`: 学習用ターゲット
- `model_type`: `'xgboost'`, `'lightgbm'`, `'catboost'`
- `params`: カスタムパラメータ（Noneでbest_params使用）
- `use_default_params`: デフォルトパラメータ使用フラグ
- `cat_features`: CatBoost用カテゴリ特徴量インデックス

**戻り値:** 学習済みモデル

##### cross_validate

```python
def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict
```

交差検証の実行。

**戻り値:**
```python
{
    'rmse_mean': float, 'rmse_std': float,
    'mae_mean': float, 'mae_std': float,
    'r2_mean': float, 'r2_std': float,
    'mape_mean': float, 'mape_std': float,
    'fold_results': {...},
    'fold_predictions': [...]
}
```

##### evaluate

```python
def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict
```

モデルの評価。

**戻り値:**
```python
{
    'rmse': float,
    'mae': float,
    'r2': float,
    'mape': float,
    'model_score': float,
    'residuals': {
        'mean': float, 'std': float,
        'min': float, 'max': float
    }
}
```

##### predict

```python
def predict(self, X: pd.DataFrame) -> np.ndarray
```

予測の実行。

##### save_model / load_model

```python
def save_model(self, filepath: str) -> str
def load_model(self, filepath: str) -> dict
```

モデルの保存/読み込み。

---

### StackingTrainer

スタッキングアンサンブルトレーナー。

#### メソッド

##### train

```python
def train(self, X_train: pd.DataFrame, y_train: pd.Series,
          use_xgboost: bool = True,
          use_lightgbm: bool = True,
          use_catboost: bool = True,
          cv_folds: int = 5) -> object
```

スタッキングモデルの学習。

---

### 関数

##### compare_models

```python
def compare_models(X_train, X_test, y_train, y_test,
                   params_xgb=None, params_lgb=None, params_cat=None,
                   include_stacking: bool = False) -> dict
```

複数モデルの比較。

**戻り値:**
```python
{
    'xgboost': {'metrics': {...}, 'model': ..., 'feature_importance': ..., 'predictions': ...},
    'lightgbm': {...},
    'catboost': {...},
    'stacking': {...},  # include_stacking=Trueの場合
    'comparison_df': DataFrame
}
```

##### get_best_model

```python
def get_best_model(comparison_results: dict, metric: str = 'r2') -> str
```

比較結果から最良モデルを選択。

---

## 3. visualizer.py

### Visualizer

可視化クラス。

#### メソッド

##### plot_actual_vs_predicted

```python
def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "実測値 vs 予測値") -> go.Figure
```

実測値 vs 予測値の散布図。残差の大きさで色分け。

##### plot_residuals

```python
def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure
```

残差分析ダッシュボード（4パネル）。

##### plot_feature_importance

```python
def plot_feature_importance(self, feature_importance: pd.Series,
                            top_n: int = 20,
                            title: str = "特徴量重要度") -> go.Figure
```

特徴量重要度の水平バーチャート。

##### plot_metrics_dashboard

```python
def plot_metrics_dashboard(self, metrics: dict, cv_scores: dict = None) -> go.Figure
```

評価指標ゲージダッシュボード（RMSE, R², MAE, MAPE）。

##### plot_cv_results

```python
def plot_cv_results(self, cv_scores: dict) -> go.Figure
```

交差検証結果の可視化。

##### plot_model_comparison

```python
def plot_model_comparison(self, comparison_df: pd.DataFrame) -> go.Figure
```

モデル比較バーチャート。

##### plot_model_comparison_radar

```python
def plot_model_comparison_radar(self, comparison_df: pd.DataFrame) -> go.Figure
```

モデル比較レーダーチャート。

##### plot_prediction_distribution

```python
def plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure
```

予測値と実測値の分布比較ヒストグラム。

##### plot_eda_dashboard

```python
def plot_eda_dashboard(self, df: pd.DataFrame, target_column: str = None) -> go.Figure
```

相関行列ヒートマップ。

---

### 定数

```python
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb33',
    'info': '#17a2b8',
    'purple': '#9467bd',
    'gradient': ['#667eea', '#764ba2'],
    'palette': px.colors.qualitative.Set2
}

MODEL_COLORS = {
    'XGBoost': '#1f77b4',
    'LightGBM': '#ff7f0e',
    'CatBoost': '#2ca02c',
    'Stacking': '#9467bd'
}
```

---

## 4. analysis.py

### SHAPAnalyzer

SHAP分析クラス。

#### コンストラクタ

```python
SHAPAnalyzer(model, X_train: pd.DataFrame)
```

#### メソッド

##### calculate_shap_values

```python
def calculate_shap_values(self, X: pd.DataFrame = None, max_samples: int = 500) -> np.ndarray
```

SHAP値を計算。モデルタイプに応じて適切なExplainerを自動選択。

##### get_feature_importance

```python
def get_feature_importance(self) -> pd.Series
```

SHAP値ベースの特徴量重要度を取得。

##### explain_prediction

```python
def explain_prediction(self, X_single: pd.DataFrame) -> dict
```

単一予測の説明を生成。

**戻り値:**
```python
{
    'base_value': float,
    'shap_values': ndarray,
    'contributions': DataFrame,  # 特徴量, 値, SHAP値, 影響
    'prediction': float
}
```

##### plot_summary / plot_waterfall / plot_force_single

```python
def plot_summary(self, top_n: int = 15) -> go.Figure
def plot_waterfall(self, explanation: dict) -> go.Figure
def plot_force_single(self, explanation: dict) -> go.Figure
```

SHAP可視化メソッド。

---

### PredictionInterval

予測信頼区間クラス。

#### コンストラクタ

```python
PredictionInterval(model, X_train: pd.DataFrame, y_train: pd.Series)
```

#### メソッド

##### fit

```python
def fit(self) -> self
```

残差分布を学習。

##### predict_with_interval

```python
def predict_with_interval(self, X: pd.DataFrame, confidence: float = 0.95) -> dict
```

**戻り値:**
```python
{
    'predictions': ndarray,
    'lower': ndarray,
    'upper': ndarray,
    'confidence': float,
    'margin': float
}
```

##### predict_single_with_interval

```python
def predict_single_with_interval(self, X_single: pd.DataFrame,
                                  confidence_levels: list = [0.5, 0.8, 0.95]) -> dict
```

**戻り値:**
```python
{
    'prediction': float,
    'intervals': {
        '50%': {'lower': float, 'upper': float, 'margin': float},
        '80%': {...},
        '95%': {...}
    },
    'residual_std': float
}
```

##### plot_prediction_interval

```python
def plot_prediction_interval(self, result: dict) -> go.Figure
```

信頼区間の可視化。

---

### WhatIfAnalyzer

What-if分析クラス。

#### コンストラクタ

```python
WhatIfAnalyzer(model, feature_columns: list, X_original: pd.DataFrame)
```

#### メソッド

##### analyze_feature_impact

```python
def analyze_feature_impact(self, X_base: pd.DataFrame, feature: str,
                           values: list = None, n_points: int = 20) -> pd.DataFrame
```

特定の特徴量を変化させた時の予測変化を分析。

##### compare_scenarios

```python
def compare_scenarios(self, X_base: pd.DataFrame, scenarios: dict) -> pd.DataFrame
```

複数シナリオの比較。

**引数例:**
```python
scenarios = {
    'リフォーム後': {'築年数': 5, '面積_m2': 80},
    '駅近物件': {'駅徒歩分': 3}
}
```

##### plot_feature_sensitivity / plot_scenario_comparison

プロット生成メソッド。

---

### PDFReportGenerator

PDFレポート生成クラス。

#### メソッド

##### generate_report

```python
def generate_report(self,
                   prediction_result: dict,
                   shap_explanation: dict = None,
                   interval_result: dict = None,
                   model_metrics: dict = None,
                   input_data: pd.DataFrame = None,
                   target_column: str = "価格") -> bytes
```

PDFレポートをバイト列として生成。

---

### SimilarPropertyFinder

類似物件検索クラス。

#### コンストラクタ

```python
SimilarPropertyFinder(df: pd.DataFrame, feature_columns: list, target_column: str)
```

#### メソッド

##### fit

```python
def fit(self) -> self
```

特徴量をスケーリング。

##### find_similar

```python
def find_similar(self, X_query: pd.DataFrame, n_neighbors: int = 5,
                method: str = 'euclidean') -> pd.DataFrame
```

類似物件を検索。

**引数:**
- `method`: `'euclidean'`（ユークリッド距離）or `'cosine'`（コサイン類似度）

**戻り値:** 類似度スコア・順位付きDataFrame

##### plot_similar_properties

```python
def plot_similar_properties(self, query_prediction: float, similar_df: pd.DataFrame) -> go.Figure
```

類似物件の比較プロット。

---

### DataQualityChecker

データ品質チェッククラス。

#### コンストラクタ

```python
DataQualityChecker(df: pd.DataFrame)
```

#### メソッド

##### check_all

```python
def check_all(self) -> dict
```

全チェックを実行。

**戻り値:**
```python
{
    'duplicates': {...},
    'missing': {...},
    'outliers': {...},
    'type_issues': [...],
    'value_ranges': {...},
    'correlations': [...],
    'summary': {
        'total_rows': int,
        'total_columns': int,
        'total_issues': int,
        'quality_score': int  # 0-100
    }
}
```

##### get_summary_dataframe

```python
def get_summary_dataframe(self) -> pd.DataFrame
```

サマリーをDataFrameで取得。

##### plot_quality_overview

```python
def plot_quality_overview(self) -> go.Figure
```

品質スコアゲージチャート。

---

### FeatureEngineer

特徴量自動生成クラス。

#### コンストラクタ

```python
FeatureEngineer(df: pd.DataFrame, target_column: str)
```

#### メソッド

##### generate_all

```python
def generate_all(self,
                include_interactions: bool = True,
                include_polynomial: bool = True,
                include_ratios: bool = True,
                include_binning: bool = True) -> pd.DataFrame
```

全ての特徴量を生成。

##### get_feature_info

```python
def get_feature_info(self) -> pd.DataFrame
```

生成した特徴量の情報を取得。

---

### PredictionHistory

予測履歴管理クラス。

#### メソッド

##### add_prediction

```python
def add_prediction(self, input_data: dict, prediction: float,
                  confidence_interval: dict = None,
                  model_type: str = None,
                  similar_properties: pd.DataFrame = None) -> dict
```

予測を履歴に追加。

##### get_history_dataframe / clear_history

履歴管理メソッド。

##### export_to_csv / export_to_json

エクスポートメソッド。

##### plot_history / get_statistics

可視化・統計メソッド。

---

## 5. utils.py

### 関数

##### load_css

```python
def load_css() -> None
```

カスタムCSSを読み込み。

##### create_header

```python
def create_header(title: str, subtitle: str = "") -> None
```

ヘッダーを作成。

##### create_metric_card

```python
def create_metric_card(label: str, value: str, icon: str = "📊") -> str
```

メトリックカードのHTMLを生成。

##### format_number

```python
def format_number(value: float, precision: int = 4) -> str
```

数値をフォーマット（K/M表記対応）。

##### get_data_info

```python
def get_data_info(df: pd.DataFrame) -> dict
```

DataFrameの情報を取得。

##### display_dataframe_info

```python
def display_dataframe_info(df: pd.DataFrame) -> None
```

DataFrameの情報を4カラムで表示。

##### show_success_message / show_warning_message

```python
def show_success_message(message: str) -> None
def show_warning_message(message: str) -> None
```

メッセージ表示。

##### show_info_card

```python
def show_info_card(title: str, content: str) -> None
```

情報カードを表示。

##### init_session_state

```python
def init_session_state() -> None
```

セッション状態を初期化。

---

## 6. app.py

### 主要関数

##### main

```python
def main() -> None
```

メインアプリケーション。7タブ構成のUI。

##### load_sample_data

```python
def load_sample_data(dataset_name: str) -> pd.DataFrame
```

サンプルデータをロード。

**引数:**
- `dataset_name`: `'california'`, `'tokyo_sample'`, `'boston_simple'`

##### create_prediction_form

```python
def create_prediction_form(feature_columns: list, df_original: pd.DataFrame, key_prefix: str = "input") -> dict
```

予測用の入力フォームを作成。

##### train_model

```python
def train_model(model_type, use_tuning, n_trials, use_cv, cv_folds, test_size, ...) -> None
```

モデル学習を実行。

##### save_model / load_saved_model

```python
def save_model() -> None
def load_saved_model(uploaded_model) -> None
```

モデルの保存/読み込み。

### タブレンダリング関数

```python
def render_data_upload_tab() -> None
def render_eda_tab() -> None
def render_training_tab(...) -> None
def render_evaluation_tab() -> None
def render_prediction_tab() -> None
def render_advanced_analysis_tab() -> None
def render_utility_tab() -> None
```

各タブのUI実装。
