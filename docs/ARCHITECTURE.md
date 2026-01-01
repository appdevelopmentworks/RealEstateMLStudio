# アーキテクチャ・技術仕様書

**プロジェクト名**: RealEstateMLStudio  
**バージョン**: 2.3  
**最終更新**: 2026-01-01

---

## 1. システムアーキテクチャ

### 1.1 全体構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI Layer                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  Data   │ │   EDA   │ │Training │ │  Eval   │ │ Predict │   │
│  │ Upload  │ │   Tab   │ │   Tab   │ │   Tab   │ │   Tab   │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │         │
│  ┌────┴───────────┴───────────┴───────────┴───────────┴────┐   │
│  │                      app.py (Controller)                 │   │
│  └─────────────────────────────┬───────────────────────────┘   │
└────────────────────────────────┼────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                     src/ (Business Logic)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │preprocessor  │  │   trainer    │  │  visualizer  │           │
│  │    .py       │  │     .py      │  │     .py      │           │
│  │              │  │              │  │              │           │
│  │-DataPreproc  │  │-ModelTrainer │  │-Visualizer   │           │
│  │ essor        │  │-StackingTr   │  │-plot_*       │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  analysis    │  │    utils     │                             │
│  │    .py       │  │     .py      │                             │
│  │              │  │              │                             │
│  │-SHAPAnalyzer │  │-load_css     │                             │
│  │-Prediction   │  │-create_*     │                             │
│  │ Interval     │  │-show_*       │                             │
│  │-WhatIfAnalyz │  │-init_session │                             │
│  │-PDFReport    │  │              │                             │
│  │-SimilarProp  │  └──────────────┘                             │
│  │-DataQuality  │                                               │
│  │-FeatureEng   │                                               │
│  │-PredHistory  │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                      External Libraries                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ XGBoost │ │LightGBM │ │CatBoost │ │  SHAP   │ │ Optuna  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Plotly  │ │ Pandas  │ │ NumPy   │ │sklearn  │ │  fpdf2  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 データフロー図

```
┌──────────┐     ┌──────────────┐     ┌───────────┐     ┌──────────┐
│  CSV     │────▶│ Preprocessor │────▶│  Trainer  │────▶│  Model   │
│  Upload  │     │              │     │           │     │ (joblib) │
└──────────┘     └──────────────┘     └───────────┘     └──────────┘
                        │                   │                 │
                        ▼                   ▼                 ▼
                 ┌──────────────┐    ┌───────────┐     ┌──────────┐
                 │ session_state│    │  Metrics  │     │ Predict  │
                 │ ['df']       │    │  CV Scores│     │ Function │
                 └──────────────┘    └───────────┘     └──────────┘
                                           │                 │
                                           ▼                 ▼
                                    ┌───────────┐     ┌──────────┐
                                    │Visualizer │     │ Analysis │
                                    │ (Plotly)  │     │(SHAP etc)│
                                    └───────────┘     └──────────┘
```

---

## 2. モジュール設計

### 2.1 モジュール一覧

| モジュール | ファイル | 責務 |
|-----------|----------|------|
| Controller | app.py | UI制御、タブ管理、セッション管理 |
| Preprocessor | src/preprocessor.py | データ前処理、変換 |
| Trainer | src/trainer.py | モデル学習、評価、チューニング |
| Visualizer | src/visualizer.py | グラフ生成、可視化 |
| Analysis | src/analysis.py | 高度分析機能 |
| Utils | src/utils.py | 共通ユーティリティ |

### 2.2 モジュール依存関係

```
app.py
  ├── src/preprocessor.py
  │     └── pandas, numpy, sklearn
  ├── src/trainer.py
  │     └── xgboost, lightgbm, catboost, optuna, sklearn
  ├── src/visualizer.py
  │     └── plotly, matplotlib, seaborn
  ├── src/analysis.py
  │     └── shap, fpdf2, sklearn
  └── src/utils.py
        └── streamlit, pandas
```

---

## 3. クラス設計

### 3.1 preprocessor.py

```python
class DataPreprocessor:
    """データ前処理クラス"""
    
    Attributes:
        label_encoders: dict    # カラム別LabelEncoder
        scaler: object          # StandardScaler/MinMaxScaler
        imputers: dict          # カラム別Imputer
        original_columns: list  # 元のカラム名
        numeric_columns: list   # 数値カラム
        categorical_columns: list  # カテゴリカラム
    
    Methods:
        analyze_data(df) -> dict
        detect_outliers_iqr(df, column, threshold) -> Series
        detect_outliers_zscore(df, column, threshold) -> Series
        handle_missing_values(df, strategy) -> DataFrame
        encode_categorical(df, columns) -> DataFrame
        scale_features(df, method, columns) -> DataFrame
        handle_outliers(df, columns, method, action) -> DataFrame
        auto_preprocess(df, target_column, **options) -> DataFrame
        transform_new_data(df) -> DataFrame
```

### 3.2 trainer.py

```python
class ModelTrainer:
    """モデルトレーニングクラス"""
    
    Attributes:
        model: object           # 学習済みモデル
        model_type: str         # 'xgboost', 'lightgbm', 'catboost'
        best_params: dict       # 最適パラメータ
        cv_scores: dict         # 交差検証結果
        feature_importance: Series  # 特徴量重要度
        metrics: dict           # 評価指標
        X_train, X_test, y_train, y_test, y_pred: array
        cat_features: list      # CatBoost用
    
    Methods:
        prepare_data(X, y, test_size, random_state) -> tuple
        tune_hyperparameters(X, y, model_type, n_trials, cv_folds) -> dict
        train(X_train, y_train, model_type, params) -> object
        cross_validate(X, y, cv_folds) -> dict
        evaluate(X_test, y_test) -> dict
        predict(X) -> ndarray
        save_model(filepath) -> str
        load_model(filepath) -> dict


class StackingTrainer:
    """スタッキングアンサンブルトレーナー"""
    
    Attributes:
        model: StackingRegressor
        base_models: dict
        meta_model: ElasticNet
        feature_importance: Series
        metrics: dict
    
    Methods:
        train(X_train, y_train, use_xgboost, use_lightgbm, use_catboost) -> object
        evaluate(X_test, y_test) -> dict
        predict(X) -> ndarray
        save_model(filepath) -> str


# ユーティリティ関数
def compare_models(X_train, X_test, y_train, y_test, **params) -> dict
def get_best_model(comparison_results, metric) -> str
```

### 3.3 visualizer.py

```python
class Visualizer:
    """可視化クラス"""
    
    Attributes:
        theme: str  # 'plotly_white'
    
    Methods:
        plot_actual_vs_predicted(y_true, y_pred, title) -> Figure
        plot_residuals(y_true, y_pred) -> Figure
        plot_feature_importance(feature_importance, top_n, title) -> Figure
        plot_metrics_dashboard(metrics, cv_scores) -> Figure
        plot_cv_results(cv_scores) -> Figure
        plot_model_comparison(comparison_df) -> Figure
        plot_model_comparison_radar(comparison_df) -> Figure
        plot_prediction_distribution(y_true, y_pred) -> Figure
        plot_eda_dashboard(df, target_column) -> Figure
        plot_shap_summary(shap_values, feature_names) -> Figure
        plot_learning_curve(train_sizes, train_scores, val_scores) -> Figure


# カラーパレット定数
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    ...
}

MODEL_COLORS = {
    'XGBoost': '#1f77b4',
    'LightGBM': '#ff7f0e',
    ...
}
```

### 3.4 analysis.py

```python
class SHAPAnalyzer:
    """SHAP分析クラス"""
    
    Attributes:
        model: object
        X_train: DataFrame
        explainer: shap.Explainer
        shap_values: ndarray
        X_sample: DataFrame
    
    Methods:
        calculate_shap_values(X, max_samples) -> ndarray
        get_feature_importance() -> Series
        explain_prediction(X_single) -> dict
        plot_summary(top_n) -> Figure
        plot_waterfall(explanation) -> Figure
        plot_force_single(explanation) -> Figure


class PredictionInterval:
    """予測信頼区間クラス"""
    
    Attributes:
        model: object
        X_train, y_train: array
        residuals: ndarray
        residual_std: float
        residual_percentiles: dict
    
    Methods:
        fit() -> self
        predict_with_interval(X, confidence) -> dict
        predict_single_with_interval(X_single, confidence_levels) -> dict
        plot_prediction_interval(result) -> Figure


class WhatIfAnalyzer:
    """What-if分析クラス"""
    
    Attributes:
        model: object
        feature_columns: list
        X_original: DataFrame
    
    Methods:
        analyze_feature_impact(X_base, feature, values, n_points) -> DataFrame
        compare_scenarios(X_base, scenarios) -> DataFrame
        plot_feature_sensitivity(sensitivity_df, feature) -> Figure
        plot_scenario_comparison(scenario_df) -> Figure


class PDFReportGenerator:
    """PDFレポート生成クラス"""
    
    Methods:
        generate_report(prediction_result, shap_explanation, 
                       interval_result, model_metrics, 
                       input_data, target_column) -> bytes
        get_download_link(pdf_bytes, filename) -> str


class SimilarPropertyFinder:
    """類似物件検索クラス"""
    
    Attributes:
        df: DataFrame
        feature_columns: list
        target_column: str
        scaler: StandardScaler
        scaled_features: ndarray
        numeric_cols: list
    
    Methods:
        fit() -> self
        find_similar(X_query, n_neighbors, method) -> DataFrame
        plot_similar_properties(query_prediction, similar_df) -> Figure


class DataQualityChecker:
    """データ品質チェッククラス"""
    
    Attributes:
        df: DataFrame
        report: dict
    
    Methods:
        check_all() -> dict
        _check_duplicates() -> dict
        _check_missing() -> dict
        _check_outliers() -> dict
        _check_type_issues() -> list
        _check_value_ranges() -> dict
        _check_high_correlations(threshold) -> list
        get_summary_dataframe() -> DataFrame
        plot_quality_overview() -> Figure


class FeatureEngineer:
    """特徴量自動生成クラス"""
    
    Attributes:
        df: DataFrame
        target_column: str
        new_features: list
        feature_info: list
    
    Methods:
        generate_all(**options) -> DataFrame
        _create_interactions(df, columns) -> DataFrame
        _create_polynomial(df, columns) -> DataFrame
        _create_ratios(df, columns) -> DataFrame
        _create_binning(df, columns, n_bins) -> DataFrame
        get_feature_info() -> DataFrame


class PredictionHistory:
    """予測履歴管理クラス"""
    
    Attributes:
        history: list
    
    Methods:
        add_prediction(input_data, prediction, **kwargs) -> dict
        get_history_dataframe() -> DataFrame
        clear_history() -> None
        export_to_csv() -> str
        export_to_json() -> str
        plot_history() -> Figure
        get_statistics() -> dict
```

---

## 4. セッション状態設計

### 4.1 セッション変数一覧

| 変数名 | 型 | 説明 |
|--------|-----|------|
| df | DataFrame | 読み込んだ生データ |
| df_processed | DataFrame | 前処理後データ |
| df_original | DataFrame | 元データのコピー |
| preprocessor | DataPreprocessor | 前処理インスタンス |
| trainer | ModelTrainer/StackingTrainer | トレーナーインスタンス |
| model | object | 学習済みモデル |
| model_type | str | モデルタイプ |
| target_column | str | ターゲット列名 |
| feature_columns | list | 特徴量列リスト |
| metrics | dict | 評価指標 |
| cv_scores | dict | 交差検証結果 |
| is_trained | bool | 学習完了フラグ |
| comparison_results | dict | モデル比較結果 |
| use_native_cat | bool | CatBoostネイティブ処理フラグ |
| sample_data_name | str | サンプルデータ名 |
| shap_analyzer | SHAPAnalyzer | SHAP分析器 |
| last_shap_explanation | dict | 最新SHAP説明 |
| last_interval_result | dict | 最新信頼区間結果 |
| last_input_data | DataFrame | 最新入力データ |
| last_similar_df | DataFrame | 最新類似物件結果 |
| last_similar_prediction | float | 最新類似物件予測 |
| quality_report | dict | 品質チェック結果 |
| quality_checker | DataQualityChecker | 品質チェッカー |
| df_engineered | DataFrame | 特徴量生成後データ |
| feature_engineer | FeatureEngineer | 特徴量エンジニア |
| prediction_history | PredictionHistory | 予測履歴 |

### 4.2 セッション状態遷移

```
[初期状態]
    │
    ▼ データアップロード
[df設定済み]
    │
    ▼ 前処理実行
[df_processed設定済み, preprocessor設定済み]
    │
    ▼ モデル学習
[is_trained=True, trainer設定済み, metrics設定済み]
    │
    ├─▶ 予測実行 ─▶ [last_*系変数更新]
    │
    └─▶ 詳細分析 ─▶ [shap_analyzer, last_*系変数更新]
```

---

## 5. 技術スタック詳細

### 5.1 主要ライブラリ

| カテゴリ | ライブラリ | バージョン | 用途 |
|---------|-----------|-----------|------|
| UI | streamlit | ≥1.28.0 | Webアプリフレームワーク |
| データ処理 | pandas | ≥2.0.0 | データフレーム操作 |
| 数値計算 | numpy | ≥1.24.0 | 数値演算 |
| ML基盤 | scikit-learn | ≥1.3.0 | 前処理、評価、スタッキング |
| MLモデル | xgboost | ≥2.0.0 | 勾配ブースティング |
| MLモデル | lightgbm | ≥4.0.0 | 高速ブースティング |
| MLモデル | catboost | ≥1.2.0 | カテゴリ対応ブースティング |
| チューニング | optuna | ≥3.4.0 | ハイパーパラメータ最適化 |
| 説明可能AI | shap | ≥0.43.0 | 特徴量貢献度分析 |
| 可視化 | plotly | ≥5.18.0 | インタラクティブグラフ |
| 可視化 | matplotlib | ≥3.8.0 | 静的グラフ |
| 可視化 | seaborn | ≥0.13.0 | 統計グラフ |
| 日本語フォント | japanize-matplotlib | ≥1.1.3 | 日本語対応 |
| モデル保存 | joblib | ≥1.3.0 | シリアライズ |
| PDF生成 | fpdf2 | ≥2.7.0 | レポート出力 |
| 画像処理 | Pillow | ≥10.0.0 | 画像操作 |
| Excel出力 | openpyxl | ≥3.1.0 | Excel対応 |
| 画像出力 | kaleido | ≥0.2.1 | Plotly画像出力 |

### 5.2 アルゴリズム詳細

#### XGBoost
```python
# デフォルトパラメータ
{
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'objective': 'reg:squarederror'
}

# Optunaチューニング範囲
{
    'n_estimators': (100, 1000),
    'max_depth': (3, 12),
    'learning_rate': (0.01, 0.3, log=True),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'gamma': (0, 0.5),
    'reg_alpha': (1e-8, 10.0, log=True),
    'reg_lambda': (1e-8, 10.0, log=True),
    'min_child_weight': (1, 10)
}
```

#### LightGBM
```python
# デフォルトパラメータ
{
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'num_leaves': 31,
    'objective': 'regression'
}

# Optunaチューニング範囲
{
    'n_estimators': (100, 1000),
    'max_depth': (3, 12),
    'learning_rate': (0.01, 0.3, log=True),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (1e-8, 10.0, log=True),
    'reg_lambda': (1e-8, 10.0, log=True),
    'min_child_samples': (5, 100),
    'num_leaves': (20, 300)
}
```

#### CatBoost
```python
# デフォルトパラメータ
{
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'bagging_temperature': 0.5,
    'random_strength': 0.5,
    'loss_function': 'RMSE'
}

# Optunaチューニング範囲
{
    'iterations': (100, 1000),
    'depth': (4, 10),
    'learning_rate': (0.01, 0.3, log=True),
    'l2_leaf_reg': (1e-8, 10.0, log=True),
    'bagging_temperature': (0, 1),
    'random_strength': (0, 1),
    'border_count': (32, 255)
}
```

#### スタッキング
```python
# 構成
StackingRegressor(
    estimators=[
        ('xgboost', XGBRegressor(...)),
        ('lightgbm', LGBMRegressor(...)),
        ('catboost', CatBoostRegressor(...))
    ],
    final_estimator=ElasticNet(alpha=0.1, l1_ratio=0.5),
    cv=5,
    n_jobs=-1
)
```

---

## 6. パフォーマンス設計

### 6.1 最適化ポイント

| 箇所 | 最適化 |
|------|--------|
| SHAP計算 | サンプル数を500に制限 |
| チューニング | Optunaのtrial callback で進捗表示 |
| モデル学習 | n_jobs=-1 で並列化 |
| データ変換 | vectorized operations使用 |
| グラフ | Plotlyのlazy loading |

### 6.2 メモリ管理

```python
# 大規模データ対策
if len(X) > max_samples:
    X_sample = X.sample(n=max_samples, random_state=42)

# SHAP計算時のバックグラウンドデータ制限
X_background = shap.sample(X_sample, min(100, len(X_sample)))
```

---

## 7. エラーハンドリング設計

### 7.1 エラー処理パターン

```python
# 基本パターン
try:
    # 処理
except Exception as e:
    st.error(f"処理中にエラーが発生しました: {str(e)}")

# SHAP計算のフォールバック
try:
    # TreeExplainer
except Exception:
    try:
        # KernelExplainer
    except Exception:
        try:
            # Permutation Explainer
        except Exception as e:
            raise ValueError(f"SHAP計算失敗: {str(e)}")
```

### 7.2 バリデーション

```python
# モデル学習前チェック
if self.model is None:
    raise ValueError("モデルが学習されていません")

# データ存在チェック
if st.session_state.get('df') is None:
    show_warning_message("先にデータをアップロードしてください")
    return

# 学習完了チェック
if not st.session_state.get('is_trained', False):
    show_warning_message("先にモデルを学習してください")
    return
```

---

## 8. 拡張ポイント

### 8.1 新モデル追加

```python
# trainer.py に以下を追加

# 1. 目的関数を追加
def _objective_newmodel(self, trial, X, y, cv_folds):
    params = {
        'param1': trial.suggest_int('param1', 1, 100),
        ...
    }
    model = NewModelRegressor(**params)
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    return -cv_scores.mean()

# 2. trainメソッドに分岐追加
elif model_type == 'newmodel':
    self.model = NewModelRegressor(**params)
    self.model.fit(X_train, y_train)

# 3. compare_modelsに追加
```

### 8.2 新分析機能追加

```python
# analysis.py に新クラス追加

class NewAnalyzer:
    def __init__(self, model, X_train):
        ...
    
    def analyze(self, X):
        ...
    
    def plot_result(self, result):
        ...

# app.py の詳細分析タブに追加
with analysis_tabs[N]:
    st.subheader("新機能")
    # UI実装
```

---

## 9. デプロイメント

### 9.1 ローカル実行

```bash
# インストール
pip install -r requirements.txt

# 起動
streamlit run app.py

# ポート指定
streamlit run app.py --server.port 8080
```

### 9.2 Streamlit Cloud

```
# 必要ファイル
- app.py
- requirements.txt
- src/
- images/

# 設定
Runtime: Python 3.9+
Main file path: app.py
```

### 9.3 Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```
