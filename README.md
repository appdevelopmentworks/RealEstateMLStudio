# 🏠 RealEstateMLStudio v2.0

**XGBoost / LightGBM / CatBoost / スタッキング による高精度不動産価格予測システム**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 概要

RealEstateMLStudioは、機械学習を活用した不動産価格予測のための包括的なツールです。
直感的なUIで、データのアップロードからモデル学習、予測実行まで一貫して行えます。

## ✨ 主な機能

### 🤖 機械学習モデル（4種類）
| モデル | 特徴 |
|--------|------|
| **XGBoost** | 高精度な勾配ブースティング。バランスの取れた性能 |
| **LightGBM** | 高速・軽量。大規模データに最適 |
| **CatBoost** | カテゴリ変数に強い。過学習しにくい。前処理不要 |
| **スタッキング** | 複数モデルを組み合わせ最高精度を実現 |

### ⚙️ 自動チューニング
- **Optuna**による自動ハイパーパラメータ最適化
- **交差検証**によるモデルの汎化性能評価
- カスタマイズ可能な試行回数とFold数

### 🔧 データ前処理
- 欠損値の自動検出・補完
- カテゴリ変数の自動エンコーディング
- CatBoostネイティブカテゴリ処理（エンコード不要）
- 異常値の検出・処理（IQR法/Z-score法）
- 特徴量スケーリング

### 📊 豪華な可視化
- **インタラクティブなPlotlyグラフ**
- 実測値 vs 予測値の散布図
- 残差分析ダッシュボード
- 特徴量重要度チャート
- 評価指標ゲージ
- 相関マトリックスヒートマップ
- モデル比較チャート（3モデル+スタッキング対応）

### 💾 モデル管理
- 学習済みモデルの保存
- モデルの読み込み・再利用

## 🚀 インストール

```bash
# リポジトリのクローン
cd RealEstateMLStudio

# 依存パッケージのインストール
pip install -r requirements.txt

# アプリケーションの起動
streamlit run app.py
```

## 📁 プロジェクト構造

```
RealEstateMLStudio/
├── app.py                 # メインアプリケーション
├── requirements.txt       # 依存パッケージ
├── README.md             # このファイル
├── src/
│   ├── __init__.py
│   ├── preprocessor.py   # データ前処理モジュール
│   ├── trainer.py        # モデル学習モジュール（4モデル対応）
│   ├── visualizer.py     # 可視化モジュール
│   └── utils.py          # ユーティリティ関数
├── models/               # 保存されたモデル
├── data/                 # データファイル
├── images/               # 画像ファイル
└── reports/              # レポート出力
```

## 📖 使い方

### Step 1: データアップロード
CSV形式の学習データをドラッグ＆ドロップ

### Step 2: データ分析 (EDA)
- 基本統計量の確認
- 欠損値の分析
- 相関マトリックスの確認

### Step 3: モデル学習
1. ターゲット列（予測したい列）を選択
2. アルゴリズムを選択
   - XGBoost
   - LightGBM
   - CatBoost（カテゴリ変数が多い場合に推奨）
   - スタッキング（最高精度を狙う場合）
   - 全モデル比較
3. 前処理を実行
4. 学習開始

### Step 4: 評価結果の確認
- 評価指標ダッシュボード
- 予測精度のグラフ
- 特徴量重要度
- モデル比較（全モデル比較時）

### Step 5: 予測実行
新しいデータをアップロードして予測

## 📊 評価指標

| 指標 | 説明 |
|------|------|
| RMSE | 二乗平均平方根誤差（低いほど良い） |
| MAE | 平均絶対誤差（低いほど良い） |
| R² | 決定係数（1に近いほど良い） |
| MAPE | 平均絶対パーセント誤差（低いほど良い） |

## 🏆 モデル選択ガイド

| 状況 | 推奨モデル |
|------|-----------|
| カテゴリ変数が多い | CatBoost |
| データが大規模 | LightGBM |
| 最高精度を狙う | スタッキング |
| バランス重視 | XGBoost |
| 最適モデルを探す | 全モデル比較 |

## 🛠️ 技術スタック

- **Frontend**: Streamlit
- **Machine Learning**: XGBoost, LightGBM, CatBoost, scikit-learn
- **Optimization**: Optuna
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Data Processing**: Pandas, NumPy

## 📝 ライセンス

MIT License

## 🤝 コントリビューション

プルリクエストは大歓迎です！

---

Made with ❤️ using Streamlit & Python
