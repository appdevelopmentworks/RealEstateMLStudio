# RealEstateMLStudio ドキュメント

## 📚 ドキュメント一覧

| ファイル | 内容 | 対象者 |
|---------|------|--------|
| [REQUIREMENTS.md](./REQUIREMENTS.md) | 要求定義書 | PM / 企画者 |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | アーキテクチャ・技術仕様書 | 開発者 |
| [API_REFERENCE.md](./API_REFERENCE.md) | クラス・関数リファレンス | 開発者 |
| [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) | 実装ガイド（バイブコーディング用） | AI / 開発者 |
| [ROADMAP.md](./ROADMAP.md) | ロードマップ・開発計画 | PM / 開発者 |
| [CHANGELOG.md](./CHANGELOG.md) | 変更履歴 | 全員 |

---

## 🏠 プロジェクト概要

**RealEstateMLStudio** は、不動産価格を機械学習で予測するStreamlitベースのWebアプリケーションです。

### 主な特徴

- **マルチモデル対応**: XGBoost, LightGBM, CatBoost, スタッキング
- **自動ハイパーパラメータチューニング**: Optunaによる最適化
- **説明可能AI**: SHAP分析による予測根拠の可視化
- **予測信頼区間**: 不確実性を考慮した予測
- **PDFレポート出力**: 分析結果のレポート化
- **What-if分析**: シミュレーション機能
- **ユーティリティ機能**: 類似物件検索、データ品質チェック、特徴量自動生成

### 技術スタック

```
Frontend:  Streamlit
ML:        XGBoost, LightGBM, CatBoost, scikit-learn
Analysis:  SHAP, Optuna
Viz:       Plotly, Matplotlib, Seaborn
PDF:       fpdf2
```

### ディレクトリ構造

```
RealEstateMLStudio/
├── app.py                 # メインアプリケーション
├── requirements.txt       # 依存パッケージ
├── README.md             # プロジェクトREADME
├── src/                  # ソースコード
│   ├── __init__.py
│   ├── preprocessor.py   # データ前処理
│   ├── trainer.py        # モデル学習
│   ├── visualizer.py     # 可視化
│   ├── analysis.py       # 高度分析（SHAP, What-if等）
│   └── utils.py          # ユーティリティ
├── data/                 # データディレクトリ
├── models/               # 保存モデル
├── reports/              # 出力レポート
├── images/               # 画像リソース
└── docs/                 # ドキュメント
```

---

## 🚀 クイックスタート

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd RealEstateMLStudio

# 依存パッケージをインストール
pip install -r requirements.txt

# アプリを起動
streamlit run app.py
```

### 必要環境

- Python 3.9+
- 4GB以上のRAM推奨
- モダンブラウザ（Chrome, Firefox, Edge）

---

## 📖 使い方ガイド

### Step 1: データアップロード
CSVファイルをアップロード、またはサンプルデータを選択

### Step 2: データ分析 (EDA)
基本統計量、欠損値、相関分析を確認

### Step 3: モデル学習
- アルゴリズムを選択（XGBoost/LightGBM/CatBoost/スタッキング/全比較）
- オプションでハイパーパラメータチューニング
- 前処理を実行後、学習開始

### Step 4: 評価結果
- 評価指標ダッシュボード（R², RMSE, MAE, MAPE）
- 実測vs予測プロット
- 残差分析
- 特徴量重要度

### Step 5: 予測実行
- CSVファイルからの一括予測
- 手入力による単一予測

### Step 6: 詳細分析
- SHAP分析（予測の説明）
- 予測信頼区間
- PDFレポート出力
- What-if分析（シミュレーション）

### Step 7: ユーティリティ
- 類似物件検索
- データ品質チェック
- 特徴量自動生成
- 予測履歴管理

---

## 📄 ライセンス

MIT License

## 👤 作成者

RealEstateMLStudio Team
