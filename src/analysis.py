"""
高度分析モジュール - SHAP分析、予測信頼区間、PDFレポート、What-if分析
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """SHAP分析クラス"""
    
    def __init__(self, model, X_train: pd.DataFrame):
        """
        Parameters:
        -----------
        model : 学習済みモデル
        X_train : 学習データ（SHAP計算用）
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        
    def calculate_shap_values(self, X: pd.DataFrame = None, max_samples: int = 500):
        """SHAP値を計算"""
        if X is None:
            X = self.X_train
        
        # サンプル数が多い場合は制限
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X.copy()
        
        # モデルタイプを判定して適切なExplainerを使用
        model_type = type(self.model).__name__
        
        try:
            if 'XGB' in model_type or 'LGBM' in model_type or 'LightGBM' in model_type:
                # XGBoost/LightGBMはTreeExplainerを使用
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X_sample)
            elif 'CatBoost' in model_type:
                # CatBoostはTreeExplainerを使用
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(X_sample)
            elif 'Stacking' in model_type:
                # スタッキングモデルはKernelExplainerを使用
                X_background = shap.sample(X_sample, min(100, len(X_sample)))
                self.explainer = shap.KernelExplainer(self.model.predict, X_background)
                self.shap_values = self.explainer.shap_values(X_sample.iloc[:min(100, len(X_sample))])
                X_sample = X_sample.iloc[:min(100, len(X_sample))]
            else:
                # その他のモデルはKernelExplainerを使用
                X_background = shap.sample(X_sample, min(100, len(X_sample)))
                self.explainer = shap.KernelExplainer(self.model.predict, X_background)
                self.shap_values = self.explainer.shap_values(X_sample.iloc[:min(100, len(X_sample))])
                X_sample = X_sample.iloc[:min(100, len(X_sample))]
        except Exception as e:
            # フォールバック: Permutation Explainerを使用
            try:
                X_background = X_sample.iloc[:min(50, len(X_sample))]
                self.explainer = shap.Explainer(self.model.predict, X_background)
                shap_result = self.explainer(X_sample.iloc[:min(200, len(X_sample))])
                self.shap_values = shap_result.values
                X_sample = X_sample.iloc[:min(200, len(X_sample))]
            except Exception as e2:
                raise ValueError(f"SHAP値の計算に失敗しました: {str(e2)}")
        
        self.X_sample = X_sample
        return self.shap_values
    
    def get_feature_importance(self) -> pd.Series:
        """SHAP値ベースの特徴量重要度を取得"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        importance = np.abs(self.shap_values).mean(axis=0)
        return pd.Series(importance, index=self.X_sample.columns).sort_values(ascending=False)
    
    def explain_prediction(self, X_single: pd.DataFrame) -> dict:
        """単一予測の説明を生成"""
        if self.explainer is None:
            self.calculate_shap_values()
        
        # 単一サンプルのSHAP値を計算
        model_type = type(self.model).__name__
        
        try:
            if hasattr(self.explainer, 'shap_values'):
                shap_values_single = self.explainer.shap_values(X_single)
            else:
                shap_result = self.explainer(X_single)
                shap_values_single = shap_result.values
        except Exception:
            # フォールバック
            X_background = self.X_sample.iloc[:min(50, len(self.X_sample))]
            temp_explainer = shap.Explainer(self.model.predict, X_background)
            shap_result = temp_explainer(X_single)
            shap_values_single = shap_result.values
        
        if len(shap_values_single.shape) > 1:
            shap_values_single = shap_values_single[0]
        
        # 基準値（期待値）
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0] if len(base_value) > 0 else float(base_value)
            base_value = float(base_value)
        else:
            base_value = float(self.model.predict(self.X_train).mean())
        
        # 各特徴量の貢献度をDataFrameに
        contributions = pd.DataFrame({
            '特徴量': X_single.columns,
            '値': X_single.values[0],
            'SHAP値': shap_values_single,
            '影響': ['↑ 価格上昇' if v > 0 else '↓ 価格下落' for v in shap_values_single]
        }).sort_values('SHAP値', key=abs, ascending=False)
        
        prediction = base_value + shap_values_single.sum()
        
        return {
            'base_value': base_value,
            'shap_values': shap_values_single,
            'contributions': contributions,
            'prediction': prediction
        }
    
    def plot_summary(self, top_n: int = 15) -> go.Figure:
        """SHAP Summary Plot（棒グラフ版）"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        importance = self.get_feature_importance().head(top_n)
        
        fig = go.Figure(go.Bar(
            x=importance.values[::-1],
            y=importance.index[::-1],
            orientation='h',
            marker=dict(
                color=importance.values[::-1],
                colorscale='RdBu_r',
                colorbar=dict(title="重要度")
            ),
            text=[f"{v:.4f}" for v in importance.values[::-1]],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(text="<b>SHAP特徴量重要度</b>", font=dict(size=20)),
            xaxis_title="平均 |SHAP値|",
            yaxis_title="特徴量",
            template='plotly_white',
            height=max(400, top_n * 30),
            margin=dict(l=150)
        )
        
        return fig
    
    def plot_waterfall(self, explanation: dict, feature_names: list = None) -> go.Figure:
        """ウォーターフォールチャート（予測の内訳）"""
        contributions = explanation['contributions']
        base_value = explanation['base_value']
        
        # 上位の貢献要因を表示
        top_positive = contributions[contributions['SHAP値'] > 0].head(5)
        top_negative = contributions[contributions['SHAP値'] < 0].head(5)
        display_df = pd.concat([top_positive, top_negative]).sort_values('SHAP値', ascending=False)
        
        fig = go.Figure()
        
        cumsum = base_value
        y_positions = []
        colors = []
        texts = []
        
        # 基準値
        y_positions.append("基準値")
        colors.append('#808080')
        texts.append(f"{base_value:.2f}")
        
        for _, row in display_df.iterrows():
            feature_label = f"{row['特徴量']}={row['値']:.2f}" if isinstance(row['値'], float) else f"{row['特徴量']}={row['値']}"
            y_positions.append(feature_label)
            colors.append('#2ca02c' if row['SHAP値'] > 0 else '#d62728')
            texts.append(f"{'+' if row['SHAP値'] > 0 else ''}{row['SHAP値']:.2f}")
        
        # 最終予測値
        y_positions.append("予測値")
        colors.append('#1f77b4')
        texts.append(f"{explanation['prediction']:.2f}")
        
        # 値を計算
        values = [base_value]
        for _, row in display_df.iterrows():
            values.append(row['SHAP値'])
        values.append(explanation['prediction'])
        
        fig = go.Figure(go.Waterfall(
            orientation="h",
            y=y_positions,
            x=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#d62728"}},
            increasing={"marker": {"color": "#2ca02c"}},
            totals={"marker": {"color": "#1f77b4"}},
            text=texts,
            textposition="outside"
        ))
        
        fig.update_layout(
            title=dict(text="<b>予測値の内訳（SHAP Waterfall）</b>", font=dict(size=18)),
            template='plotly_white',
            height=400 + len(display_df) * 30,
            showlegend=False
        )
        
        return fig
    
    def plot_force_single(self, explanation: dict) -> go.Figure:
        """Force Plot風の水平バーチャート"""
        contributions = explanation['contributions']
        base_value = explanation['base_value']
        prediction = explanation['prediction']
        
        # 正負で分けてソート
        positive = contributions[contributions['SHAP値'] > 0].sort_values('SHAP値', ascending=False)
        negative = contributions[contributions['SHAP値'] < 0].sort_values('SHAP値', ascending=True)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("価格を上げている要因 ↑", "価格を下げている要因 ↓"),
            vertical_spacing=0.15
        )
        
        # 正の要因
        if len(positive) > 0:
            top_pos = positive.head(7)
            labels = [f"{row['特徴量']}={row['値']}" if not isinstance(row['値'], str) else f"{row['特徴量']}={row['値']}" 
                     for _, row in top_pos.iterrows()]
            fig.add_trace(go.Bar(
                x=top_pos['SHAP値'].values,
                y=labels,
                orientation='h',
                marker_color='#2ca02c',
                text=[f"+{v:.1f}" for v in top_pos['SHAP値'].values],
                textposition='outside',
                name='価格上昇要因'
            ), row=1, col=1)
        
        # 負の要因
        if len(negative) > 0:
            top_neg = negative.head(7)
            labels = [f"{row['特徴量']}={row['値']}" if not isinstance(row['値'], str) else f"{row['特徴量']}={row['値']}" 
                     for _, row in top_neg.iterrows()]
            fig.add_trace(go.Bar(
                x=top_neg['SHAP値'].values,
                y=labels,
                orientation='h',
                marker_color='#d62728',
                text=[f"{v:.1f}" for v in top_neg['SHAP値'].values],
                textposition='outside',
                name='価格下落要因'
            ), row=2, col=1)
        
        fig.update_layout(
            title=dict(
                text=f"<b>予測価格: {prediction:.2f}</b>（基準値: {base_value:.2f}）",
                font=dict(size=18)
            ),
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig


class PredictionInterval:
    """予測信頼区間クラス"""
    
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.residuals = None
        self.residual_std = None
        
    def fit(self):
        """残差分布を学習"""
        # 学習データでの予測
        y_pred_train = self.model.predict(self.X_train)
        self.residuals = self.y_train.values - y_pred_train
        self.residual_std = np.std(self.residuals)
        
        # 残差の分位点を計算
        self.residual_percentiles = {
            '50': (np.percentile(self.residuals, 25), np.percentile(self.residuals, 75)),
            '80': (np.percentile(self.residuals, 10), np.percentile(self.residuals, 90)),
            '95': (np.percentile(self.residuals, 2.5), np.percentile(self.residuals, 97.5)),
        }
        
        return self
    
    def predict_with_interval(self, X: pd.DataFrame, confidence: float = 0.95) -> dict:
        """信頼区間付き予測"""
        if self.residual_std is None:
            self.fit()
        
        predictions = self.model.predict(X)
        
        # 信頼区間の計算（正規分布を仮定）
        from scipy import stats
        z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
        margin = z_score * self.residual_std
        
        lower = predictions - margin
        upper = predictions + margin
        
        return {
            'predictions': predictions,
            'lower': lower,
            'upper': upper,
            'confidence': confidence,
            'margin': margin
        }
    
    def predict_single_with_interval(self, X_single: pd.DataFrame, confidence_levels: list = [0.5, 0.8, 0.95]) -> dict:
        """単一予測の複数信頼区間"""
        if self.residual_std is None:
            self.fit()
        
        prediction = self.model.predict(X_single)[0]
        
        from scipy import stats
        intervals = {}
        for conf in confidence_levels:
            z = stats.norm.ppf(1 - (1 - conf) / 2)
            margin = z * self.residual_std
            intervals[f'{int(conf*100)}%'] = {
                'lower': prediction - margin,
                'upper': prediction + margin,
                'margin': margin
            }
        
        return {
            'prediction': prediction,
            'intervals': intervals,
            'residual_std': self.residual_std
        }
    
    def plot_prediction_interval(self, result: dict) -> go.Figure:
        """信頼区間の可視化（単一予測用）"""
        prediction = result['prediction']
        intervals = result['intervals']
        
        fig = go.Figure()
        
        colors = {
            '50%': 'rgba(31, 119, 180, 0.8)',
            '80%': 'rgba(31, 119, 180, 0.5)',
            '95%': 'rgba(31, 119, 180, 0.3)'
        }
        
        # 信頼区間を追加（広い順）
        for conf in ['95%', '80%', '50%']:
            if conf in intervals:
                interval = intervals[conf]
                fig.add_trace(go.Bar(
                    x=[interval['upper'] - interval['lower']],
                    y=[conf],
                    base=[interval['lower']],
                    orientation='h',
                    marker_color=colors.get(conf, 'rgba(31, 119, 180, 0.5)'),
                    name=f'{conf}信頼区間',
                    text=[f"{interval['lower']:.1f} 〜 {interval['upper']:.1f}"],
                    textposition='inside',
                    hovertemplate=f"{conf}信頼区間<br>下限: {interval['lower']:.2f}<br>上限: {interval['upper']:.2f}<extra></extra>"
                ))
        
        # 予測値のマーカー
        fig.add_trace(go.Scatter(
            x=[prediction],
            y=['50%', '80%', '95%'],
            mode='markers',
            marker=dict(color='red', size=15, symbol='diamond'),
            name=f'予測値: {prediction:.2f}'
        ))
        
        fig.update_layout(
            title=dict(text=f"<b>予測信頼区間</b>（予測値: {prediction:.2f}）", font=dict(size=18)),
            xaxis_title="予測価格",
            yaxis_title="信頼水準",
            template='plotly_white',
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='overlay'
        )
        
        return fig


class WhatIfAnalyzer:
    """What-if分析クラス"""
    
    def __init__(self, model, feature_columns: list, X_original: pd.DataFrame):
        self.model = model
        self.feature_columns = feature_columns
        self.X_original = X_original
        
    def analyze_feature_impact(self, X_base: pd.DataFrame, feature: str, 
                               values: list = None, n_points: int = 20) -> pd.DataFrame:
        """特定の特徴量を変化させた時の予測変化を分析"""
        
        if values is None:
            # 元のデータの範囲で値を生成
            if feature in self.X_original.columns:
                min_val = self.X_original[feature].min()
                max_val = self.X_original[feature].max()
                values = np.linspace(min_val, max_val, n_points)
            else:
                raise ValueError(f"Feature '{feature}' not found")
        
        results = []
        base_prediction = self.model.predict(X_base)[0]
        
        for val in values:
            X_modified = X_base.copy()
            X_modified[feature] = val
            new_prediction = self.model.predict(X_modified)[0]
            
            results.append({
                feature: val,
                '予測価格': new_prediction,
                '変化量': new_prediction - base_prediction,
                '変化率(%)': (new_prediction - base_prediction) / base_prediction * 100
            })
        
        return pd.DataFrame(results)
    
    def compare_scenarios(self, X_base: pd.DataFrame, scenarios: dict) -> pd.DataFrame:
        """複数シナリオの比較"""
        results = []
        base_prediction = self.model.predict(X_base)[0]
        
        # ベースケース
        results.append({
            'シナリオ': '現状（ベース）',
            '予測価格': base_prediction,
            '変化量': 0,
            '変化率(%)': 0
        })
        
        for scenario_name, changes in scenarios.items():
            X_modified = X_base.copy()
            for feature, value in changes.items():
                if feature in X_modified.columns:
                    X_modified[feature] = value
            
            new_prediction = self.model.predict(X_modified)[0]
            results.append({
                'シナリオ': scenario_name,
                '予測価格': new_prediction,
                '変化量': new_prediction - base_prediction,
                '変化率(%)': (new_prediction - base_prediction) / base_prediction * 100
            })
        
        return pd.DataFrame(results)
    
    def plot_feature_sensitivity(self, sensitivity_df: pd.DataFrame, feature: str) -> go.Figure:
        """特徴量感度分析のプロット"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("予測価格の変化", "変化率 (%)"),
            horizontal_spacing=0.12
        )
        
        # 予測価格
        fig.add_trace(go.Scatter(
            x=sensitivity_df[feature],
            y=sensitivity_df['予測価格'],
            mode='lines+markers',
            marker=dict(size=8, color='#1f77b4'),
            line=dict(width=2),
            name='予測価格'
        ), row=1, col=1)
        
        # 変化率
        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in sensitivity_df['変化率(%)']]
        fig.add_trace(go.Bar(
            x=sensitivity_df[feature],
            y=sensitivity_df['変化率(%)'],
            marker_color=colors,
            name='変化率'
        ), row=1, col=2)
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        fig.update_layout(
            title=dict(text=f"<b>What-if分析: {feature}の影響</b>", font=dict(size=18)),
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text=feature, row=1, col=1)
        fig.update_xaxes(title_text=feature, row=1, col=2)
        fig.update_yaxes(title_text="予測価格", row=1, col=1)
        fig.update_yaxes(title_text="変化率 (%)", row=1, col=2)
        
        return fig
    
    def plot_scenario_comparison(self, scenario_df: pd.DataFrame) -> go.Figure:
        """シナリオ比較のプロット"""
        
        colors = ['#808080'] + ['#2ca02c' if v >= 0 else '#d62728' for v in scenario_df['変化量'].iloc[1:]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=scenario_df['シナリオ'],
            y=scenario_df['予測価格'],
            marker_color=colors,
            text=[f"{v:,.0f}<br>({scenario_df['変化率(%)'].iloc[i]:+.1f}%)" 
                  for i, v in enumerate(scenario_df['予測価格'])],
            textposition='outside'
        ))
        
        # ベースラインを追加
        base_price = scenario_df['予測価格'].iloc[0]
        fig.add_hline(y=base_price, line_dash="dash", line_color="gray",
                     annotation_text=f"ベース: {base_price:,.0f}")
        
        fig.update_layout(
            title=dict(text="<b>シナリオ比較分析</b>", font=dict(size=18)),
            xaxis_title="シナリオ",
            yaxis_title="予測価格",
            template='plotly_white',
            height=450
        )
        
        return fig


class PDFReportGenerator:
    """PDFレポート生成クラス"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_report(self, 
                       prediction_result: dict,
                       shap_explanation: dict = None,
                       interval_result: dict = None,
                       model_metrics: dict = None,
                       input_data: pd.DataFrame = None,
                       target_column: str = "価格") -> bytes:
        """PDFレポートを生成"""
        from fpdf import FPDF
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Helvetica', 'B', 16)
                self.cell(0, 10, 'RealEstateMLStudio', 0, 1, 'C')
                self.set_font('Helvetica', '', 10)
                self.cell(0, 5, 'Property Price Prediction Report', 0, 1, 'C')
                self.ln(5)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(5)
                
            def footer(self):
                self.set_y(-15)
                self.set_font('Helvetica', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # 1. 予測結果サマリー
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, '1. Prediction Summary', 0, 1)
        pdf.set_font('Helvetica', '', 11)
        
        prediction = prediction_result.get('prediction', 0)
        pdf.set_font('Helvetica', 'B', 24)
        pdf.set_text_color(31, 119, 180)
        pdf.cell(0, 15, f'Predicted {target_column}: {prediction:,.2f}', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)
        
        # 信頼区間
        if interval_result:
            pdf.set_font('Helvetica', '', 10)
            intervals = interval_result.get('intervals', {})
            if '95%' in intervals:
                lower = intervals['95%']['lower']
                upper = intervals['95%']['upper']
                pdf.cell(0, 8, f'95% Confidence Interval: {lower:,.2f} - {upper:,.2f}', 0, 1, 'C')
        
        pdf.ln(5)
        
        # 2. 入力データ
        if input_data is not None:
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, '2. Input Features', 0, 1)
            pdf.set_font('Helvetica', '', 9)
            
            col_width = 60
            row_height = 7
            
            for col in input_data.columns:
                value = input_data[col].iloc[0]
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                pdf.cell(col_width, row_height, str(col)[:25], 1, 0)
                pdf.cell(col_width, row_height, value_str[:25], 1, 1)
            
            pdf.ln(5)
        
        # 3. SHAP分析結果
        if shap_explanation:
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, '3. SHAP Analysis - Price Factors', 0, 1)
            pdf.set_font('Helvetica', '', 9)
            
            contributions = shap_explanation.get('contributions', pd.DataFrame())
            if not contributions.empty:
                # 上位要因
                positive = contributions[contributions['SHAP値'] > 0].head(5)
                negative = contributions[contributions['SHAP値'] < 0].head(5)
                
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(44, 160, 44)
                pdf.cell(0, 8, 'Positive Factors (Increase Price):', 0, 1)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font('Helvetica', '', 9)
                
                for _, row in positive.iterrows():
                    feature = str(row['特徴量'])[:20]
                    value = row['値']
                    shap_val = row['SHAP値']
                    if isinstance(value, float):
                        pdf.cell(0, 6, f"  + {feature} = {value:.2f} (SHAP: +{shap_val:.2f})", 0, 1)
                    else:
                        pdf.cell(0, 6, f"  + {feature} = {value} (SHAP: +{shap_val:.2f})", 0, 1)
                
                pdf.ln(3)
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(214, 39, 40)
                pdf.cell(0, 8, 'Negative Factors (Decrease Price):', 0, 1)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font('Helvetica', '', 9)
                
                for _, row in negative.iterrows():
                    feature = str(row['特徴量'])[:20]
                    value = row['値']
                    shap_val = row['SHAP値']
                    if isinstance(value, float):
                        pdf.cell(0, 6, f"  - {feature} = {value:.2f} (SHAP: {shap_val:.2f})", 0, 1)
                    else:
                        pdf.cell(0, 6, f"  - {feature} = {value} (SHAP: {shap_val:.2f})", 0, 1)
            
            pdf.ln(5)
        
        # 4. モデル性能
        if model_metrics:
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, '4. Model Performance Metrics', 0, 1)
            pdf.set_font('Helvetica', '', 10)
            
            metrics_text = [
                f"R2 Score: {model_metrics.get('r2', 0):.4f}",
                f"RMSE: {model_metrics.get('rmse', 0):.4f}",
                f"MAE: {model_metrics.get('mae', 0):.4f}",
                f"MAPE: {model_metrics.get('mape', 0):.2f}%"
            ]
            
            for text in metrics_text:
                pdf.cell(0, 7, text, 0, 1)
        
        # 5. 免責事項
        pdf.ln(10)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(0, 4, 
            'Disclaimer: This prediction is generated by a machine learning model and should be used for reference only. '
            'Actual property prices may vary based on market conditions, property-specific factors, and other variables '
            'not captured in the model. Please consult with a qualified professional for important decisions.')
        
        # PDFをバイトとして返す
        return bytes(pdf.output())
    
    def get_download_link(self, pdf_bytes: bytes, filename: str = "prediction_report.pdf") -> str:
        """ダウンロードリンクを生成"""
        b64 = base64.b64encode(pdf_bytes).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 PDFレポートをダウンロード</a>'


class SimilarPropertyFinder:
    """類似物件検索クラス"""
    
    def __init__(self, df: pd.DataFrame, feature_columns: list, target_column: str):
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaler = None
        self.scaled_features = None
        
    def fit(self):
        """特徴量をスケーリング"""
        from sklearn.preprocessing import StandardScaler
        
        # 数値列のみ抽出
        numeric_cols = [col for col in self.feature_columns 
                       if col in self.df.columns and self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        self.numeric_cols = numeric_cols
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[numeric_cols].fillna(0))
        
        return self
    
    def find_similar(self, X_query: pd.DataFrame, n_neighbors: int = 5, 
                    method: str = 'euclidean') -> pd.DataFrame:
        """類似物件を検索"""
        if self.scaler is None:
            self.fit()
        
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        # クエリをスケーリング
        query_numeric = X_query[self.numeric_cols].fillna(0).values
        query_scaled = self.scaler.transform(query_numeric)
        
        # 距離/類似度を計算
        if method == 'cosine':
            similarities = cosine_similarity(query_scaled, self.scaled_features)[0]
            indices = np.argsort(similarities)[::-1][:n_neighbors]
            scores = similarities[indices]
        else:  # euclidean
            distances = euclidean_distances(query_scaled, self.scaled_features)[0]
            indices = np.argsort(distances)[:n_neighbors]
            scores = 1 / (1 + distances[indices])  # 類似度に変換
        
        # 結果を作成
        similar_df = self.df.iloc[indices].copy()
        similar_df['類似度スコア'] = scores
        similar_df['順位'] = range(1, len(similar_df) + 1)
        
        # 列の順序を調整
        cols = ['順位', '類似度スコア'] + [self.target_column] + \
               [c for c in similar_df.columns if c not in ['順位', '類似度スコア', self.target_column]]
        
        return similar_df[cols]
    
    def plot_similar_properties(self, query_prediction: float, similar_df: pd.DataFrame) -> go.Figure:
        """類似物件の比較プロット"""
        fig = go.Figure()
        
        # 類似物件の実績価格
        fig.add_trace(go.Bar(
            x=[f"類似物件{i+1}" for i in range(len(similar_df))],
            y=similar_df[self.target_column].values,
            name='実績価格',
            marker_color='#1f77b4',
            text=[f"{v:,.0f}<br>(類似度:{s:.2f})" 
                  for v, s in zip(similar_df[self.target_column].values, similar_df['類似度スコア'].values)],
            textposition='outside'
        ))
        
        # 予測価格のライン
        fig.add_hline(y=query_prediction, line_dash="dash", line_color="red",
                     annotation_text=f"予測価格: {query_prediction:,.0f}")
        
        # 平均価格のライン
        avg_price = similar_df[self.target_column].mean()
        fig.add_hline(y=avg_price, line_dash="dot", line_color="green",
                     annotation_text=f"類似物件平均: {avg_price:,.0f}")
        
        fig.update_layout(
            title=dict(text="<b>類似物件との価格比較</b>", font=dict(size=18)),
            xaxis_title="物件",
            yaxis_title=self.target_column,
            template='plotly_white',
            height=450
        )
        
        return fig


class DataQualityChecker:
    """データ品質チェッククラス"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {}
        
    def check_all(self) -> dict:
        """全てのチェックを実行"""
        self.report = {
            'duplicates': self._check_duplicates(),
            'missing': self._check_missing(),
            'outliers': self._check_outliers(),
            'type_issues': self._check_type_issues(),
            'value_ranges': self._check_value_ranges(),
            'correlations': self._check_high_correlations(),
            'summary': {}
        }
        
        # サマリーを作成
        total_issues = (
            self.report['duplicates']['count'] +
            len(self.report['missing']['columns_with_missing']) +
            sum(len(v) for v in self.report['outliers'].values()) +
            len(self.report['type_issues'])
        )
        
        self.report['summary'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'total_issues': total_issues,
            'quality_score': max(0, 100 - total_issues * 2)  # 簡易スコア
        }
        
        return self.report
    
    def _check_duplicates(self) -> dict:
        """重複行チェック"""
        duplicates = self.df.duplicated()
        duplicate_rows = self.df[duplicates]
        
        return {
            'count': duplicates.sum(),
            'percentage': (duplicates.sum() / len(self.df) * 100),
            'indices': duplicate_rows.index.tolist()[:10]  # 最初の10件
        }
    
    def _check_missing(self) -> dict:
        """欠損値チェック"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        columns_with_missing = missing[missing > 0].to_dict()
        
        return {
            'total_missing_cells': self.df.isnull().sum().sum(),
            'columns_with_missing': columns_with_missing,
            'missing_percentage': missing_pct[missing_pct > 0].to_dict()
        }
    
    def _check_outliers(self) -> dict:
        """異常値チェック（IQR法）"""
        outliers = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outlier_values = self.df.loc[outlier_mask, col].head(5).tolist()
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_count / len(self.df) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'sample_values': outlier_values
                }
        
        return outliers
    
    def _check_type_issues(self) -> list:
        """データ型の問題をチェック"""
        issues = []
        
        for col in self.df.columns:
            # 数値列に文字列が混入していないかチェック
            if self.df[col].dtype == 'object':
                # 数値に変換可能かチェック
                numeric_convertible = pd.to_numeric(self.df[col], errors='coerce')
                non_numeric_count = numeric_convertible.isna().sum() - self.df[col].isna().sum()
                
                if non_numeric_count > 0 and non_numeric_count < len(self.df) * 0.5:
                    issues.append({
                        'column': col,
                        'issue': '数値と文字列が混在',
                        'non_numeric_count': int(non_numeric_count)
                    })
        
        return issues
    
    def _check_value_ranges(self) -> dict:
        """値の範囲チェック（負の値など）"""
        issues = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_issues = []
            
            # 負の値チェック
            negative_count = (self.df[col] < 0).sum()
            if negative_count > 0:
                col_issues.append(f"負の値: {negative_count}件")
            
            # ゼロ値チェック
            zero_count = (self.df[col] == 0).sum()
            if zero_count > len(self.df) * 0.5:  # 50%以上がゼロ
                col_issues.append(f"ゼロ値が多い: {zero_count}件 ({zero_count/len(self.df)*100:.1f}%)")
            
            if col_issues:
                issues[col] = col_issues
        
        return issues
    
    def _check_high_correlations(self, threshold: float = 0.95) -> list:
        """高い相関をチェック（多重共線性）"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return []
        
        corr_matrix = numeric_df.corr().abs()
        
        # 上三角行列から高い相関を抽出
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= threshold:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': round(corr_matrix.iloc[i, j], 4)
                    })
        
        return high_corr
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """サマリーをDataFrameで取得"""
        if not self.report:
            self.check_all()
        
        data = [
            {'チェック項目': '総行数', '結果': self.report['summary']['total_rows'], '状態': '✅'},
            {'チェック項目': '総列数', '結果': self.report['summary']['total_columns'], '状態': '✅'},
            {'チェック項目': '重複行', '結果': self.report['duplicates']['count'], 
             '状態': '✅' if self.report['duplicates']['count'] == 0 else '⚠️'},
            {'チェック項目': '欠損値のある列', '結果': len(self.report['missing']['columns_with_missing']),
             '状態': '✅' if len(self.report['missing']['columns_with_missing']) == 0 else '⚠️'},
            {'チェック項目': '異常値のある列', '結果': len(self.report['outliers']),
             '状態': '✅' if len(self.report['outliers']) == 0 else '⚠️'},
            {'チェック項目': '型の問題', '結果': len(self.report['type_issues']),
             '状態': '✅' if len(self.report['type_issues']) == 0 else '⚠️'},
            {'チェック項目': '高相関ペア', '結果': len(self.report['correlations']),
             '状態': '✅' if len(self.report['correlations']) == 0 else '⚠️'},
            {'チェック項目': '品質スコア', '結果': f"{self.report['summary']['quality_score']}/100",
             '状態': '✅' if self.report['summary']['quality_score'] >= 80 else '⚠️'}
        ]
        
        return pd.DataFrame(data)
    
    def plot_quality_overview(self) -> go.Figure:
        """品質概要のプロット"""
        if not self.report:
            self.check_all()
        
        # ゲージチャート
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=self.report['summary']['quality_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "データ品質スコア", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ff6b6b'},
                    {'range': [50, 80], 'color': '#ffd93d'},
                    {'range': [80, 100], 'color': '#6bcb77'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            template='plotly_white'
        )
        
        return fig


class FeatureEngineer:
    """特徴量自動生成クラス"""
    
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
        self.new_features = []
        self.feature_info = []
        
    def generate_all(self, include_interactions: bool = True,
                    include_polynomial: bool = True,
                    include_ratios: bool = True,
                    include_binning: bool = True) -> pd.DataFrame:
        """全ての特徴量を生成"""
        
        df_new = self.df.copy()
        numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
        
        # ターゲット列を除外
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        if include_interactions:
            df_new = self._create_interactions(df_new, numeric_cols[:5])  # 上位5列
        
        if include_polynomial:
            df_new = self._create_polynomial(df_new, numeric_cols[:5])
        
        if include_ratios:
            df_new = self._create_ratios(df_new, numeric_cols[:5])
        
        if include_binning:
            df_new = self._create_binning(df_new, numeric_cols)
        
        return df_new
    
    def _create_interactions(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """交互作用項を作成"""
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                new_col = f"{col1}_×_{col2}"
                df[new_col] = df[col1] * df[col2]
                self.new_features.append(new_col)
                self.feature_info.append({
                    '特徴量': new_col,
                    'タイプ': '交互作用',
                    '元の特徴量': f"{col1}, {col2}",
                    '説明': f"{col1}と{col2}の積"
                })
        
        return df
    
    def _create_polynomial(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """多項式特徴量を作成"""
        for col in columns:
            # 2乗
            new_col = f"{col}_squared"
            df[new_col] = df[col] ** 2
            self.new_features.append(new_col)
            self.feature_info.append({
                '特徴量': new_col,
                'タイプ': '多項式',
                '元の特徴量': col,
                '説明': f"{col}の2乗"
            })
            
            # 平方根（正の値のみ）
            if (df[col] >= 0).all():
                new_col = f"{col}_sqrt"
                df[new_col] = np.sqrt(df[col])
                self.new_features.append(new_col)
                self.feature_info.append({
                    '特徴量': new_col,
                    'タイプ': '多項式',
                    '元の特徴量': col,
                    '説明': f"{col}の平方根"
                })
        
        return df
    
    def _create_ratios(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """比率特徴量を作成"""
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # ゼロ除算を避ける
                if (df[col2] != 0).all():
                    new_col = f"{col1}_per_{col2}"
                    df[new_col] = df[col1] / df[col2]
                    self.new_features.append(new_col)
                    self.feature_info.append({
                        '特徴量': new_col,
                        'タイプ': '比率',
                        '元の特徴量': f"{col1}, {col2}",
                        '説明': f"{col1}÷{col2}"
                    })
        
        return df
    
    def _create_binning(self, df: pd.DataFrame, columns: list, n_bins: int = 5) -> pd.DataFrame:
        """ビニング特徴量を作成"""
        for col in columns[:3]:  # 上位3列のみ
            new_col = f"{col}_bin"
            try:
                df[new_col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                self.new_features.append(new_col)
                self.feature_info.append({
                    '特徴量': new_col,
                    'タイプ': 'ビニング',
                    '元の特徴量': col,
                    '説明': f"{col}を{n_bins}分位に分割"
                })
            except Exception:
                pass  # ビニングできない場合はスキップ
        
        return df
    
    def get_feature_info(self) -> pd.DataFrame:
        """生成した特徴量の情報を取得"""
        return pd.DataFrame(self.feature_info)
    
    def evaluate_features(self, model, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        """特徴量の有効性を評価"""
        from sklearn.metrics import r2_score
        
        results = []
        
        # ベースライン
        model.fit(X_train, y_train)
        base_score = r2_score(y_test, model.predict(X_test))
        
        # 各新特徴量を追加して評価
        for feat in self.new_features:
            if feat in X_train.columns:
                continue
                
            # 新特徴量を追加
            X_train_new = X_train.copy()
            X_test_new = X_test.copy()
            
            # 特徴量を追加
            # ... (実際の実装では元データから特徴量を再計算)
            
        return pd.DataFrame(results)


class PredictionHistory:
    """予測履歴管理クラス"""
    
    def __init__(self):
        self.history = []
        
    def add_prediction(self, input_data: dict, prediction: float, 
                      confidence_interval: dict = None,
                      model_type: str = None,
                      similar_properties: pd.DataFrame = None):
        """予測を履歴に追加"""
        record = {
            'id': len(self.history) + 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_data': input_data,
            'prediction': prediction,
            'confidence_interval': confidence_interval,
            'model_type': model_type,
            'similar_avg': similar_properties[similar_properties.columns[2]].mean() if similar_properties is not None else None
        }
        
        self.history.append(record)
        return record
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """履歴をDataFrameで取得"""
        if not self.history:
            return pd.DataFrame()
        
        records = []
        for h in self.history:
            record = {
                'ID': h['id'],
                '日時': h['timestamp'],
                '予測価格': h['prediction'],
                'モデル': h['model_type'] or '-',
            }
            
            # 信頼区間
            if h['confidence_interval']:
                intervals = h['confidence_interval'].get('intervals', {})
                if '95%' in intervals:
                    record['95%下限'] = intervals['95%']['lower']
                    record['95%上限'] = intervals['95%']['upper']
            
            # 類似物件平均
            if h['similar_avg']:
                record['類似物件平均'] = h['similar_avg']
            
            # 入力データのサマリー
            input_summary = ', '.join([f"{k}={v}" for k, v in list(h['input_data'].items())[:3]])
            record['入力条件'] = input_summary + '...' if len(h['input_data']) > 3 else input_summary
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def clear_history(self):
        """履歴をクリア"""
        self.history = []
    
    def export_to_csv(self) -> str:
        """CSVとしてエクスポート"""
        df = self.get_history_dataframe()
        return df.to_csv(index=False)
    
    def export_to_json(self) -> str:
        """JSONとしてエクスポート"""
        import json
        return json.dumps(self.history, ensure_ascii=False, indent=2, default=str)
    
    def plot_history(self) -> go.Figure:
        """予測履歴のプロット"""
        if not self.history:
            return None
        
        df = self.get_history_dataframe()
        
        fig = go.Figure()
        
        # 予測価格
        fig.add_trace(go.Scatter(
            x=df['日時'],
            y=df['予測価格'],
            mode='lines+markers',
            name='予測価格',
            marker=dict(size=10, color='#1f77b4'),
            line=dict(width=2)
        ))
        
        # 信頼区間があれば追加
        if '95%下限' in df.columns and '95%上限' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['日時'].tolist() + df['日時'].tolist()[::-1],
                y=df['95%上限'].tolist() + df['95%下限'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95%信頼区間'
            ))
        
        # 類似物件平均があれば追加
        if '類似物件平均' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['日時'],
                y=df['類似物件平均'],
                mode='markers',
                name='類似物件平均',
                marker=dict(size=8, color='#2ca02c', symbol='diamond')
            ))
        
        fig.update_layout(
            title=dict(text="<b>予測履歴</b>", font=dict(size=18)),
            xaxis_title="日時",
            yaxis_title="価格",
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def get_statistics(self) -> dict:
        """履歴の統計情報"""
        if not self.history:
            return {}
        
        predictions = [h['prediction'] for h in self.history]
        
        return {
            '予測回数': len(self.history),
            '平均予測価格': np.mean(predictions),
            '最高予測価格': np.max(predictions),
            '最低予測価格': np.min(predictions),
            '標準偏差': np.std(predictions)
        }
