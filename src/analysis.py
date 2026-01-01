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
