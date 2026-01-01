"""
可視化モジュール - 豪華なグラフと評価指標の可視化
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# カラーパレット
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

# モデル別カラー
MODEL_COLORS = {
    'XGBoost': '#1f77b4',
    'LightGBM': '#ff7f0e',
    'CatBoost': '#2ca02c',
    'Stacking': '#9467bd'
}


class Visualizer:
    """可視化クラス"""
    
    def __init__(self):
        self.theme = 'plotly_white'
        
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  title: str = "実測値 vs 予測値") -> go.Figure:
        """実測値 vs 予測値の豪華な散布図"""
        
        # 残差を計算
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        
        fig = go.Figure()
        
        # メインの散布図（残差の大きさで色分け）
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            marker=dict(
                size=10,
                color=abs_residuals,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="残差の絶対値"),
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"実測: {t:.2f}<br>予測: {p:.2f}<br>残差: {r:.2f}" 
                  for t, p, r in zip(y_true, y_pred, residuals)],
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="データポイント"
        ))
        
        # 45度線（完全予測線）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='完全予測線 (y=x)'
        ))
        
        # 回帰線を追加
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'回帰線 (y={z[0]:.3f}x + {z[1]:.3f})'
        ))
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=20)),
            xaxis_title="実測値",
            yaxis_title="予測値",
            template=self.theme,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            height=600
        )
        
        # アスペクト比を1:1に
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """残差プロット（複合グラフ）"""
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("残差 vs 予測値", "残差のヒストグラム", 
                          "残差のQ-Qプロット", "残差の時系列"),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. 残差 vs 予測値
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            marker=dict(color=COLORS['primary'], opacity=0.6, size=8),
            name="残差"
        ), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. 残差のヒストグラム
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color=COLORS['secondary'],
            opacity=0.7,
            name="残差分布"
        ), row=1, col=2)
        
        # 正規分布曲線を追加
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = (1/(residuals.std() * np.sqrt(2*np.pi))) * \
                 np.exp(-0.5*((x_norm - residuals.mean())/residuals.std())**2)
        y_norm = y_norm * len(residuals) * (residuals.max() - residuals.min()) / 30
        fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            mode='lines',
            line=dict(color='red', width=2),
            name="正規分布"
        ), row=1, col=2)
        
        # 3. Q-Qプロット
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.linspace(0.01, 0.99, len(residuals))
        theoretical_values = np.quantile(np.random.normal(0, residuals.std(), 10000), theoretical_quantiles)
        
        fig.add_trace(go.Scatter(
            x=theoretical_values,
            y=sorted_residuals,
            mode='markers',
            marker=dict(color=COLORS['success'], size=6),
            name="Q-Q"
        ), row=2, col=1)
        
        qq_min = min(theoretical_values.min(), sorted_residuals.min())
        qq_max = max(theoretical_values.max(), sorted_residuals.max())
        fig.add_trace(go.Scatter(
            x=[qq_min, qq_max],
            y=[qq_min, qq_max],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name="基準線"
        ), row=2, col=1)
        
        # 4. 残差の時系列（インデックス順）
        fig.add_trace(go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='lines+markers',
            marker=dict(color=COLORS['info'], size=4),
            line=dict(width=1),
            name="残差推移"
        ), row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(
            title=dict(text="<b>残差分析ダッシュボード</b>", font=dict(size=20)),
            template=self.theme,
            height=700,
            showlegend=False
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.Series, 
                                top_n: int = 20,
                                title: str = "特徴量重要度") -> go.Figure:
        """特徴量重要度の水平バーチャート"""
        
        # 上位N個を取得
        fi = feature_importance.head(top_n).sort_values(ascending=True)
        
        # グラデーションカラーを生成
        colors = px.colors.sample_colorscale('Viridis', len(fi))
        
        fig = go.Figure(go.Bar(
            x=fi.values,
            y=fi.index,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f"{v:.4f}" for v in fi.values],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>重要度: %{x:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b> (Top {top_n})", font=dict(size=20)),
            xaxis_title="重要度スコア",
            yaxis_title="特徴量",
            template=self.theme,
            height=max(400, top_n * 25),
            margin=dict(l=150)
        )
        
        return fig
    
    def plot_metrics_dashboard(self, metrics: dict, cv_scores: dict = None) -> go.Figure:
        """評価指標ダッシュボード"""
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.35,
            horizontal_spacing=0.15
        )
        
        # RMSE
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['rmse'],
            title={"text": "RMSE<br><span style='font-size:11px;color:gray'>低いほど良い</span>", 
                   "font": {"size": 16}},
            gauge=dict(
                axis=dict(range=[0, metrics['rmse'] * 2]),
                bar=dict(color=COLORS['primary']),
                steps=[
                    dict(range=[0, metrics['rmse']], color="lightgray"),
                ],
                threshold=dict(
                    line=dict(color="red", width=4),
                    thickness=0.75,
                    value=metrics['rmse']
                )
            ),
            delta=dict(reference=metrics['rmse'] * 1.1, relative=True) if cv_scores else None,
            domain={'row': 0, 'column': 0}
        ), row=1, col=1)
        
        # R² Score
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['r2'],
            title={"text": "R² Score<br><span style='font-size:11px;color:gray'>1に近いほど良い</span>",
                   "font": {"size": 16}},
            number=dict(suffix="", valueformat=".4f"),
            gauge=dict(
                axis=dict(range=[0, 1]),
                bar=dict(color=COLORS['success']),
                steps=[
                    dict(range=[0, 0.5], color="#ffebee"),
                    dict(range=[0.5, 0.7], color="#fff3e0"),
                    dict(range=[0.7, 0.9], color="#e8f5e9"),
                    dict(range=[0.9, 1], color="#c8e6c9"),
                ],
                threshold=dict(
                    line=dict(color="green", width=4),
                    thickness=0.75,
                    value=metrics['r2']
                )
            ),
            domain={'row': 0, 'column': 1}
        ), row=1, col=2)
        
        # MAE
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['mae'],
            title={"text": "MAE<br><span style='font-size:11px;color:gray'>平均絶対誤差</span>",
                   "font": {"size": 16}},
            gauge=dict(
                axis=dict(range=[0, metrics['mae'] * 2]),
                bar=dict(color=COLORS['secondary']),
                steps=[
                    dict(range=[0, metrics['mae']], color="lightgray"),
                ],
            ),
            domain={'row': 1, 'column': 0}
        ), row=2, col=1)
        
        # MAPE
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['mape'],
            title={"text": "MAPE (%)<br><span style='font-size:11px;color:gray'>平均絶対%誤差</span>",
                   "font": {"size": 16}},
            number=dict(suffix="%", valueformat=".2f"),
            gauge=dict(
                axis=dict(range=[0, min(100, metrics['mape'] * 2)]),
                bar=dict(color=COLORS['info']),
                steps=[
                    dict(range=[0, 10], color="#c8e6c9"),
                    dict(range=[10, 20], color="#fff3e0"),
                    dict(range=[20, 100], color="#ffebee"),
                ],
            ),
            domain={'row': 1, 'column': 1}
        ), row=2, col=2)
        
        fig.update_layout(
            title=dict(
                text="<b>モデル評価指標ダッシュボード</b>", 
                font=dict(size=22),
                y=0.98,
                x=0.5,
                xanchor='center'
            ),
            height=700,
            margin=dict(t=80, b=30, l=30, r=30),
            template=self.theme
        )
        
        return fig
    
    def plot_cv_results(self, cv_scores: dict) -> go.Figure:
        """交差検証結果の可視化"""
        
        metrics = ['rmse', 'mae', 'r2', 'mape']
        metric_names = ['RMSE', 'MAE', 'R²', 'MAPE (%)']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_names,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['info']]
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            fold_values = cv_scores['fold_results'][metric]
            mean_val = cv_scores[f'{metric}_mean']
            std_val = cv_scores[f'{metric}_std']
            
            # 各Foldの値をバーで表示
            fig.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(fold_values))],
                y=fold_values,
                marker_color=colors[idx],
                opacity=0.7,
                name=name,
                text=[f"{v:.4f}" for v in fold_values],
                textposition='outside'
            ), row=row, col=col)
            
            # 平均線
            fig.add_hline(
                y=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"平均: {mean_val:.4f} (±{std_val:.4f})",
                row=row, col=col
            )
        
        fig.update_layout(
            title=dict(text="<b>交差検証結果 (各Fold)</b>", font=dict(size=20)),
            height=600,
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """モデル比較グラフ（動的に全モデル対応）"""
        
        fig = go.Figure()
        
        metrics = comparison_df.index.tolist()
        models = comparison_df.columns.tolist()
        
        # デフォルトカラーパレット
        default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', 
            '#d62728', '#8c564b', '#e377c2', '#7f7f7f'
        ]
        
        for i, model in enumerate(models):
            color = MODEL_COLORS.get(model, default_colors[i % len(default_colors)])
            fig.add_trace(go.Bar(
                name=model,
                x=metrics,
                y=comparison_df[model].values,
                marker_color=color,
                text=[f"{v:.4f}" for v in comparison_df[model].values],
                textposition='outside'
            ))
        
        # タイトルを動的に生成
        model_names = " vs ".join(models)
        
        fig.update_layout(
            title=dict(text=f"<b>{model_names} モデル比較</b>", font=dict(size=20)),
            xaxis_title="評価指標",
            yaxis_title="スコア",
            barmode='group',
            template=self.theme,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_model_comparison_radar(self, comparison_df: pd.DataFrame) -> go.Figure:
        """モデル比較レーダーチャート"""
        
        fig = go.Figure()
        
        models = comparison_df.columns.tolist()
        metrics = comparison_df.index.tolist()
        
        # R²以外は反転（低いほど良い指標を高いほど良いに変換）
        normalized_df = comparison_df.copy()
        for metric in metrics:
            if metric != 'R²':
                max_val = normalized_df.loc[metric].max()
                if max_val > 0:
                    normalized_df.loc[metric] = 1 - (normalized_df.loc[metric] / max_val)
        
        default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', 
            '#d62728', '#8c564b', '#e377c2', '#7f7f7f'
        ]
        
        for i, model in enumerate(models):
            color = MODEL_COLORS.get(model, default_colors[i % len(default_colors)])
            values = normalized_df[model].values.tolist()
            values.append(values[0])  # 閉じる
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model,
                line_color=color,
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=dict(text="<b>モデル性能レーダーチャート</b>", font=dict(size=20)),
            template=self.theme,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """予測値と実測値の分布比較"""
        
        fig = go.Figure()
        
        # 実測値のヒストグラム
        fig.add_trace(go.Histogram(
            x=y_true,
            name='実測値',
            opacity=0.7,
            marker_color=COLORS['primary'],
            nbinsx=30
        ))
        
        # 予測値のヒストグラム
        fig.add_trace(go.Histogram(
            x=y_pred,
            name='予測値',
            opacity=0.7,
            marker_color=COLORS['secondary'],
            nbinsx=30
        ))
        
        fig.update_layout(
            title=dict(text="<b>実測値と予測値の分布比較</b>", font=dict(size=20)),
            xaxis_title="値",
            yaxis_title="頻度",
            barmode='overlay',
            template=self.theme,
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_eda_dashboard(self, df: pd.DataFrame, target_column: str = None) -> go.Figure:
        """探索的データ分析ダッシュボード"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        # 相関行列のヒートマップ
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>相関係数: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(text="<b>特徴量相関マトリックス</b>", font=dict(size=20)),
            template=self.theme,
            height=600,
            width=800
        )
        
        return fig
    
    def plot_shap_summary(self, shap_values, feature_names: list) -> go.Figure:
        """SHAP値のサマリープロット（簡易版）"""
        
        # 平均絶対SHAP値を計算
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # DataFrameに変換してソート
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=shap_df['importance'],
            y=shap_df['feature'],
            orientation='h',
            marker=dict(
                color=shap_df['importance'],
                colorscale='Viridis',
                line=dict(color='white', width=1)
            ),
            text=[f"{v:.4f}" for v in shap_df['importance']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(text="<b>SHAP特徴量重要度</b>", font=dict(size=20)),
            xaxis_title="平均|SHAP値|",
            yaxis_title="特徴量",
            template=self.theme,
            height=max(400, len(feature_names) * 25),
            margin=dict(l=150)
        )
        
        return fig
    
    def plot_learning_curve(self, train_sizes: list, train_scores: list, 
                           val_scores: list) -> go.Figure:
        """学習曲線の可視化"""
        
        fig = go.Figure()
        
        # トレーニングスコア
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=COLORS['primary']),
            marker=dict(size=8)
        ))
        
        # バリデーションスコア
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_scores,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=COLORS['secondary']),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=dict(text="<b>学習曲線</b>", font=dict(size=20)),
            xaxis_title="トレーニングサンプル数",
            yaxis_title="スコア",
            template=self.theme,
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig


def create_metrics_cards(metrics: dict) -> str:
    """評価指標のHTMLカードを生成"""
    
    cards_html = """
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 1.1em;
            opacity: 0.9;
        }
    </style>
    """
    
    return cards_html
