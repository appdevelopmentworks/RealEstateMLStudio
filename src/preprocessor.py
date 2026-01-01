"""
前処理モジュール - データの自動前処理機能
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st


class DataPreprocessor:
    """データ前処理クラス"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.imputers = {}
        self.original_columns = None
        self.numeric_columns = None
        self.categorical_columns = None
        
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """データの基本分析を行う"""
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        return analysis
    
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
        """IQR法による異常値検出"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    def detect_outliers_zscore(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
        """Z-score法による異常値検出"""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """欠損値の処理"""
        df_processed = df.copy()
        
        for column in df_processed.columns:
            if df_processed[column].isnull().sum() == 0:
                continue
                
            if strategy == 'auto':
                if df_processed[column].dtype in ['int64', 'float64']:
                    # 数値列: 中央値で補完
                    imputer = SimpleImputer(strategy='median')
                    df_processed[column] = imputer.fit_transform(df_processed[[column]])
                    self.imputers[column] = imputer
                else:
                    # カテゴリ列: 最頻値で補完
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_processed[column] = imputer.fit_transform(df_processed[[column]])
                    self.imputers[column] = imputer
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[column])
            elif strategy == 'mean':
                if df_processed[column].dtype in ['int64', 'float64']:
                    df_processed[column].fillna(df_processed[column].mean(), inplace=True)
            elif strategy == 'median':
                if df_processed[column].dtype in ['int64', 'float64']:
                    df_processed[column].fillna(df_processed[column].median(), inplace=True)
            elif strategy == 'mode':
                df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)
                
        return df_processed
    
    def encode_categorical(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """カテゴリ変数のエンコーディング"""
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in columns:
            if column in df_processed.columns:
                le = LabelEncoder()
                # NaN値を一時的に文字列に変換
                df_processed[column] = df_processed[column].astype(str)
                df_processed[column] = le.fit_transform(df_processed[column])
                self.label_encoders[column] = le
                
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', columns: list = None) -> pd.DataFrame:
        """特徴量のスケーリング"""
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return df_processed
            
        df_processed[columns] = self.scaler.fit_transform(df_processed[columns])
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame, columns: list = None, method: str = 'iqr', action: str = 'clip') -> pd.DataFrame:
        """異常値の処理"""
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if method == 'iqr':
                outliers = self.detect_outliers_iqr(df_processed, column)
            else:
                outliers = self.detect_outliers_zscore(df_processed, column)
            
            if action == 'remove':
                df_processed = df_processed[~outliers]
            elif action == 'clip':
                Q1 = df_processed[column].quantile(0.25)
                Q3 = df_processed[column].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_processed[column] = df_processed[column].clip(lower, upper)
                
        return df_processed
    
    def auto_preprocess(self, df: pd.DataFrame, target_column: str = None, 
                        handle_missing: bool = True,
                        encode_cat: bool = True,
                        handle_outliers_flag: bool = False,
                        scale: bool = False) -> pd.DataFrame:
        """自動前処理パイプライン"""
        df_processed = df.copy()
        self.original_columns = df.columns.tolist()
        
        # ターゲット列を一時的に分離
        target = None
        if target_column and target_column in df_processed.columns:
            target = df_processed[target_column].copy()
            df_processed = df_processed.drop(columns=[target_column])
        
        # 数値列とカテゴリ列を識別
        self.numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 欠損値処理
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed, strategy='auto')
        
        # カテゴリ変数エンコーディング
        if encode_cat and self.categorical_columns:
            df_processed = self.encode_categorical(df_processed, self.categorical_columns)
        
        # 異常値処理
        if handle_outliers_flag:
            df_processed = self.handle_outliers(df_processed, self.numeric_columns)
        
        # スケーリング
        if scale:
            df_processed = self.scale_features(df_processed, method='standard')
        
        # ターゲット列を戻す
        if target is not None:
            df_processed[target_column] = target.values
            
        return df_processed
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習時と同じ変換を新しいデータに適用"""
        df_processed = df.copy()
        
        # 欠損値処理
        for column, imputer in self.imputers.items():
            if column in df_processed.columns:
                df_processed[column] = imputer.transform(df_processed[[column]])
        
        # カテゴリ変数エンコーディング
        for column, le in self.label_encoders.items():
            if column in df_processed.columns:
                df_processed[column] = df_processed[column].astype(str)
                # 未知のカテゴリを処理
                known_classes = set(le.classes_)
                df_processed[column] = df_processed[column].apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                df_processed[column] = le.transform(df_processed[column])
        
        # スケーリング
        if self.scaler is not None:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
            
        return df_processed


def get_data_summary(df: pd.DataFrame) -> dict:
    """データの要約統計を取得"""
    summary = {
        'basic_stats': df.describe().to_dict(),
        'info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
    }
    return summary
