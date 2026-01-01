"""
モデルトレーニングモジュール - XGBoost/LightGBM/CatBoost対応、パラメータチューニング、交差検証、スタッキング
"""
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
from optuna.samplers import TPESampler
import joblib
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optunaのログを抑制
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """モデルトレーニングクラス"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.best_params = None
        self.cv_scores = None
        self.feature_importance = None
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.cat_features = None  # CatBoost用カテゴリ特徴量インデックス
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """データの分割"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _objective_xgboost(self, trial, X, y, cv_folds):
        """XGBoost用Optuna目的関数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            **params
        )
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        return -cv_scores.mean()
    
    def _objective_lightgbm(self, trial, X, y, cv_folds):
        """LightGBM用Optuna目的関数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        }
        
        model = LGBMRegressor(
            objective='regression',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params
        )
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        return -cv_scores.mean()
    
    def _objective_catboost(self, trial, X, y, cv_folds):
        """CatBoost用Optuna目的関数"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'border_count': trial.suggest_int('border_count', 32, 255),
        }
        
        model = CatBoostRegressor(
            loss_function='RMSE',
            random_state=42,
            verbose=False,
            **params
        )
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        return -cv_scores.mean()
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                            model_type: str = 'xgboost',
                            n_trials: int = 50,
                            cv_folds: int = 5,
                            progress_callback=None) -> dict:
        """Optunaによるハイパーパラメータチューニング"""
        self.model_type = model_type
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        if model_type == 'xgboost':
            objective = lambda trial: self._objective_xgboost(trial, X, y, cv_folds)
        elif model_type == 'lightgbm':
            objective = lambda trial: self._objective_lightgbm(trial, X, y, cv_folds)
        else:  # catboost
            objective = lambda trial: self._objective_catboost(trial, X, y, cv_folds)
        
        # プログレスバー付きで最適化
        if progress_callback:
            def callback(study, trial):
                progress_callback(trial.number / n_trials)
            study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
        else:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.optimization_history = study.trials_dataframe()
        
        return {
            'best_params': self.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials,
            'optimization_history': self.optimization_history
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              model_type: str = 'xgboost',
              params: dict = None,
              use_default_params: bool = False,
              cat_features: list = None) -> object:
        """モデルの学習"""
        self.model_type = model_type
        self.cat_features = cat_features
        
        if params is None and self.best_params is not None:
            params = self.best_params
        elif params is None and use_default_params:
            if model_type == 'xgboost':
                params = {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.1,
                }
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.1,
                    'num_leaves': 31,
                }
            else:  # catboost
                params = {
                    'iterations': 500,
                    'depth': 6,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 3,
                    'bagging_temperature': 0.5,
                    'random_strength': 0.5,
                }
        
        if model_type == 'xgboost':
            self.model = XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                **params
            )
            self.model.fit(X_train, y_train)
        elif model_type == 'lightgbm':
            self.model = LGBMRegressor(
                objective='regression',
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **params
            )
            self.model.fit(X_train, y_train)
        else:  # catboost
            self.model = CatBoostRegressor(
                loss_function='RMSE',
                random_state=42,
                verbose=False,
                **params
            )
            if cat_features:
                self.model.fit(X_train, y_train, cat_features=cat_features)
            else:
                self.model.fit(X_train, y_train)
        
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return self.model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
        """交差検証の実行"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'mape': []
        }
        
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 同じパラメータで新しいモデルを作成
            if self.model_type == 'xgboost':
                fold_model = XGBRegressor(**self.model.get_params())
                fold_model.fit(X_cv_train, y_cv_train)
            elif self.model_type == 'lightgbm':
                fold_model = LGBMRegressor(**self.model.get_params())
                fold_model.fit(X_cv_train, y_cv_train)
            else:  # catboost
                fold_model = CatBoostRegressor(**self.model.get_params())
                if self.cat_features:
                    fold_model.fit(X_cv_train, y_cv_train, cat_features=self.cat_features)
                else:
                    fold_model.fit(X_cv_train, y_cv_train)
            
            y_pred = fold_model.predict(X_cv_val)
            
            cv_results['rmse'].append(np.sqrt(mean_squared_error(y_cv_val, y_pred)))
            cv_results['mae'].append(mean_absolute_error(y_cv_val, y_pred))
            cv_results['r2'].append(r2_score(y_cv_val, y_pred))
            cv_results['mape'].append(mean_absolute_percentage_error(y_cv_val, y_pred) * 100)
            
            fold_predictions.append({
                'fold': fold + 1,
                'y_true': y_cv_val.values,
                'y_pred': y_pred
            })
        
        self.cv_scores = {
            'rmse_mean': np.mean(cv_results['rmse']),
            'rmse_std': np.std(cv_results['rmse']),
            'mae_mean': np.mean(cv_results['mae']),
            'mae_std': np.std(cv_results['mae']),
            'r2_mean': np.mean(cv_results['r2']),
            'r2_std': np.std(cv_results['r2']),
            'mape_mean': np.mean(cv_results['mape']),
            'mape_std': np.std(cv_results['mape']),
            'fold_results': cv_results,
            'fold_predictions': fold_predictions
        }
        
        return self.cv_scores
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """モデルの評価"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        self.y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, self.y_pred)),
            'mae': mean_absolute_error(y_test, self.y_pred),
            'r2': r2_score(y_test, self.y_pred),
            'mape': mean_absolute_percentage_error(y_test, self.y_pred) * 100,
            'model_score': self.model.score(X_test, y_test)
        }
        
        # 残差の統計
        residuals = y_test - self.y_pred
        self.metrics['residuals'] = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max()
        }
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測の実行"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """モデルの保存"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'cv_scores': self.cv_scores,
            'cat_features': self.cat_features,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        return filepath
    
    def load_model(self, filepath: str):
        """モデルの読み込み"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.metrics = model_data['metrics']
        self.cv_scores = model_data.get('cv_scores')
        self.cat_features = model_data.get('cat_features')
        return model_data


class StackingTrainer:
    """スタッキングアンサンブルトレーナー"""
    
    def __init__(self):
        self.model = None
        self.base_models = {}
        self.meta_model = None
        self.feature_importance = None
        self.metrics = {}
        self.y_pred = None
        self.y_test = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              use_xgboost: bool = True,
              use_lightgbm: bool = True,
              use_catboost: bool = True,
              cv_folds: int = 5) -> object:
        """スタッキングモデルの学習"""
        
        estimators = []
        
        if use_xgboost:
            xgb = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
            estimators.append(('xgboost', xgb))
            self.base_models['xgboost'] = xgb
            
        if use_lightgbm:
            lgb = LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            estimators.append(('lightgbm', lgb))
            self.base_models['lightgbm'] = lgb
            
        if use_catboost:
            cat = CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
            estimators.append(('catboost', cat))
            self.base_models['catboost'] = cat
        
        # メタモデル（Elastic Net）
        self.meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        
        # スタッキングモデル
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=self.meta_model,
            cv=cv_folds,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # 特徴量重要度（各ベースモデルの平均）
        importance_list = []
        for name, estimator in self.model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importance_list.append(
                    pd.Series(estimator.feature_importances_, index=X_train.columns)
                )
        
        if importance_list:
            self.feature_importance = pd.concat(importance_list, axis=1).mean(axis=1).sort_values(ascending=False)
        
        return self.model
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """モデルの評価"""
        self.y_pred = self.model.predict(X_test)
        self.y_test = y_test
        
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, self.y_pred)),
            'mae': mean_absolute_error(y_test, self.y_pred),
            'r2': r2_score(y_test, self.y_pred),
            'mape': mean_absolute_percentage_error(y_test, self.y_pred) * 100,
            'model_score': self.model.score(X_test, y_test)
        }
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測の実行"""
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """モデルの保存"""
        model_data = {
            'model': self.model,
            'model_type': 'stacking',
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        return filepath


def compare_models(X_train, X_test, y_train, y_test, 
                   params_xgb=None, params_lgb=None, params_cat=None,
                   include_stacking: bool = False) -> dict:
    """XGBoost、LightGBM、CatBoost（+スタッキング）の比較"""
    results = {}
    
    # XGBoost
    trainer_xgb = ModelTrainer()
    if params_xgb:
        trainer_xgb.train(X_train, y_train, model_type='xgboost', params=params_xgb)
    else:
        trainer_xgb.train(X_train, y_train, model_type='xgboost', use_default_params=True)
    metrics_xgb = trainer_xgb.evaluate(X_test, y_test)
    results['xgboost'] = {
        'metrics': metrics_xgb,
        'model': trainer_xgb.model,
        'feature_importance': trainer_xgb.feature_importance,
        'predictions': trainer_xgb.y_pred
    }
    
    # LightGBM
    trainer_lgb = ModelTrainer()
    if params_lgb:
        trainer_lgb.train(X_train, y_train, model_type='lightgbm', params=params_lgb)
    else:
        trainer_lgb.train(X_train, y_train, model_type='lightgbm', use_default_params=True)
    metrics_lgb = trainer_lgb.evaluate(X_test, y_test)
    results['lightgbm'] = {
        'metrics': metrics_lgb,
        'model': trainer_lgb.model,
        'feature_importance': trainer_lgb.feature_importance,
        'predictions': trainer_lgb.y_pred
    }
    
    # CatBoost
    trainer_cat = ModelTrainer()
    if params_cat:
        trainer_cat.train(X_train, y_train, model_type='catboost', params=params_cat)
    else:
        trainer_cat.train(X_train, y_train, model_type='catboost', use_default_params=True)
    metrics_cat = trainer_cat.evaluate(X_test, y_test)
    results['catboost'] = {
        'metrics': metrics_cat,
        'model': trainer_cat.model,
        'feature_importance': trainer_cat.feature_importance,
        'predictions': trainer_cat.y_pred
    }
    
    # スタッキング（オプション）
    if include_stacking:
        stacking_trainer = StackingTrainer()
        stacking_trainer.train(X_train, y_train)
        metrics_stacking = stacking_trainer.evaluate(X_test, y_test)
        results['stacking'] = {
            'metrics': metrics_stacking,
            'model': stacking_trainer.model,
            'feature_importance': stacking_trainer.feature_importance,
            'predictions': stacking_trainer.y_pred
        }
    
    # 比較サマリー
    comparison_data = {
        'XGBoost': [metrics_xgb['rmse'], metrics_xgb['mae'], metrics_xgb['r2'], metrics_xgb['mape']],
        'LightGBM': [metrics_lgb['rmse'], metrics_lgb['mae'], metrics_lgb['r2'], metrics_lgb['mape']],
        'CatBoost': [metrics_cat['rmse'], metrics_cat['mae'], metrics_cat['r2'], metrics_cat['mape']]
    }
    
    if include_stacking:
        comparison_data['Stacking'] = [
            metrics_stacking['rmse'], metrics_stacking['mae'], 
            metrics_stacking['r2'], metrics_stacking['mape']
        ]
    
    comparison_df = pd.DataFrame(comparison_data, index=['RMSE', 'MAE', 'R²', 'MAPE (%)'])
    results['comparison_df'] = comparison_df
    
    return results


def get_best_model(comparison_results: dict, metric: str = 'r2') -> str:
    """比較結果から最良のモデルを選択"""
    models = ['xgboost', 'lightgbm', 'catboost']
    if 'stacking' in comparison_results:
        models.append('stacking')
    
    if metric == 'r2':
        # R²は高いほど良い
        best_model = max(models, key=lambda m: comparison_results[m]['metrics']['r2'])
    else:
        # RMSE, MAE, MAPEは低いほど良い
        best_model = min(models, key=lambda m: comparison_results[m]['metrics'][metric])
    
    return best_model
