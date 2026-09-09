"""
XGBoost Forecaster: Machine Learning model using recursive strategy.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from datetime import timedelta
from src.models.base_forecaster import ForecastModel, ForecastResult, ForecastPoint
from src.models.model_evaluation import evaluate_forecast

class XGBoostForecaster(ForecastModel):
    """
    XGBoost Forecaster utilizing lag and rolling features.
    Predicts multi-step via recursive forecasting.
    """
    def __init__(self, lags=[1, 2, 3, 7], **xgb_kwargs):
        self.lags = lags
        self.model = xgb.XGBRegressor(**xgb_kwargs)
        self.feature_cols = [f"lag_{lag}" for lag in lags]
        self.is_fitted = False

    def _create_features(self, series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'y': series})
        for lag in self.lags:
            df[f'lag_{lag}'] = df['y'].shift(lag)
        return df

    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        df_sorted = df_train.sort_values(by=date_col).reset_index(drop=True)
        series = df_sorted[target_col]
        
        df_feat = self._create_features(series).dropna()
        X = df_feat[self.feature_cols]
        y = df_feat['y']
        
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, horizon_days: int, context_df: pd.DataFrame, date_col: str, **kwargs) -> ForecastResult:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        target_col = kwargs.get('target_col', context_df.columns[-1])
        df_sorted = context_df.sort_values(by=date_col).reset_index(drop=True)
        
        # We need the last max(lags) values to start predicting recursively
        history = list(df_sorted[target_col].tail(max(self.lags)).values)
        last_date = pd.to_datetime(df_sorted[date_col].iloc[-1])
        
        predictions = []
        for i in range(horizon_days):
            # Create feature vector from the end of history
            features = {}
            for lag in self.lags:
                features[f'lag_{lag}'] = [history[-lag]]
                
            X_pred = pd.DataFrame(features)
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            history.append(pred) # Append for recursive next step
            
        series_out = []
        # XGBoost doesn't provide native confidence intervals easily.
        # We assign an expanding uncertainty heuristic based on historical residual variance (simplified)
        uncertainty = np.std(history) * 0.1 # 10% of standard deviation as base uncertainty
        
        for i in range(horizon_days):
            pred_date = last_date + timedelta(days=i+1)
            pred_val = float(predictions[i])
            ci_margin = uncertainty * (1 + i * 0.05) # Expanding funnel
            series_out.append(ForecastPoint(
                date=pred_date.date(),
                predicted_rate=pred_val,
                lower_ci=pred_val - ci_margin,
                upper_ci=pred_val + ci_margin
            ))

        return ForecastResult(
            route=kwargs.get("route", "unknown"),
            vessel_type=kwargs.get("vessel_type", "unknown"),
            horizon_days=horizon_days,
            model_used=f"XGBoost_Lags_{len(self.lags)}",
            series=series_out,
            confidence_available=True
        )

    def evaluate(self, df_test: pd.DataFrame, target_col: str, date_col: str) -> dict:
        # Strictly, evaluate should be done using recursive predict on test context or via one-step ahead
        # For simplicity in this demo, we'll do one-step ahead forecasting over the test set using actual lags
        df_feat = self._create_features(df_test[target_col]).dropna()
        if len(df_feat) == 0:
            return {"Model": "XGBoost", "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
            
        X = df_feat[self.feature_cols]
        y_true = df_feat['y']
        y_pred = self.model.predict(X)
        
        metrics = evaluate_forecast(y_true, pd.Series(y_pred))
        metrics["Model"] = f"XGBoost_Lags_{len(self.lags)}"
        return metrics

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'lags': self.lags, 'feature_cols': self.feature_cols}, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.lags = data['lags']
            self.feature_cols = data['feature_cols']
            self.is_fitted = True
