"""
Baseline Forecaster: A simple Naive / Moving Average approach.
Used to establish a performance floor.
"""
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from src.models.base_forecaster import ForecastModel, ForecastResult, ForecastPoint
from src.models.model_evaluation import evaluate_forecast

class BaselineForecaster(ForecastModel):
    """
    A simple baseline model that predicts the future by carrying forward 
    the moving average of the last `window_size` days.
    """
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.last_value = None

    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        """
        'Training' the baseline model just means calculating the average 
        of the last `window_size` points in the training set.
        """
        if len(df_train) == 0:
            raise ValueError("Training data is empty.")
        
        # Sort just to be sure
        df_sorted = df_train.sort_values(by=date_col)
        self.last_value = df_sorted[target_col].tail(self.window_size).mean()

    def predict(self, horizon_days: int, context_df: pd.DataFrame, date_col: str, **kwargs) -> ForecastResult:
        """
        Predict flat values for the horizon.
        """
        route = kwargs.get("route", "unknown")
        vessel_type = kwargs.get("vessel_type", "unknown")
        
        # If context is provided, use the context's last values, else use the fitted last_value
        if not context_df.empty:
            df_sorted = context_df.sort_values(by=date_col)
            base_val = df_sorted[kwargs.get('target_col', df_sorted.columns[-1])].tail(self.window_size).mean()
            last_date = pd.to_datetime(df_sorted[date_col].iloc[-1])
        else:
            base_val = self.last_value
            last_date = pd.to_datetime('today')

        if base_val is None or pd.isna(base_val):
            base_val = 0.0

        series = []
        for i in range(1, horizon_days + 1):
            pred_date = last_date + timedelta(days=i)
            # Baseline doesn't provide confidence intervals easily
            series.append(ForecastPoint(date=pred_date.date(), predicted_rate=float(base_val)))

        return ForecastResult(
            route=route,
            vessel_type=vessel_type,
            horizon_days=horizon_days,
            model_used=f"Baseline_MA{self.window_size}",
            series=series,
            confidence_available=False
        )

    def evaluate(self, df_test: pd.DataFrame, target_col: str, date_col: str) -> dict:
        """Evaluate baseline on a given test set sequentially (recursive predicting 1 step or horizon)."""
        # For simplicity, we just predict flat over the length of df_test using the initial state
        df_sorted = df_test.sort_values(by=date_col).reset_index(drop=True)
        predictions = np.full(len(df_sorted), self.last_value)
        metrics = evaluate_forecast(df_sorted[target_col], pd.Series(predictions))
        metrics["Model"] = f"Baseline_MA{self.window_size}"
        return metrics

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))
