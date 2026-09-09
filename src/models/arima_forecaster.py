"""
ARIMA Forecaster: Statistical time-series model.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from statsmodels.tsa.arima.model import ARIMA
from src.models.base_forecaster import ForecastModel, ForecastResult, ForecastPoint
from src.models.model_evaluation import evaluate_forecast

class ARIMAForecaster(ForecastModel):
    """
    ARIMA Model for univariate freight rate forecasting.
    Provides confidence intervals natively.
    """
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model_fit = None

    def fit(self, df_train: pd.DataFrame, target_col: str, date_col: str) -> None:
        df_sorted = df_train.sort_values(by=date_col).set_index(date_col)
        # Ensure daily frequency if possible, else just fit on series
        series = df_sorted[target_col]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=self.order)
            self.model_fit = model.fit()

    def predict(self, horizon_days: int, context_df: pd.DataFrame, date_col: str, **kwargs) -> ForecastResult:
        if self.model_fit is None:
            raise ValueError("Model is not fitted yet.")

        route = kwargs.get("route", "unknown")
        vessel_type = kwargs.get("vessel_type", "unknown")

        # In a strict implementation, if context_df is given, we should apply it.
        # But statsmodels ARIMA handles new data via append/extend.
        # For simplicity, if we don't extend, we just forecast from end of training data.
        # We will assume context_df is the data we've seen up to the prediction point.
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not context_df.empty:
                # We can't trivially pass context to statsmodels without refitting or using append
                # A quick workaround for the demo is to re-instantiate and fit on context
                series = context_df.sort_values(by=date_col).set_index(date_col)[kwargs.get('target_col', context_df.columns[-1])]
                model = ARIMA(series, order=self.order)
                temp_fit = model.fit()
                forecast_res = temp_fit.get_forecast(steps=horizon_days)
                last_date = pd.to_datetime(context_df[date_col].iloc[-1])
            else:
                forecast_res = self.model_fit.get_forecast(steps=horizon_days)
                last_date = pd.to_datetime('today')

        pred_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05) # 95% CI

        series_out = []
        for i in range(horizon_days):
            pred_date = last_date + pd.Timedelta(days=i+1)
            series_out.append(ForecastPoint(
                date=pred_date.date(),
                predicted_rate=float(pred_mean.iloc[i]),
                lower_ci=float(conf_int.iloc[i, 0]),
                upper_ci=float(conf_int.iloc[i, 1])
            ))

        return ForecastResult(
            route=route,
            vessel_type=vessel_type,
            horizon_days=horizon_days,
            model_used=f"ARIMA_{self.order}",
            series=series_out,
            confidence_available=True
        )

    def evaluate(self, df_test: pd.DataFrame, target_col: str, date_col: str) -> dict:
        df_sorted = df_test.sort_values(by=date_col).reset_index(drop=True)
        # We perform a single multi-step forecast over the length of the test set
        preds = self.model_fit.forecast(steps=len(df_sorted))
        metrics = evaluate_forecast(df_sorted[target_col], pd.Series(preds.values))
        metrics["Model"] = f"ARIMA_{self.order}"
        return metrics

    def save(self, filepath: str) -> None:
        if self.model_fit:
            self.model_fit.save(filepath)

    def load(self, filepath: str) -> None:
        from statsmodels.tsa.arima.model import ARIMAResults
        self.model_fit = ARIMAResults.load(filepath)
