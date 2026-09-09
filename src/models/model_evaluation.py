"""
Evaluation metrics for freight forecasting models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    Safely handles zeros in y_true by filtering them out.
    """
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute standard forecasting metrics given true and predicted series.
    Returns MAE, RMSE, and MAPE.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have the same length.")
        
    return {
        "MAE": calculate_mae(y_t, y_p),
        "RMSE": calculate_rmse(y_t, y_p),
        "MAPE": calculate_mape(y_t, y_p)
    }

def compare_models(results: List[Dict[str, any]]) -> pd.DataFrame:
    """
    Generate a comparison report across multiple models.
    
    Args:
        results: A list of dictionaries. Each dict should have a 'Model' key and metrics.
        e.g. [{'Model': 'Baseline', 'MAE': 1.5, 'RMSE': 2.0, 'MAPE': 5.0}, ...]
        
    Returns:
        A pandas DataFrame sorted by MAE.
    """
    df = pd.DataFrame(results)
    if not df.empty and 'MAE' in df.columns:
        df = df.sort_values('MAE').reset_index(drop=True)
    return df
