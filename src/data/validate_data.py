"""
Data validation module for CharterAI.
"""
import pandas as pd
from typing import List, Dict, Any
from src.utils.logging import get_logger

logger = get_logger(__name__)

def check_required_columns(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """Check if all expected columns are present in the DataFrame."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        logger.error(f"Validation failed: Missing required columns: {missing}")
        return False
    return True

def check_missing_values(df: pd.DataFrame, threshold_pct: float = 0.1) -> Dict[str, float]:
    """
    Check for missing values in the DataFrame.
    Logs a warning if missing values for any column exceed the threshold.
    """
    missing_pct = df.isnull().mean().to_dict()
    warnings = {}
    for col, pct in missing_pct.items():
        if pct > threshold_pct:
            logger.warning(f"Column '{col}' has {pct:.1%} missing values (threshold: {threshold_pct:.1%})")
            warnings[col] = pct
    return warnings

def check_duplicates(df: pd.DataFrame, subset: List[str] = None) -> int:
    """
    Check for duplicate rows.
    Returns the number of duplicate rows found.
    """
    duplicates = df.duplicated(subset=subset).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows based on subset {subset}")
    else:
        logger.info(f"No duplicate rows found based on subset {subset}")
    return duplicates

def validate_dates(df: pd.DataFrame, date_columns: List[str]) -> bool:
    """Check if specified columns contain valid dates."""
    valid = True
    for col in date_columns:
        if col not in df.columns:
            continue
        try:
            # Check if parsing succeeds without errors (coerce puts NaT for errors)
            parsed = pd.to_datetime(df[col], errors='coerce')
            invalid_count = parsed.isna().sum() - df[col].isna().sum()
            if invalid_count > 0:
                logger.error(f"Column '{col}' contains {invalid_count} invalid date formats.")
                valid = False
        except Exception as e:
            logger.error(f"Failed to validate dates in '{col}': {e}")
            valid = False
    return valid

def generate_quality_report(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Generate a basic data quality report."""
    return {
        "dataset": dataset_name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values_count": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": df.dtypes.astype(str).to_dict()
    }
