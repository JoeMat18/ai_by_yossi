import pandas as pd
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from agent_core.config import DATA_PATH

REQUIRED_COLUMNS = [
    "entity_name",
    "property_name",
    "tenant_name",
    "ledger_type",
    "ledger_group",
    "ledger_category",
    "ledger_code",
    "ledger_description",
    "month",
    "quarter",
    "year",
    "profit",
]

PROPERTY_COLUMN = "property_name"
TENANT_COLUMN = "tenant_name"
VALUE_COLUMN = "profit"
YEAR_COLUMN = "year"
MONTH_COLUMN = "month" 


class DatasetConfigError(RuntimeError):
    """Raised when we can't interpret the dataset"""


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetConfigError(
            f"Dataset is missing required columns: {', '.join(missing)}"
        )


@lru_cache(maxsize=1)
def get_assets_df() -> pd.DataFrame:
    """
    Load the assets dataset and cache (1 time) in memory.
    Handles '\\N' as NaN values correctly.
    """
    path: Path = DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, na_values=["\\N"], keep_default_na=True)

    _validate_columns(df)
    
    # Clean the data
    df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors='coerce').fillna(0.0)
    df[YEAR_COLUMN] = pd.to_numeric(df[YEAR_COLUMN], errors='coerce')
    
    # Ensure the month column is a string (for filtering)
    df[MONTH_COLUMN] = df[MONTH_COLUMN].astype(str)

    return df


def _filter_by_property(df: pd.DataFrame, property_query: str) -> pd.DataFrame:
    mask = df[PROPERTY_COLUMN].astype(str).str.contains(
        property_query, case=False, na=False
    )
    return df[mask]


def get_single_asset(property_query: str) -> Optional[Dict[str, Any]]:
    df = get_assets_df()
    subset = _filter_by_property(df, property_query)
    if subset.empty:
        return None

    tenants = (
        subset[TENANT_COLUMN]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    
    ledger_totals = (
        subset.groupby("ledger_type")[VALUE_COLUMN]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    prop_name = subset[PROPERTY_COLUMN].iloc[0] if not subset.empty else "Unknown"
    entity_name = subset["entity_name"].iloc[0] if not subset.empty else "Unknown"

    summary = {
        "property_name": prop_name,
        "entity_name": entity_name,
        "total_records": int(len(subset)),
        "total_profit": float(subset[VALUE_COLUMN].sum()),
        "tenants": tenants,
        "ledger_totals": ledger_totals,
    }
    return summary


def compare_assets_by_price(address1: str, address2: str) -> Dict[str, Any]:
    df = get_assets_df()

    def _aggregate_value(property_query: str) -> Dict[str, Any]:
        subset = _filter_by_property(df, property_query)
        if subset.empty:
            raise ValueError(
                f"Property with name like '{property_query}' not found in dataset."
            )
        total_value = float(subset[VALUE_COLUMN].sum())
        real_name = subset.iloc[0][PROPERTY_COLUMN]
        return {
            "property_name": real_name,
            "value": total_value,
            "records": int(len(subset)),
        }

    asset_1 = _aggregate_value(address1)
    asset_2 = _aggregate_value(address2)
    diff = asset_1["value"] - asset_2["value"]

    return {
        "asset_1": asset_1,
        "asset_2": asset_2,
        "difference": diff,
    }


def compute_total_pnl(year: Optional[int] = None, month: Optional[int] = None) -> float:
    """
    Calculates P&L (Profit & Loss).
    :param year: Year (int), e.g. 2025
    :param month: Month (int), e.g. 1 (January), 12 (December)
    """
    df = get_assets_df()
    subset = df

    # 1. Filter by year
    if year is not None:
        subset = subset[subset[YEAR_COLUMN] == year]
    
    # 2. Filter by month (if provided)
    if month is not None:
        if year is None:
            raise ValueError("To filter by month, you must also specify the year (e.g., 'January 2025').")
        
        # CSV format: "2025-M01", "2025-M12"
        # Form the search string
        month_str = f"{year}-M{month:02d}" 
        
        # Filter (exact match on month column)
        subset = subset[subset[MONTH_COLUMN] == month_str]

    if subset.empty:
        time_period = f"{year}" if month is None else f"{year}-M{month:02d}"
        raise ValueError(f"No financial records found for the period: {time_period}.")

    return float(subset[VALUE_COLUMN].sum())


def query_data_flexible(filters: Dict[str, Any] = None, action: str = "show", limit: int = 100) -> Dict[str, Any]:
    """
    Flexible data querying function that can filter and aggregate data based on any parameters.
    
    :param filters: Dictionary with column names as keys and filter values
                   Examples:
                   - {"month": "2025-M01"}
                   - {"year": 2025, "property_name": "Building 17"}
                   - {"ledger_type": "revenue"}
    :param action: "show" (return rows), "aggregate" (return summary), "count" (return count)
    :param limit: Maximum number of rows to return (default 100)
    :return: Dictionary with filtered data and metadata
    """
    df = get_assets_df()
    subset = df.copy()
    
    # Apply filters
    if filters:
        for column, value in filters.items():
            if column in subset.columns:
                if isinstance(value, str):
                    # String matching (case-insensitive, partial match)
                    subset = subset[subset[column].astype(str).str.contains(str(value), case=False, na=False)]
                else:
                    # Exact match for numbers
                    subset = subset[subset[column] == value]
    
    if subset.empty:
        return {
            "status": "no_data",
            "message": "No records found matching the filters.",
            "filters": filters,
            "count": 0
        }
    
    result = {
        "status": "success",
        "filters": filters,
        "count": len(subset),
        "total_profit": float(subset[VALUE_COLUMN].sum()),
    }
    
    if action == "show":
        # Return actual rows (limited)
        rows_to_return = subset.head(limit)
        result["rows"] = rows_to_return.to_dict(orient="records")
        result["showing"] = len(rows_to_return)
        result["total_rows"] = len(subset)
        
    elif action == "aggregate":
        # Return aggregated statistics
        result["aggregations"] = {
            "by_ledger_type": subset.groupby("ledger_type")[VALUE_COLUMN].sum().to_dict(),
            "by_property": subset.groupby("property_name")[VALUE_COLUMN].sum().to_dict() if "property_name" in subset.columns else {},
            "by_year": subset.groupby("year")[VALUE_COLUMN].sum().to_dict() if "year" in subset.columns else {},
        }
        
    elif action == "count":
        # Return counts by various dimensions
        result["counts"] = {
            "total": len(subset),
            "by_ledger_type": subset["ledger_type"].value_counts().to_dict(),
            "by_property": subset["property_name"].value_counts().to_dict() if "property_name" in subset.columns else {},
        }
    
    return result


def summarize_asset_record(record: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Property: {record.get('property_name')}")
    lines.append(f"Entity: {record.get('entity_name')}")
    lines.append(f"Total records: {record.get('total_records')}")
    lines.append(f"Net Profit (P&L): {record.get('total_profit'):,.2f}")

    tenants = record.get("tenants") or []
    if tenants:
        lines.append(f"Tenants: {', '.join(tenants)}")

    ledger_totals = record.get("ledger_totals") or {}
    if ledger_totals:
        lines.append("Breakdown by Ledger Type:")
        for ledger, value in ledger_totals.items():
            lines.append(f"  - {ledger}: {value:,.2f}")

    return "\n".join(lines)