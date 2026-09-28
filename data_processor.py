"""
Melbourne Housing CSV Data Processor

Usage:
    python data_processor.py

The program asks for a CSV path, validates the expected Melbourne
housing columns, cleans missing/invalid values, and saves the result to:

    ~/data/processed/<original_name>_processed.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


EXPECTED_COLUMNS = [
    "Suburb", "Address", "Rooms", "Type", "Price", "Method", "SellerG",
    "Date", "Distance", "Postcode", "Bedroom2", "Bathroom", "Car",
    "Landsize", "BuildingArea", "YearBuilt", "CouncilArea",
    "Lattitude", "Longtitude", "Regionname", "Propertycount"
]

# Domain sanity limits. Values outside these ranges become NaN and are
# subsequently replaced by a suitable median/mode.
VALID_RANGES = {
    "Rooms": (1, 10),
    "Distance": (0, 100),
    "Postcode": (3000, 3999),
    "Bedroom2": (1, 10),
    "Bathroom": (1, 8),
    "Car": (0, 10),
    "Landsize": (1, 20_000),
    "BuildingArea": (1, 5_000),
    "YearBuilt": (1800, 2026),
    "Lattitude": (-39.0, -37.0),
    "Longtitude": (143.0, 146.0),
    "Propertycount": (1, 100_000),
}


def validate_columns(df):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "The CSV does not have the expected melb_data.csv structure.\n"
            f"Missing columns: {missing}"
        )

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        print(f"Warning: extra columns found and removed: {extra}")


def replace_invalid_values(df):
    for column, (minimum, maximum) in VALID_RANGES.items():
        df[column] = pd.to_numeric(df[column], errors="coerce")

        invalid = (df[column] < minimum) | (df[column] > maximum)
        count = int(invalid.sum())

        if count:
            print(
                f"  {column}: replaced {count} invalid value(s) "
                f"outside [{minimum}, {maximum}] with NaN"
            )
            df.loc[invalid, column] = np.nan

    return df


def clean_dates(df):
    parsed = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    if parsed.isna().any():
        if parsed.notna().any():
            valid = parsed.dropna().sort_values()
            median_date = valid.iloc[len(valid) // 2]
            count = int(parsed.isna().sum())
            print(f"  Date: filled {count} invalid/missing date(s)")
            parsed = parsed.fillna(median_date)

    df["Date"] = parsed.dt.strftime("%Y-%m-%d")
    return df


def fill_missing_values(df):
    # Numeric missing values -> median.
    for column in df.select_dtypes(include=[np.number]).columns:
        count = int(df[column].isna().sum())
        if count:
            median = df[column].median()
            df[column] = df[column].fillna(median)
            print(
                f"  {column}: filled {count} missing value(s) "
                f"with median ({median:g})"
            )

    # Text/categorical missing values -> most common value.
    for column in df.select_dtypes(include=["object", "category"]).columns:
        count = int(df[column].isna().sum())
        if count:
            mode = df[column].mode(dropna=True)
            replacement = mode.iloc[0] if not mode.empty else "Unknown"
            df[column] = df[column].fillna(replacement)
            print(
                f"  {column}: filled {count} missing value(s) "
                f"with mode ('{replacement}')"
            )

    return df


def process_csv(input_path, remove_high_prices=False):
    input_file = Path(input_path).expanduser()
    if not input_file.exists(): raise FileNotFoundError(f"CSV file not found: {input_file}")
    if input_file.suffix.lower() != ".csv": raise ValueError("The input file must be a .csv file.")
    print(f"\nReading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    validate_columns(df)
    df = df[EXPECTED_COLUMNS].copy()
    print("\n1. Checking invalid values...")
    df = replace_invalid_values(df)

    # Price is never automatically range-checked or replaced.
    if remove_high_prices:
        print("\n2. Removing unusually high prices...")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        upper = df["Price"].quantile(0.99)
        mask = df["Price"] > upper
        print(f"  99th-percentile training limit: ${upper:,.2f}")
        print(f"  Removing {int(mask.sum())} price value(s) above this limit")
        df = df.loc[~mask].copy()
    else:
        print("\n2. Keeping all Price values unchanged.")

    # Price is the target, so missing Price rows cannot be used for training.
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    missing_price = int(df["Price"].isna().sum())
    if missing_price:
        print(f"  Removing {missing_price} row(s) with missing Price (target)")
        df = df.dropna(subset=["Price"])

    print("\n3. Cleaning dates...")
    df = clean_dates(df)
    print("\n4. Filling missing values...")
    df = fill_missing_values(df)
    print("\n5. Removing exact duplicate rows...")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Removed {before - len(df)} duplicate row(s)")
    remaining = int(df.isna().sum().sum())
    if remaining: raise RuntimeError(f"{remaining} missing values remain after processing.")
    # Save inside the project: <project root>/data/processed/
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}_processed.csv"
    df.to_csv(output_file, index=False)
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final shape: {df.shape}")
    print(f"Output: {output_file}")
    print(f"Remaining missing values: {remaining}")
    return output_file


def main():
    print("=" * 60)
    print("Melbourne Housing CSV Data Processor")
    print("=" * 60)
    input_path = input("\nEnter the path to the CSV file: ").strip().strip('"')
    if not input_path:
        print("No file path provided."); return
    print("\nPrice handling:")
    print("  y = remove prices above the 99th percentile (training-range filter)")
    print("  n = keep all prices; no high-price manipulation")
    while True:
        choice = input("\nRemove unusually high prices? [y/N]: ").strip().lower()
        if choice in ("", "n", "no"): remove_high_prices=False; break
        if choice in ("y", "yes"): remove_high_prices=True; break
        print("Please enter 'y' or 'n'.")
    try:
        process_csv(input_path, remove_high_prices=remove_high_prices)
    except Exception as error:
        print(f"\nERROR: {error}")


if __name__ == "__main__":
    main()
