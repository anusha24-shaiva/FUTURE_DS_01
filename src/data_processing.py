"""
Data Processing & Feature Engineering Pipeline
Business Sales Performance Analytics (Future Interns - Task 1)
Author: Data Analytics Team
"""

import os
import pandas as pd
import numpy as np

def clean_and_process_sales_data(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    initial_rows = len(df)
    print(f"Initial record count: {initial_rows} rows, {len(df.columns)} columns")

    # 1. Clean Column Names (Standardize format: snake_case / Title Case)
    df.columns = [col.strip() for col in df.columns]

    # 2. Check and remove exact duplicate rows if any
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found and removed {duplicates} duplicate rows.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")

    # 3. Parse Dates
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    # Drop records with invalid Order Date
    invalid_dates = df["Order Date"].isna().sum()
    if invalid_dates > 0:
        print(f"Removing {invalid_dates} records with invalid dates.")
        df = df.dropna(subset=["Order Date"])

    # 4. Feature Engineering: Date & Time Dimensions
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_Month_Name"] = df["Order Date"].dt.strftime("%b")
    df["Order_Year_Month"] = df["Order Date"].dt.strftime("%Y-%m")
    df["Order_Quarter"] = "Q" + df["Order Date"].dt.quarter.astype(str)
    df["Order_Day_Of_Week"] = df["Order Date"].dt.day_name()
    df["Shipping_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # Fix any negative shipping days if anomalies exist
    df["Shipping_Days"] = df["Shipping_Days"].apply(lambda x: max(x, 0) if pd.notnull(x) else x)

    # 5. Financial & Business Metrics
    # Ensure Sales, Quantity, Discount, Profit are numeric
    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Unit Price before discount (effective revenue per unit)
    df["Unit_Price"] = np.where(df["Quantity"] > 0, df["Sales"] / df["Quantity"], 0.0)

    # Estimated Cost of Goods Sold (COGS)
    df["Cost"] = df["Sales"] - df["Profit"]

    # Profit Margin Percentage: (Profit / Sales) * 100
    df["Profit_Margin_Pct"] = np.where(
        df["Sales"] != 0,
        (df["Profit"] / df["Sales"]) * 100,
        0.0
    )

    # Business Flags
    df["Is_Profitable"] = df["Profit"] > 0
    df["Profit_Status"] = np.where(
        df["Profit"] > 0, "Profitable",
        np.where(df["Profit"] < 0, "Loss-Making", "Break-Even")
    )

    # Discount Categorization Bands
    def categorize_discount(disc):
        if disc == 0:
            return "0% (No Discount)"
        elif disc <= 0.10:
            return "1% - 10%"
        elif disc <= 0.20:
            return "11% - 20%"
        elif disc <= 0.40:
            return "21% - 40%"
        else:
            return "> 40% (Deep Discount)"

    df["Discount_Band"] = df["Discount"].apply(categorize_discount)

    # 6. Save Cleaned Dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_path}")
    print(f"Final record count: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range covered: {df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Revenue: ${df['Sales'].sum():,.2f} | Total Profit: ${df['Profit'].sum():,.2f}")
    
    return df

if __name__ == "__main__":
    raw_file = os.path.join("data", "raw", "superstore_raw.csv")
    clean_file = os.path.join("data", "processed", "superstore_cleaned.csv")
    clean_and_process_sales_data(raw_file, clean_file)
