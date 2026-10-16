"""Generate synthetic sample datasets for testing and demonstration."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

OUT_DIR = Path(__file__).parent


def make_sales(n: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    regions = ["North", "South", "East", "West"]
    products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Service Z"]
    channels = ["Online", "Retail", "B2B"]

    df = pd.DataFrame({
        "Order_ID": [f"ORD-{i:05d}" for i in range(1, n + 1)],
        "Order_Date": dates,
        "Region": rng.choice(regions, n),
        "Product": rng.choice(products, n),
        "Channel": rng.choice(channels, n),
        "Units_Sold": rng.integers(1, 200, n),
        "Unit_Price": rng.uniform(10, 500, n).round(2),
        "Discount_Pct": rng.uniform(0, 30, n).round(1),
        "Customer_Rating": rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[0.05, 0.1, 0.2, 0.35, 0.25, 0.05]),
    })
    df["Revenue"] = (df["Units_Sold"] * df["Unit_Price"] * (1 - df["Discount_Pct"] / 100)).round(2)
    # Add some missing values
    idx_miss = rng.choice(n, 20, replace=False)
    df.loc[idx_miss, "Channel"] = np.nan
    # Add a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def make_operations(n: int = 400) -> pd.DataFrame:
    plants = ["Plant A", "Plant B", "Plant C"]
    shifts = ["Morning", "Afternoon", "Night"]
    df = pd.DataFrame({
        "Production_ID": [f"PROD-{i:04d}" for i in range(1, n + 1)],
        "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "Plant": rng.choice(plants, n),
        "Shift": rng.choice(shifts, n),
        "Output_Units": rng.integers(500, 2000, n),
        "Defect_Count": rng.integers(0, 50, n),
        "Machine_Downtime_Hrs": rng.uniform(0, 8, n).round(2),
        "Energy_Consumed_MWh": rng.uniform(5, 40, n).round(2),
        "Labor_Hours": rng.uniform(50, 200, n).round(1),
        "OEE_Pct": rng.uniform(50, 98, n).round(1),
        "Maintenance_Flag": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })
    df["Yield_Pct"] = ((df["Output_Units"] - df["Defect_Count"]) / df["Output_Units"] * 100).round(2)
    # Some missing
    df.loc[rng.choice(n, 15, replace=False), "OEE_Pct"] = np.nan
    return df


def make_energy(n: int = 300) -> pd.DataFrame:
    sites = ["Refinery A", "Refinery B", "Field Operations", "Terminal"]
    fuel_types = ["Natural Gas", "Diesel", "HFO", "Electricity"]
    df = pd.DataFrame({
        "Record_ID": [f"ENG-{i:04d}" for i in range(1, n + 1)],
        "Period": pd.period_range("2021-01", periods=n, freq="M").astype(str),
        "Site": rng.choice(sites, n),
        "Fuel_Type": rng.choice(fuel_types, n),
        "Production_Volume_bbl": rng.integers(5000, 50000, n),
        "Energy_Consumed_GJ": rng.uniform(500, 5000, n).round(1),
        "CO2_Emissions_tonne": rng.uniform(50, 500, n).round(1),
        "Flaring_Volume_m3": rng.uniform(0, 1000, n).round(1),
        "Water_Injected_m3": rng.uniform(100, 5000, n).round(1),
        "OPEX_USD": rng.uniform(100000, 2000000, n).round(0),
        "Uptime_Pct": rng.uniform(70, 99.9, n).round(1),
        "Safety_Incidents": rng.integers(0, 5, n),
    })
    df["Energy_Intensity"] = (df["Energy_Consumed_GJ"] / df["Production_Volume_bbl"]).round(4)
    df["Emission_Intensity"] = (df["CO2_Emissions_tonne"] / df["Production_Volume_bbl"]).round(4)
    # Some missing
    df.loc[rng.choice(n, 10, replace=False), "Flaring_Volume_m3"] = np.nan
    return df


def generate_all(output_dir: Path = OUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sales = make_sales()
    ops = make_operations()
    energy = make_energy()

    sales.to_csv(output_dir / "synthetic_sales.csv", index=False)
    ops.to_csv(output_dir / "synthetic_operations.csv", index=False)
    energy.to_csv(output_dir / "synthetic_energy.csv", index=False)

    # Also save as Excel for workbook testing
    with pd.ExcelWriter(output_dir / "synthetic_workbook.xlsx", engine="openpyxl") as writer:
        sales.to_excel(writer, sheet_name="Sales", index=False)
        ops.to_excel(writer, sheet_name="Operations", index=False)
        energy.to_excel(writer, sheet_name="Energy", index=False)

    print(f"Generated sample data in {output_dir}")


if __name__ == "__main__":
    generate_all()
