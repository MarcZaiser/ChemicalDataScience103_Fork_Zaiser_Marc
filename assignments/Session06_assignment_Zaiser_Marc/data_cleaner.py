import pandas as pd
import numpy as np


def standardize_columns(df):
    """Convert column names to a consistent format."""
    cleaned_columns = []

    for col in df.columns:
        # Make lowercase, replace spaces with underscores, strip extra punctuation
        normalized = str(col).strip().lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("(", "")
        normalized = normalized.replace(")", "")
        normalized = normalized.replace("°", "")
        normalized = normalized.replace("-", "_")

        # Common aliases
        aliases = {
            "temperature_c": "temperature",
            "temp_c": "temperature",
            "temperature_celsius": "temperature",
            "temp": "temperature",
            "temp_f": "temperature",
            "temperature_f": "temperature",
            "yield_percent": "yield",
            "yield_pct": "yield",
            "product_yield": "yield",
            "time_s": "time",
            "time_sec": "time",
            "time_seconds": "time",
            "time_min": "time",
            "time_minutes": "time",
            "concentration_mol_l": "concentration",
            "conc": "concentration",
            "setup_name": "setup",
            "experiment": "setup",
        }

        cleaned_columns.append(aliases.get(normalized, normalized))

    df.columns = cleaned_columns
    return df


def clean_missing_and_inconsistent_data(df):
    """Handle missing values and inconsistent entries."""
    df = df.copy()

    # Replace common missing-value placeholders
    df = df.replace(["", " ", "nan", "NaN", "N/A", "NA", "null", "None"], np.nan)

    # Convert all numeric-like columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

    # Drop rows that are completely empty
    df = df.dropna(how="all").reset_index(drop=True)

    return df


def convert_temperature_to_celsius(df):
    """
    Ensure the temperature column is in Celsius.
    If the column name or values suggest Fahrenheit, convert to Celsius.
    """
    if "temperature" not in df.columns:
        return df

    temp_col = df["temperature"]

    # Detect likely Fahrenheit values if the column contains 'f' in name or values are unusually high
    # For safety, we only convert when values look like Fahrenheit (e.g., > 100 and not impossible for test data)
    if temp_col.max() > 120:
        df["temperature"] = (temp_col - 32) * 5 / 9

    return df


def load_and_clean_setup_data(filepath, setup_name):
    """
    Load one CSV file, standardize columns, clean missing values,
    convert temperature to Celsius, and label the setup.
    """
    df = pd.read_csv(filepath)

    # Standardize structure
    df = standardize_columns(df)
    df = clean_missing_and_inconsistent_data(df)
    df = convert_temperature_to_celsius(df)

    # Ensure a setup label exists
    df["setup"] = str(setup_name).strip().upper()

    return df


def combine_datasets(setup_a_data, setup_b_data):
    """
    Combine datasets from both setups into one dataframe.
    Adds a setup column indicating whether the row belongs to setup A or B.
    """
    # Ensure both dataframes have same columns
    missing_in_a = set(setup_b_data.columns) - set(setup_a_data.columns)
    missing_in_b = set(setup_a_data.columns) - set(setup_b_data.columns)

    if missing_in_a or missing_in_b:
        all_cols = sorted(set(setup_a_data.columns) | set(setup_b_data.columns))
        setup_a_data = setup_a_data.reindex(columns=all_cols)
        setup_b_data = setup_b_data.reindex(columns=all_cols)

    combined = pd.concat([setup_a_data, setup_b_data], ignore_index=True)

    # Make sure setup labels are consistent and uppercase
    if "setup" in combined.columns:
        combined["setup"] = combined["setup"].astype(str).str.upper()

    return combined


if __name__ == "__main__":
    setup_a_path = "/Users/marczaiser/Desktop/Studium /Masterstudium/2. Semester/DSA103_Advanced Chemical Data Science/ChemicalDataScience103_Fork_Zaiser_Marc/assignments/Session06_assignment_Zaiser_Marc/setup_A.csv"
    setup_b_path = "/Users/marczaiser/Desktop/Studium /Masterstudium/2. Semester/DSA103_Advanced Chemical Data Science/ChemicalDataScience103_Fork_Zaiser_Marc/assignments/Session06_assignment_Zaiser_Marc/setup_B.csv"

    # Load and clean each dataset
    setup_a_data = load_and_clean_setup_data(setup_a_path, "A")
    setup_b_data = load_and_clean_setup_data(setup_b_path, "B")

    # Combine both datasets into one dataframe
    combined_data = combine_datasets(setup_a_data, setup_b_data)

    # Display results
    print("Setup A cleaned data:")
    print(setup_a_data.head())

    print("\nSetup B cleaned data:")
    print(setup_b_data.head())

    print("\nCombined dataset:")
    print(combined_data.head())

    # Optional: save the combined dataset
    # combined_data.to_csv("combined_setup_data.csv", index=False)