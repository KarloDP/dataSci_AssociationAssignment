import pandas as pd
import numpy as np
import re

# ----------------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------------
input_file = r"src/Copy of Team06-TouristDestinations.csv"
df = pd.read_csv(input_file)

# Remove duplicate columns (Google Forms sometimes creates them)
df = df.loc[:, ~df.columns.duplicated()]

# ----------------------------------------------------------
# 2. Clean column names
# ----------------------------------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("?", "", regex=False)
)

# ----------------------------------------------------------
# 3. Helper function to parse ranges (duration/budget)
# ----------------------------------------------------------
def parse_range(value):

    if pd.isna(value):
        return pd.Series([np.nan, np.nan])

    value = str(value).lower().strip()
    value = value.replace(",", "").replace("₱", "").replace("php", "")

    # special cases
    if "day trip" in value:
        return pd.Series([1, 1])

    if "forever" in value:
        return pd.Series([np.nan, np.nan])

    # extract numbers
    numbers = re.findall(r"\d+", value)

    if len(numbers) == 0:
        return pd.Series([np.nan, np.nan])

    if "less than" in value:
        return pd.Series([0, int(numbers[0])])

    if "more than" in value or "+" in value:
        return pd.Series([int(numbers[0]), np.nan])

    if len(numbers) >= 2:
        return pd.Series([int(numbers[0]), int(numbers[1])])

    if len(numbers) == 1:
        return pd.Series([int(numbers[0]), int(numbers[0])])

    return pd.Series([np.nan, np.nan])


# ----------------------------------------------------------
# 4. Convert trip duration
# ----------------------------------------------------------
for col in df.columns:
    if "how_long" in col or "duration" in col:
        df[["trip_duration_min","trip_duration_max"]] = df[col].apply(parse_range)
        df.drop(columns=[col], inplace=True)
        print("Trip duration converted.")


# ----------------------------------------------------------
# 5. Convert preferred budget
# ----------------------------------------------------------
for col in df.columns:
    if "budget" in col:
        df[["budget_min","budget_max"]] = df[col].apply(parse_range)
        df.drop(columns=[col], inplace=True)
        print("Budget converted.")


# ----------------------------------------------------------
# 6. Detect multi-select columns
# ----------------------------------------------------------
multi_cols = [c for c in df.columns if df[c].astype(str).str.contains(";").any()]

print("Multi-select columns:", multi_cols)

# ----------------------------------------------------------
# 7. One-hot encode multi-select answers
# ----------------------------------------------------------
for col in multi_cols:

    df[col] = df[col].astype(str).str.replace("; ", ";")

    dummies = df[col].str.get_dummies(sep=";")

    dummies.columns = (
        dummies.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    dummies = dummies.add_prefix(col + "_")

    df = pd.concat([df, dummies], axis=1)

# remove original multi-select columns
df.drop(columns=multi_cols, inplace=True)

# ----------------------------------------------------------
# 8. Save cleaned dataset
# ----------------------------------------------------------
output_file = "../Transformed_TouristDestinations.csv"
df.to_csv(output_file, index=False)

print("\nProcessing complete.")
print("Saved as:", output_file)