# ==========================================
# TOURISM DATASET CORRELATION ANALYSIS
# PyCharm Version
# ==========================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------------
# Simplify Column Names
# ------------------------------------------
def simplify_columns(df):

    df.columns = df.columns.str.lower()

    df.columns = df.columns.str.replace(
        "what_specific_phillippine_tourist_destination_do_you_prefer_", "dest_", regex=False)

    df.columns = df.columns.str.replace(
        "what_activities_are_you_most_interested_in_during_your_trip_", "act_", regex=False)

    df.columns = df.columns.str.replace(
        "home_address:_region", "origin", regex=False)

    df.columns = df.columns.str.replace(
        "what_is_your_estimated_budget_for_this_trip_", "budget_", regex=False)

    return df


# ------------------------------------------
# Compute Pearson Correlation
# ------------------------------------------
def compute_corr_matrix(df, cols1, cols2):

    subset = df[cols1 + cols2]

    corr = subset.corr(method="pearson")

    return corr.loc[cols1, cols2]


# ------------------------------------------
# Plot Heatmap
# ------------------------------------------
def plot_heatmap(corr_matrix, title):

    plt.figure(figsize=(14, 8))

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title(title, fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.show()


# ------------------------------------------
# Main Program
# ------------------------------------------
def main():

    # Load dataset
    df = pd.read_csv("../Transformed_TouristDestinations.csv")

    # Simplify column names
    df = simplify_columns(df)

    # Identify column groups
    destination_cols = [col for col in df.columns if col.startswith("dest_")]
    activity_cols = [col for col in df.columns if col.startswith("act_")]
    budget_cols = [col for col in df.columns if col.startswith("budget")]

    # Encode origin
    origin_encoded = pd.get_dummies(df["origin"], prefix="origin")

    df = pd.concat([df, origin_encoded], axis=1)

    origin_cols = origin_encoded.columns.tolist()

    # ======================================
    # Correlation Analysis
    # ======================================

    # 1 Destination vs Activities
    dest_act_corr = compute_corr_matrix(df, destination_cols, activity_cols)
    plot_heatmap(dest_act_corr, "Destination vs Activities")

    # Complex correlations

    # Budget vs Activities
    budget_act_corr = compute_corr_matrix(df, budget_cols, activity_cols)
    plot_heatmap(budget_act_corr, "Budget vs Activities")

    # Budget vs Origin
    budget_origin_corr = compute_corr_matrix(df, budget_cols, origin_cols)
    plot_heatmap(budget_origin_corr, "Budget vs Origin")

    # Budget vs Destination
    budget_dest_corr = compute_corr_matrix(df, budget_cols, destination_cols)
    plot_heatmap(budget_dest_corr, "Budget vs Destination")

    # Activities vs Origin
    act_origin_corr = compute_corr_matrix(df, activity_cols, origin_cols)
    plot_heatmap(act_origin_corr, "Activities vs Origin")

    # Activities vs Destination
    act_dest_corr = compute_corr_matrix(df, activity_cols, destination_cols)
    plot_heatmap(act_dest_corr, "Activities vs Destination")

    # Origin vs Destination
    origin_dest_corr = compute_corr_matrix(df, origin_cols, destination_cols)
    plot_heatmap(origin_dest_corr, "Origin vs Destination")


# Run program
if __name__ == "__main__":
    main()