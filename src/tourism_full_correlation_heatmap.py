import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():

    # -----------------------------------
    # 1 Load dataset
    # -----------------------------------
    file_path = "Transformed_TouristDestinations.csv"
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # -----------------------------------
    # 2 Remove duplicate columns
    # -----------------------------------
    df = df.loc[:, ~df.columns.duplicated()]

    # -----------------------------------
    # 3 Select only numeric columns
    # -----------------------------------
    numeric_df = df.select_dtypes(include=["number"])

    print("Numeric columns used for correlation:", numeric_df.shape[1])

    # -----------------------------------
    # 4 Compute Pearson correlation
    # -----------------------------------
    corr_matrix = numeric_df.corr(method="pearson")

    # -----------------------------------
    # 5 Plot full correlation heatmap
    # -----------------------------------
    plt.figure(figsize=(18, 16))

    sns.heatmap(
        corr_matrix,
        cmap="viridis",
        annot=False,
        linewidths=0.3,
        center=0,
        square=True,
        cbar_kws={"label": "Pearson Correlation"}
    )

    plt.title("Full Correlation Matrix of Tourist Survey Variables", fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    # -----------------------------------
    # 6 Save heatmap
    # -----------------------------------
    output_file = "Tourism_Full_Correlation_Heatmap.png"
    plt.savefig(output_file, dpi=300)

    print("Heatmap saved as:", output_file)

    # Show the heatmap
    plt.show()


if __name__ == "__main__":
    main()