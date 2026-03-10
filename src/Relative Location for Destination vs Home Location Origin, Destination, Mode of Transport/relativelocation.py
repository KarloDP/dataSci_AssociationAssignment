import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Transformed_TouristDestinations.csv")

# Extract and clean destination columns
dest_cols = [c for c in df.columns if "what_specific_phillippine_tourist_destination_do_you_prefer" in c.lower()]
rename_dict = {
    c: c.replace("what_specific_phillippine_tourist_destination_do_you_prefer_", "").replace("_", " ").title()
    for c in dest_cols
}
df = df.rename(columns=rename_dict)
clean_dest_cols = list(rename_dict.values())
df[clean_dest_cols] = df[clean_dest_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# Create cross-tabulation
region_dest_crosstab = pd.DataFrame()
for region in df["home_address:_region"].unique():
    region_data = df[df["home_address:_region"] == region]
    dest_counts = region_data[clean_dest_cols].sum().astype(int)
    region_dest_crosstab[region] = dest_counts
region_dest_crosstab = region_dest_crosstab.T

# Calculate percentages
region_totals = df["home_address:_region"].value_counts()
region_dest_percentage = region_dest_crosstab.div(region_totals, axis=0) * 100

# Generate heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(region_dest_percentage, annot=True, fmt=".1f", cmap="YlOrRd", 
            linewidths=0.5, cbar_kws={"label": "Percentage (%)"})
plt.title("Relative Location Analysis: Preferred Destinations by Home Region (%)", 
          fontsize=16, fontweight='bold')
plt.xlabel("Tourist Destinations")
plt.ylabel("Home Region (Origin)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Relative_Location_Heatmap.png", dpi=300)