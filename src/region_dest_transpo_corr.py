import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Transformed_TouristDestinations.csv")

destination = [col for col in df.columns if col.startswith("what_specific_phillippine_tourist_destination_do_you_prefer")]
transport = pd.get_dummies(df['what_is_your_preferred_mode_of_travel_during_your_trip'], prefix='transport')
origin = pd.get_dummies(df['home_address:_region'], prefix='home_address')

all_three = pd.concat([origin, df[destination], transport], axis=1)

all_corr = all_three.corr(method='pearson')

region_transport = all_corr.loc[origin.columns, transport.columns]
region_destination = all_corr.loc[origin.columns, destination]
destination_and_transport = all_corr.loc[destination, transport.columns]

def makeCorrPlot(dataframe, title, colour):
    sns.heatmap(dataframe, annot=True, cmap=colour)
    plt.title(title)
    plt.show()

makeCorrPlot(region_destination, 'correlation between region and destination', 'coolwarm')
makeCorrPlot(region_transport, 'correlation between region and transport', 'rocket')
makeCorrPlot(destination_and_transport, 'correlation between destination and transport', 'rocket')


