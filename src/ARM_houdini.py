import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv("../Transformed_TouristDestinations.csv")

participants = data.groupby('Username')['']