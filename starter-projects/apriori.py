'''
Association Rule Learning - Apriori
'''

from apyori import apriori
import pandas as pd

alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")

alcohol_transactions.drop("transaction_id", axis = 1, inplace = True)

# modify data for apriori algorithm
# it doesn't accept a pandas dataframe, it needs a list of lists

