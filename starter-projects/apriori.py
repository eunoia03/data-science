'''
Association Rule Learning - Apriori
'''

from apyori import apriori
import pandas as pd

alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")

alcohol_transactions.drop("transaction_id", axis = 1, inplace = True)

# modify data for apriori algorithm
# it doesn't accept a pandas dataframe, it needs a list of lists

transactions_list = []

for index, row in alcohol_transactions.iterrows():
    transaction = list(row.dropna())  # for each row in our dataframe, drop missing values
    transactions_list.append(transaction)
    
# Apply Apriori algorithm
apriori_rules = apriori(transactions_list,
                        min_support = 0.003, # % of all transactions with A & B
                        min_confidence = 0.2, # of all A, what proportion includes B
                        min_lift = 3, # factor by which confidence > expected confidence
                        min_length = 2,
                        max_length = 2)

# At this point, apriori_rules is in generator format so we convert it into a list
# Generate is a special function that is used for interating through data

apriori_rules = list(apriori_rules)

apriori_rules[0]

# To add actual value, we convert it to a DataFrame

product1 = [list(rule[2][0][0]) for rule in apriori_rules]
product2 = [list(rule[2][0][1]) for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

apriori_rules_df = pd.DataFrame({ "product1": product1,
                                 "product2": product2,
                                 "support": support,
                                 "confidence": confidence,
                                 "lift": lift})

# Sort rules by descending lift
apriori_rules_df.sort_values(by = "lift", ascending = False, inplace = True)

# Search rules
apriori_rules_df[apriori_rules_df["product1"].str.contains("New Zealand")]  