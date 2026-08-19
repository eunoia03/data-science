'''
Causal Impact Analysis
'''

from causalimpact import CausalImpact
import pandas as pd

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
campaign_data = pd.read_excel("data/grocery_database.xlsx", sheet_name = "campaign_data")

# Aggregate transactions to customer, date level
customer_daily_sales = transactions.groupby(["customer_id", "transaction_date"])["sales_cost"].sum().reset_index()

# Merge on the signup flag
customer_daily_sales = pd.merge(customer_daily_sales, campaign_data, how = "inner", on = "customer_id")


# Pivot the data to aggregate daily sales by signup group
causal_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                                    columns = "signup_flag",
                                                    values = "sales_cost",
                                                    aggfunc = "mean")

# Provide a frequency for our DateTimeIndex (avoids a warning message)
# At this point, the frequency is set to none.  It will still work regardless but it will give a warning message
causal_impact_df.index.freq = "D"

# For causal impact, we need the impacted group in the first column
causal_impact_df = causal_impact_df[[1,0]] # flip around the columns

# Rename the columns to something more meaningful
causal_impact_df.columns = ["member", "non-member"]

# Apply Causal Impact
pre_period = ["2020-04-01", "2020-06-30"]
post_period = ["2020-07-01", "2020-09-01"]

ci = CausalImpact(causal_impact_df, pre_period, post_period)

# Plotting the impact
ci.plot()
'''
The vertical dot down the middle of each chart is the date of membership - everything to the left is the pre-period
The top chart is the actual data - the actual average daily sales for those that signed up for the membership
The purple area around the main lines are the confidence intervals around the counterfactual

Based on what we see on the graph, there appears to be an impact on sales caused by this membership
Customers appear to increase their daily average spend over and above what the model suggested they would have spent if the membership didn't exist
'''

# Extract the summary statistics and report
print(ci.summary())
'''
Posterior Inference {Causal Impact}
                          Average            Cumulative
Actual                    170.92             10768.19
Prediction (s.d.)         120.69 (4.55)      7603.4 (286.43)
95% CI                    [111.75, 129.57]   [7040.19, 8162.97]

Absolute effect (s.d.)    50.23 (4.55)       3164.79 (286.43)
95% CI                    [41.35, 59.17]     [2605.22, 3728.0]

Relative effect (s.d.)    41.62% (3.77%)     41.62% (3.77%)
95% CI                    [34.26%, 49.03%]   [34.26%, 49.03%]

Posterior tail-area probability p: 0.0
Posterior prob. of a causal effect: 100.0%

For more details run the command: print(impact.summary('report'))
'''

print(ci.summary(output = "report"))
'''
Analysis report {CausalImpact}


During the post-intervention period, the response variable had
an average value of approx. 170.92. By contrast, in the absence of an
intervention, we would have expected an average response of 120.69.
The 95% interval of this counterfactual prediction is [111.75, 129.57].
Subtracting this prediction from the observed response yields
an estimate of the causal effect the intervention had on the
response variable. This effect is 50.23 with a 95% interval of
[41.35, 59.17]. For a discussion of the significance of this effect,
see below.


Summing up the individual data points during the post-intervention
period (which can only sometimes be meaningfully interpreted), the
response variable had an overall value of 10768.19.
By contrast, had the intervention not taken place, we would have expected
a sum of 7603.4. The 95% interval of this prediction is [7040.19, 8162.97].


The above results are given in terms of absolute numbers. In relative
terms, the response variable showed an increase of +41.62%. The 95%
interval of this percentage is [34.26%, 49.03%].


This means that the positive effect observed during the intervention
period is statistically significant and unlikely to be due to random
fluctuations. It should be noted, however, that the question of whether
this increase also bears substantive significance can only be answered
by comparing the absolute effect (50.23) to the original goal
of the underlying intervention.


The probability of obtaining this effect by chance is very small
(Bayesian one-sided tail-area probability p = 0.0).
This means the causal effect can be considered statistically
significant.

'''