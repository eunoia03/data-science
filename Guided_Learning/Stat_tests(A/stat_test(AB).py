'''
This document outlines introductory statistical tests.  Tests include: chi squared, one sample t-test, independent samples, and paired
samples.
Note: Since all the tests were combined into one file, all the variable names are the same and 
you will have to run partial code at a time to get the correct results

Just as a quick reminder, p - value represents the probability of achieving the result as extreme as the one observed, assuming that
the null hypothesis is true

'''



'''
Chi Square Test for independence
'''

import pandas as pd
from scipy.stats import chi2_contingency, chi2

campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name = "campaign_data")

#Filter the data to not include the control group for now
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]

# Create 1 2 x 2 matrix in the form of an array
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values

mailer1_signup_rate = 123 / (252 + 123)
mailer1_signup_rate = 127 / (209 + 127)

#With the signup rate, we need to figure out if these two values are significantly difference
null_hypothesis = "There is no relationship between mailer type and signup rate.  They are independent"
alternate_hypothesis = "There is a relationship between mailer type and signup rate. They are not independent"
acceptance_criteria = 0.05

chi2_statistic, p_value, dof, expected_value = chi2_contingency(observed_values, correction = False) #Yates correction

#Find the critical value for the test
critical_value = chi2.ppf(1 - acceptance_criteria, dof)

#printing the results
if chi2_statistic >= critical_value:
    print(f" As out chi square statistic of {chi2_statistic} is higher than our critical value of {critical_value}, we reject the null hypothesis and conclude that {null_hypothesis}")
else:
    print(f" As out chi square statistic of {chi2_statistic} is lower than our critical value of {critical_value}, we reject the null hypothesis and conclude that {null_hypothesis}")

if p_value <= acceptance_criteria:
    print(f" As out p value of {p_value} is lower than our critical value of {acceptance_criteria}, we reject the null hypothesis and conclude that {null_hypothesis}")
else:
    print(f" As out p value of {p_value} is higher than our critical value of {acceptance_criteria}, we retain the null hypothesis and conclude that {alternate_hypothesis}")


'''
One Sample T-Test
comparing a sample mean to the population
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, norm

'''following code allows us to create random variables in the form of a normal distribution 
- loc specifies the mean as 500
- scale: standard deviation
- size: sample size
- random_state
- astype: to return whole numbers
'''
population = norm.rvs(loc = 500, scale = 100, size = 1000, random_state = 42).astype(int) 

np.random.seed(42)
sample = np.random.choice(population, 250) #create a sample of 250 data points

plt.hist(population, density = True, alpha = 0.5) #density = True means that we will be seeing a proportional frequency on the axis rather than the numbers
plt.hist(sample, density = True, alpha = 0.5)
plt.show()

population_mean = population.mean()
sample_mean = sample.mean()
print(population_mean, sample_mean)

null_hypothesis = "The mean of the sample is equal to the mean of the population"
alternate_hypothesis = "The mean of the sample is different to the mean of the population"
acceptance_criteria = 0.05

t_statistic, p_value = ttest_1samp(sample, population_mean)
print(t_statistic, p_value)

if p_value <= acceptance_criteria:
    print(f" As out p value of {p_value} is lower than our critical value of {acceptance_criteria}, we reject the null hypothesis and conclude that {alternate_hypothesis}")
else:
    print(f" As out p value of {p_value} is higher than our critical value of {acceptance_criteria}, we retain the null hypothesis and conclude that {null_hypothesis}")

'''
Independent Samples T-Test
comparing 2 sample means
'''
from scipy.stats import ttest_ind

sample_a = norm.rvs(loc = 500, scale = 100, size = 250, random_state = 42).astype(int) 
sample_b = norm.rvs(loc = 550, scale = 150, size = 100, random_state = 42).astype(int) 



plt.hist(sample_a, density = True, alpha = 0.5) 
plt.hist(sample_b, density = True, alpha = 0.5)
plt.show()

sample_a_mean = sample_a.mean()
sample_b_mean = sample_b.mean()
print(sample_a_mean, sample_b_mean)

null_hypothesis = "The mean of sample a is equal to the mean of sample b"
alternate_hypothesis = "The mean of the sample is different to the mean of sample b"
acceptance_criteria = 0.05

t_statistic, p_value = ttest_ind(sample_a, sample_b)
print(t_statistic, p_value)

if p_value <= acceptance_criteria:
    print(f" As our p value of {p_value} is lower than our critical value of {acceptance_criteria}, we reject the null hypothesis and conclude that {alternate_hypothesis}")
else:
    print(f" As our p value of {p_value} is higher than our critical value of {acceptance_criteria}, we retain the null hypothesis and conclude that {null_hypothesis}")


#Welch's T-Test
t_statistic, p_value = ttest_ind(sample_a, sample_b, equal_var = False)
print(t_statistic, p_value)

if p_value <= acceptance_criteria:
    print(f" As our p value of {p_value} is lower than our critical value of {acceptance_criteria}, we reject the null hypothesis and conclude that {alternate_hypothesis}")
else:
    print(f" As our p value of {p_value} is higher than our critical value of {acceptance_criteria}, we retain the null hypothesis and conclude that {null_hypothesis}")

'''
Paired Samples T-Test
'''
from scipy.stats import ttest_rel

before = norm.rvs(loc = 500, scale = 100, size = 100, random_state = 42).astype(int) 
np.random.seed(42)
after = before + np.random.randint(low = -50, high = 75, size = 100)



plt.hist(before, density = True, alpha = 0.5, label = "before") 
plt.hist(after, density = True, alpha = 0.5, label = "after")
plt.legend()
plt.show()

before_mean = before.mean()
after_mean = after.mean()
print(before_mean, after_mean)

null_hypothesis = "The mean of before sample is equal to the mean of after"
alternate_hypothesis = "The mean of the before sample is different to the mean of after sample "
acceptance_criteria = 0.05

t_statistic, p_value = ttest_rel(before, after)
print(t_statistic, p_value)

if p_value <= acceptance_criteria:
    print(f" As our p value of {p_value} is lower than our acceptance criteria of {acceptance_criteria}, we reject the null hypothesis and conclude that {alternate_hypothesis}")
else:
    print(f" As our p value of {p_value} is higher than our acceptance criteria of {acceptance_criteria}, we retain the null hypothesis and conclude that {null_hypothesis}")












