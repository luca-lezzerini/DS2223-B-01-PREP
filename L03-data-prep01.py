# -*- coding: utf-8 -*-

# Refer to link https://archive.ics.uci.edu/ml/datasets/bank+marketing#

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ============= DATA PREPARATION ==============

# Read dataset from CSV with ; separator
bank_train = pd.read_csv("bank-additional-full.csv", sep=";")

# Get size of read DataFrame
bank_shape = bank_train.shape
print(bank_shape)
print(bank_shape[0])

# Generate an index 
bank_train['my_index'] = pd.Series(range(0, bank_shape[0]))

# Show first rows ...
print(bank_train.head())

# Plot original pdays histogram
# bank_train['pdays'].plot(kind='hist', title="Days between last call")

bank_train['pdays'].plot(kind='hist', title="Days between last call")


# Correct misleading value 999 into NaN
bank_train['pdays'] = bank_train['pdays'].replace(999, np.NaN)

# Plot updated pdays histogram
bank_train['pdays'].plot(kind='hist', title="Days between last call")

# Convert categorical to numeric value for education
bank_train['education_numeric'] = bank_train['education']
dict_edu = { "education_numeric": {"illiterate": 0, "basic.4y":4,
                                   "basic.6y":6, "basic.9y":9,
                                   "high.school":12,"professional.course":12,
                                   "university.degree":16,
                                   "unknown":np.NaN }}
bank_train.replace(dict_edu, inplace = True)

# Plot updated pdays histogram
# bank_train['education_numeric'].plot(kind='hist', title="Education plot")

# Transform into a standard distribution the age field
bank_train['age_z'] = stats.zscore(bank_train['age'])
#bank_train['age_z'].plot(kind='hist', title="age")

# Transform into a standard distribution the age field
bank_train['campaign_z'] = stats.zscore(bank_train['campaign'])
#bank_train['campaign_z'].plot(kind='hist', title="Number of contacts Z")

# Filter outliers according to contacts (i.e. anomalies)
bank_outliers_campaign = bank_train.query("campaign_z <-3 | campaign_z > 3")

# Filter outliers according to age (i.e. anomalies)
bank_outliers_age = bank_train.query("age_z <-3 | age_z > 3")

# Sort by age
bank_train_sort =  bank_train.sort_values(['age_z'], ascending = False)

# Display only two columns
print(bank_train_sort[['age', 'marital']].head(n=15))

# ============= EXPLORATORY DATA ANAYSIS ==============

# Create a bar graph for EDA
# poutcome
# 1) Create a contingency table
crosstab_01 = pd.crosstab(bank_train['poutcome'], bank_train['y'])
# 2) draw bar graph
# crosstab_01.plot(kind = 'bar', stacked = True)

# Create a normalized bar graph
# 1) normalize the contingency table
crosstab_norm = crosstab_01.div(crosstab_01.sum(1), axis = 0)
# 2) draw the bar graph
# crosstab_norm.plot(kind = 'bar', stacked = True)

# loan
# 1) Create a contingency table
crosstab_02 = pd.crosstab(bank_train['loan'], bank_train['y'])
# 2) draw bar graph
# crosstab_01.plot(kind = 'bar', stacked = True)

# Create a normalized bar graph
# 1) normalize the contingency table
crosstab_norm02 = crosstab_02.div(crosstab_02.sum(1), axis = 0)
# 2) draw the bar graph
# crosstab_norm02.plot(kind = 'bar', stacked = True)

# education
# 1) Create a contingency table
crosstab_03 = pd.crosstab(bank_train['education'], bank_train['y'])
# 2) draw bar graph
# crosstab_01.plot(kind = 'bar', stacked = True)

# Create a normalized bar graph
# 1) normalize the contingency table
crosstab_norm03 = crosstab_03.div(crosstab_03.sum(1), axis = 0)
# 2) draw the bar graph
# crosstab_norm03.plot(kind = 'bar', stacked = True)

# contact
# 1) Create a contingency table
crosstab_04 = pd.crosstab(bank_train['contact'], bank_train['y'])
# 2) draw bar graph
# crosstab_01.plot(kind = 'bar', stacked = True)

# Create a normalized bar graph
# 1) normalize the contingency table
crosstab_norm04 = crosstab_04.div(crosstab_04.sum(1), axis = 0)
# 2) draw the bar graph
# crosstab_norm04.plot(kind = 'barh', stacked = True)

# histograms over age in overlay with response 
bt_age_y = bank_train[bank_train.y == "yes"]['age']
bt_age_n = bank_train[bank_train.y == "no"]['age']
plt.hist([bt_age_y, bt_age_n], bins = 5, stacked = True)
plt.legend(['Y = Yes', 'Y = No'])
plt.title('Histogram of Age with Y Overlay')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()







