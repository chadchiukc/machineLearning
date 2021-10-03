# source: https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# Importing the Dataset
store_data = pd.read_csv('store_data.csv', header=None)
print(store_data.head())

# Data Proprocessing
records = []
for i in range(0, 5): #i: num of transactions
    records.append([str(store_data.values[i,j]) for j in range(0, 4)]) # j: num of items
print(records)

# Applying Apriori
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=0.3, max_length=4)
association_results = list(association_rules)

# Viewing the Results
print(len(association_results))
print(association_results[0])

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    # pair = item[0]
    # items = [x for x in pair]
    # if len(items)==2 :
    #     print("Rule: " + items[0] + " -> " + items[1])
    #
    #     #second index of the inner list
    #     print("Support: " + str(item[1])) #Support = freq(A,B)/N
    #
    #     #third index of the list located at 0th
    #     #of the third index of the inner list
    #
    #     print("Confidence: " + str(item[2][0][2]))
    #     print("Lift: " + str(item[2][0][3]))
    #     print("=====================================")
    # if len(items)==3:
    for i in range(len(item[2])):
        print("Rule: " + str(item[2][i][0])+ " -> " + str(item[2][i][1]))

        #second index of the inner list
        print("Support: " + str(item[1])) #Support = freq(A,B)/N

        #third index of the list located at 0th
        #of the third index of the inner list

        print("Confidence: " + str(item[2][i][2]))
        print("Lift: " + str(item[2][i][3]))
        print("=====================================")