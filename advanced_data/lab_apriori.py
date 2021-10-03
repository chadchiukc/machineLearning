import pandas as pd
import numpy as np
from apyori import apriori


df = pd.read_csv('Lab_data2.csv')
print(df['Drug'].unique())
df = df.replace({None: 'drugY',
                 'DrugA': 'drugA',
                 'DrugB': 'drugB',
                 'DrugC': 'drugC',
                 'DrugX': 'drugX',
                 'DrugY': 'drugY',
                 'drugsX': 'drugX',
                 'drugsY': 'drugY'})
drug_count = df.Drug.value_counts()
drug_percent = df.Drug.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
drug_cp = pd.DataFrame({'counts': drug_count, '%': drug_percent})
print(drug_cp)

drug_basket = df.groupby(["ID", "Drug"]).size().reset_index(name="Count")
my_basket = (drug_basket.groupby(['ID', 'Drug'])['Count'].sum().unstack().reset_index().fillna(0).set_index('ID'))


def encode_data(datapoint):
    if datapoint <= 0:
        return 0
    else:
        return 1


my_basket = my_basket.applymap(encode_data)
print(my_basket)
drug_itemset = apriori(my_basket, min_support=0.01)
drug_itemset = list(drug_itemset)
print(drug_itemset)