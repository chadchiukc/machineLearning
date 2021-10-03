import pandas as pd
import pycountry
import geopandas

WorldHappiness = pd.read_csv("merged-world-happiness-dataset.csv")

df = WorldHappiness.copy() # to protect main

df = df.groupby('Country name')['Ladder score'].mean()
print(df)

def alpha3code(column):
    CODE=[]
    for country in column:
        try:
            code=pycountry.countries.get(name=country).alpha_3
           # .alpha_3 means 3-letter country code
           # .alpha_2 means 2-letter country code
            CODE.append(code)
        except:
            CODE.append('None')
    return CODE

