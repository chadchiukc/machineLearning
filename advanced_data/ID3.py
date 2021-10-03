import numpy as np
import pandas as pd

eps = np.finfo(float).eps
from numpy import log2 as log
from collections import defaultdict


def find_entropy(df):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    return entropy


def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  # This gives all 'Yes' and 'No'
    variables = df[
        attribute].unique()  # This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        #         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def buildTree(df, tree=None):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name

    # Here we build our decision tree

    # Get attribute with maximum information gain
    node = find_winner(df)

    # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])

    # Create an empty dictionary to create tree
    if tree is None:
        tree = {}
        tree[node] = {}

    # We make loop to construct a tree by calling this function recursively.
    # In this we check if the subset is pure and stops if it is pure.

    for value in attValue:

        subtable = get_subtable(df, node, value)
        clValue, counts = np.unique(subtable['Eat'], return_counts=True)

        if len(counts) == 1:  # Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable)  # Calling the function recursively

    return tree


spare = ''.split(',')
a = 'a,a,b,c,c,c,b,a,a,c,a,b,b,c'.split(',')
fever = 'a,a,a,b,c,c,c,b,c,b,b,b,a,b'.split(',')
cough = 'n,n,n,n,y,y,y,n,y,y,y,n,y,n'.split(',')
brea = 'f,e,f,f,f,e,e,f,f,f,e,e,f,e'.split(',')
infected = 'n,n,y,y,y,n,y,n,y,y,y,y,y,n'.split(',')

columns = ['a', 'fever', 'cough', 'brea', 'infected']
dataset = {'a': a, 'fever': fever, 'cough': cough, 'brea': brea, 'infected': infected}
df = pd.DataFrame(dataset, columns=columns)
print(df)

print(df)
print(find_entropy(df))
for x in df.columns:
    print('{}: {} ---{}'.format(x, find_entropy_attribute(df, x), find_entropy(df) - find_entropy_attribute(df, x)))
print(find_winner(df))

# print(np.log2(6 / 8) * -6 / 8 + np.log2(2 / 8) * -2 / 8)
