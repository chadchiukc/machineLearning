# source: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# df_old = pd.read_csv('hypothesis_test.csv')

# Example of the Analysis of Variance Test
from scipy.stats import f_oneway
print('Null hypothesis: no statistically significant difference \nAlternative hypothesis: there is a statistically significant difference')
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = f_oneway(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print(f'We fail to reject the null hypothesis as our p-value={p} > 0.05. Probably the same distribution \n')
else:
	print(f'We can reject the null hypothesis as our p-value={p} <= 0.05. Probably different distributions \n')

# Example of the Chi-Squared Test
from scipy.stats import chi2_contingency
print('Null hypothesis: no relationship \nAlternative hypothesis: there is a relationship')
table = [[10, 20, 30],[6,  9,  17]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print(f'We fail to reject the null hypothesis as our p-value={p} > 0.05. Probably independent \n')
else:
	print(f'We can reject the null hypothesis as our p-value={p} <= 0.05. Probably dependent \n')

# Example of the Student's t-test
from scipy.stats import ttest_ind
print('Null hypothesis: no statistical difference \nAlternative hypothesis: there is a statistical difference ')
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_ind(data1, data2, equal_var=False)
print('stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:
	print(f'We fail to reject the null hypothesis as our p-value={p} > 0.05. Probably the same distribution\n')
else:
	print(f'We can reject the null hypothesis as our p-value={p} <= 0.05. Probably different distributions\n')


print('Null hypothesis: no statistical difference \nAlternative hypothesis: there is a statistical difference ')
data1 = [351.1860809, 469.7613277, 347.9830783, 491.7653132, 409.2207765, 370.1781664, 425.2268617, 351.4768394, 385.1146477]
data2 = [592.659021, 451.143481, 475.5735624, 376.4565931, 315.5499]
stat, p = ttest_ind(data1, data2, equal_var=True)
print('stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:
	print(f'We fail to reject the null hypothesis as our p-value={p} > 0.05. Probably the same distribution')
else:
	print(f'We can reject the null hypothesis as our p-value={p} <= 0.05. Probably different distributions')