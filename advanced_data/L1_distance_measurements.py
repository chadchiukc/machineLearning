# source: https://www.kdnuggets.com/2017/08/comparing-distance-measurements-python-scipy.html

import scipy.spatial.distance as dist
import numpy as np

# Prepare 2 vectors (data points) of 10 dimensions
A = [3,3,3]
B = [6,6,6]

print ('\n2 10-dimensional vectors')
print ('------------------------')
print (A)
print (B)

# Perform distance measurements
print ('\nDistance measurements with 10-dimensional vectors')
print ('-------------------------------------------------')
print ('\nEuclidean distance is', dist.euclidean(A, B))
print ('Manhattan distance is', dist.cityblock(A, B))
print ('Chebyshev distance is', dist.chebyshev(A, B))
print ('Canberra distance is', dist.canberra(A, B))
print ('Cosine distance is', dist.cosine(A, B))

# Prepare 2 vectors of 100 dimensions
AA = np.random.uniform(0, 10, 100)
BB = np.random.uniform(0, 10, 100)

# Perform distance measurements
print ('\nDistance measurements with 100-dimensional vectors')
print ('--------------------------------------------------')
print ('\nEuclidean distance is', dist.euclidean(AA, BB))
print ('Manhattan distance is', dist.cityblock(AA, BB))
print ('Chebyshev distance is', dist.chebyshev(AA, BB))
print ('Canberra distance is', dist.canberra(AA, BB))
print ('Cosine distance is', dist.cosine(AA, BB))