import numpy as np
import scipy.linalg as la
from scipy.stats.mstats import gmean
import operator

criteria = {1: 'Apple', 2: 'Orange', 3: 'Banana'}
A = np.array([[1, 3, 7, 6], [1 / 3, 1, 5, 4], [1 / 7, 1 / 5, 1, 1 / 3], [1 / 6, 1 / 4, 3, 1]])
print("The matrix of A is: \n", A)

print("\n------Eigen system approach------\n")

eigvals, eigvecs = la.eig(A)
print("The eigenvalues of A are: \n", eigvals)
print("The eigenvectors of A are: \n", eigvecs)

real_eigenvalues = eigvals[eigvals.real == eigvals].max()
real_idx = np.where(real_eigenvalues == eigvals)

print("The real eigenvalues of A are: \n", eigvals[real_idx])

# real_eigenvectors = eigvecs[eigvecs.real == eigvecs]
print("The real eigenvectors of A are: \n", eigvecs.T[real_idx])

CI = (eigvals[real_idx] - len(A)) / (len(A) - 1)

print("\nConsistency Index: ", CI)

RI = [0, 0, 0, 0.52, 0.89, 1.11, 1.25, 1.35, 1.4]

CR = CI / RI[len(A)]

print("Consistency Ratio: ", CR)

if CR < 0.1:
    print("\nConsistency Ratio < 0.1, the survey is acceptable.")
    norm = eigvecs.T[real_idx] / np.sum(eigvecs.T[real_idx])
    print("The Priority Vector of {} is: \n".format([criteria[i] for i in range(1, len(A) + 1)]), norm)
    index, value = max(enumerate(norm), key=operator.itemgetter(1))
    print("")
else:
    print("Consistency Ratio >= 0.1, the survey needs to be revised again.")

print("\n------Geometric means approach------\n")

print("The geometric mean of A are: \n", gmean(A, axis=1))
norm2 = gmean(A, axis=1) / np.sum(gmean(A, axis=1))

print("The normalized real eigenvectors of A are: \n", norm2)
