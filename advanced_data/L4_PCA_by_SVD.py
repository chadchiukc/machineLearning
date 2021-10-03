# source: https://stackoverflow.com/questions/60508233/python-implement-a-pca-using-svd

from sklearn.decomposition import PCA
import numpy as np

np.set_printoptions(precision=3)

print("======From sklearn.decomposition.PCA======")
B = np.array([[1.0, 2, 5, 3, 2], [3, 4, 9, 2, 1], [5, 6, 9 ,9 ,1]])

B1 = B.copy()
B1 -= np.mean(B1, axis=0)
n_samples = B1.shape[0]
print("B1 is B after centering:")
print(B1)

cov_mat = np.cov(B1.T)
pca = PCA(n_components=2)

X = pca.fit_transform(B1)
print("X")
print(X)

eigenvecmat = []
print("Eigenvectors:")
for eigenvector in pca.components_:
    if len(eigenvecmat) == 0:
        eigenvecmat = eigenvector
    else:
        eigenvecmat = np.vstack((eigenvecmat, eigenvector))
    print(eigenvector)
print("eigenvector-matrix")
print(eigenvecmat)

print("CHECK FOR PCA:")
print("X * eigenvector-matrix (=B1)")
print(np.dot(X, eigenvecmat))

print("B1 is B after centering:")
print(B1)

from numpy.linalg import svd

U, S, Vt = svd(B1, full_matrices=True)
print('======From numpy.linalg.svd======')
print("U:")
print(U)
print("S used for building Sigma:")
print(S)
Sigma = np.zeros(B.shape, dtype=float)
Sigma[:2, :2] = np.diag(S)
print("Sigma:")
print(Sigma)
print("V already transposed:")
print(Vt)
print("CHECK FOR SVD:")
print("U * Sigma * Vt (=B1)")
print(np.dot(U, np.dot(Sigma, Vt)))

# source: https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
# choosing number of components k
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(B)

# 95% of variance
from sklearn.decomposition import PCA

variance_preserve_fraction = 0.95
pca = PCA(n_components=variance_preserve_fraction)
pca.fit(data_rescaled)
reduced = pca.transform(data_rescaled)
pca = PCA().fit(data_rescaled)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 6)

fig, ax = plt.subplots()
xi = np.arange(1, len(pca.explained_variance_ratio_) + 1, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0, 1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, len(pca.explained_variance_ratio_) + 1,
                     step=1))  # change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, f'{variance_preserve_fraction * 100}% cut-off threshold', color='red', fontsize=16)

ax.grid(axis='x')
plt.show()
