import numpy as np

test = np.arange(27).reshape(3,3,3)
# test = test.reshape(12)
# print(test)
ind = np.unravel_index(np.argmax(test, axis=None), test.shape)
print(ind)
test[ind]