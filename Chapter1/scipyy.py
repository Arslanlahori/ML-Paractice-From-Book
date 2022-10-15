from scipy import sparse
import numpy as np
eye = np.eye(5)
print(eye)
# convert numpy array to scipy sparse matrix in CSR format
sparse_matrix = sparse.csc_matrix(eye)
print(sparse_matrix)
# coo representation
data = np.ones(4)
row = np.arange(4)
col = np.arange(4)
coo = sparse.coo_matrix((data, (row, col)))
print("coo representaion", coo)
