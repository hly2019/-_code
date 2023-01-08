import numpy as np
import jittor as jt
import jsparse.nn.functional as F
import scipy
from scipy.sparse import linalg as linalg
jt.flags.nvcc_flags += '-lcusparse'
jt.flags.use_cuda = 1

size = 1024

### construct sparse matrix
mat = np.random.rand(size,size).astype(np.float32)
mat = np.where(mat > 0.995, mat, 0)
indices = np.nonzero(mat)
values = mat[indices]

### construct vector
vec = np.random.rand(size,1).astype(np.float32)

## execute sparse matrix multiply a vector
output = F.spmm(
        rows=jt.array(indices[0]), 
        cols=jt.array(indices[1]), 
        vals=jt.array(values), 
        size=(size,size), 
        mat=jt.array(vec), 
        cuda_spmm_alg=1)

print(output)

def solveMatrix(A, b, usejsparse=False):
    row = A.shape[0]
    # col = A.shape[1]
    D_inv = np.zeros((row, row, ))
    L = np.zeros((row, row, ))
    U = np.zeros((row, row, ))
    for i in range(row):
        D_inv[i][i] = 1 / A[i][i]
    for i in range(row):
        for j in range(row):
            if i > j:
                L[i][j] = A[i][j]
            elif i < j:
                U[i][j] = A[i][j]
    B = np.dot(-D_inv, (L + U))
    print("b is: {}".format(b))
    b = b.reshape(b.shape[0], 1)
    d = np.dot(D_inv, b)
    
    d = d.reshape(d.shape[0], 1)
    print("d is: {}".format(d))
    vec = np.random.rand(row, 1).astype(np.float32)
    print("vec is :{}".format(vec))
    print("shape: {},  vec:{}".format(A.shape, vec.shape))
    indices = np.nonzero(B)
    values = B[indices]
    if usejsparse == True:
        while True:
            output = F.spmm(
                rows=jt.array(indices[0]), 
                cols=jt.array(indices[1]), 
                vals=jt.array(values), 
                size=(row, row), 
                mat=jt.array(vec, dtype=jt.float32),
                cuda_spmm_alg=1)
            output = np.dot(A, vec)
            # print("output: {}".format(output))
            vec1 = output + d
            if (vec == vec1).all():
                break
            else:
                print(vec1)
                pass
            vec = vec1
        return vec.numpy().astype(np.float32)
    else: 
        vec, _ = linalg.cg(A, b)
        return vec.astype(np.float32)
    
    