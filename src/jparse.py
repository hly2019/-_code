import numpy as np
import jittor as jt
import jsparse.nn.functional as F
import scipy
from scipy.sparse import linalg as linalg
jt.flags.nvcc_flags += '-lcusparse'
jt.flags.use_cuda = 1


# 通过雅可比迭代法，求解线性方程组。其中雅可比每轮迭代中的Bx这一次矩阵乘法，通过jsparse库实现。
def solveMatrix(A, b, usejsparse=True):
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
    b = b.reshape(b.shape[0], 1)
    d = np.dot(D_inv, b)
    
    d = d.reshape(d.shape[0], 1)
    vec = np.random.rand(row, 1).astype(np.float32)
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
            vec1 = output + d
            if (vec == vec1).all():
                break
            vec = vec1
        return vec.numpy().astype(np.float32)
    else: 
        vec, _ = linalg.cg(A, b)
        return vec.astype(np.float32)
    
    