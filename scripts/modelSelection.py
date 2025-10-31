import numpy as np
import numpy.linalg as la
from sklearn.covariance import graphical_lasso
from scipy.linalg import sqrtm

def tlasso(data, reg_param, T=100):
    thres = 1e-6
    
    data = np.asarray(data)
    dims = data.shape
    K = len(dims) - 1
    m_vec = dims[1:]
    n = dims[0]
    
    Omega_list = [np.eye(m) for m in m_vec]
    Omega_list_sqrt = [np.eye(m) for m in m_vec]
    
    for iter_idx in range(T):
        Omega_list_old = [O.copy() for O in Omega_list]
        for k in range(K):
            Omega_list_sqrt[k] = np.eye(m_vec[k])
            S_array = np.zeros((n, m_vec[k], m_vec[k]))
    
            for i in range(n):
                sample = data[i, ...]
                transformed_tensor = sample.copy()
                for j in range(K):
                    transformed_tensor = np.tensordot(
                        Omega_list_sqrt[j], transformed_tensor, axes=(1, j)
                    )
                    transformed_tensor = np.moveaxis(transformed_tensor, 0, j)
                
                unfolded = np.moveaxis(transformed_tensor, k, 0)
                unfolded = unfolded.reshape((m_vec[k], -1))
                S_array[i] = unfolded @ unfolded.T
            
            S_mat = np.mean(S_array, axis=0) * (m_vec[k] / np.prod(m_vec))
            _, precision = graphical_lasso(S_mat, reg_param, max_iter=100, tol=1e-4)
            precision /= la.norm(precision)
    
            Omega_list[k] = precision
            Omega_list_sqrt[k] = sqrtm(precision)
    
        diff = sum(
            la.norm(Omega_list_old[i] - Omega_list[i], 'fro')
            for i in range(K)
        )
    
        if diff < thres:
            break
            
    return Omega_list
