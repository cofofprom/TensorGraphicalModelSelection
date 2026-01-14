import numpy as np
import numpy.linalg as la
from sklearn.covariance import graphical_lasso
from scipy.linalg import sqrtm
from scipy.stats import norm
from utils import matrix2Edges

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

def vectorTests(data, pred_omega, alpha):
    n_samples, dim = data.shape
    regression_params = []

    for i in range(dim):
        regression_param = -np.delete(pred_omega[i, :], i) / pred_omega[i, i]
        regression_params.append(regression_param)

    regression_params = np.asarray(regression_params)

    residuals = np.zeros(data.shape, dtype=np.float64) 

    for l, sample in enumerate(data):
        for i in range(dim):
            residual = sample[i] - (np.delete(sample[:], i).T @ regression_params[i])
            residuals[l, i] = residual

    residual_cov = np.cov(residuals, rowvar=False)

    statistic = np.zeros(residual_cov.shape, dtype=np.float64)

    for i in range(dim):
        for j in range(i + 1, dim):            
            bias_correction = residual_cov[i, i] * regression_params[j, i] + residual_cov[j, j] * regression_params[i, j - 1]
            stat = residual_cov[i, j] + bias_correction
            norm_stat = np.sqrt((n_samples - 1) / (residual_cov[i, i] * residual_cov[j, j])) * stat
            statistic[i, j] = norm_stat

    return matrix2Edges((2 * norm.sf(np.abs(statistic)) < alpha).astype(int))
        