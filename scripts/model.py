from utils import generateDominantDiagonal

import numpy as np
import numpy.linalg as la
import tensorly as tl
from scipy.linalg import sqrtm
from scipy.stats import chi2

class TensorGraphicalModel:
    """
    Class that represents tensor graphical model.
    It's used to group way covarianes, precisions and sample from tensor distribution.
    """

    def __init__(self, dims, densities, precision_generate_fn=generateDominantDiagonal):
        if len(dims) != len(densities): raise ValueError("Both dims and densities have to be of the same len")
            
        self.order = len(dims)
        self.dims = dims
        self.densities = densities

        self.generatePrecisions(precision_generate_fn)

    def generatePrecisions(self, generate_fn):
        precisions = [generate_fn(self.dims[k], self.densities[k]) for k in range(self.order)]
        self.updatePrecisions(precisions)

    def updatePrecisions(self, new_precisions):
        self.precisions = new_precisions
        self.covariances = [la.inv(self.precisions[k]) for k in range(self.order)]
        self.sqrt_cache = [sqrtm(self.covariances[k]) for k in range(self.order)]

    def rvs(self, size=1, eps=0., dof=3):
        def sample():
            Z = tl.tensor(np.random.randn(*self.dims))
            Z = tl.tenalg.multi_mode_dot(Z, self.sqrt_cache)
            return Z

        samples = list()

        for _ in range(size):
            Z = sample()

            if np.random.uniform(0, 1) < eps:
                s = chi2.rvs(dof)
                r = np.sqrt(dof / s)

                samples.append(r * Z)
            else:
                samples.append(Z)
                

        return np.asarray(samples)
