import numpy as np
from tqdm import trange

class NMFGD:
    def __init__(self, n_components, randominit = True, eps = 1e-10):
        self.n_components = n_components
        self.wrand = randominit
        self.W = None
        self.H = None
        self.eps = eps
        self.norms = []
    
    @staticmethod
    def randominit(A, n_components):
        W = np.random.uniform(0, 1, size = (A.shape[0], n_components))
        H = np.random.uniform(0, 1, size = (n_components, A.shape[1]))
        return W, H

    def fit(self, A, maxiter = 1000):
        if self.wrand:
            self.W, self.H = self.randominit(A, self.n_components)
        else:
            self.W, self.H = self.rnadominit(A, self.n_components)
        for epoch in (t := trange(maxiter)):
            # UPDATE H
            W_TA = self.W.T @ A
            W_TWH = self.W.T @ self.W @ self.H + self.eps
            for i in range(np.size(self.H, 0)):
                for j in range(np.size(self.H, 1)):
                    self.H[i, j] = self.H[i, j] * W_TA[i, j] / W_TWH[i, j]
            
            #UPDATE W
            AH_T = A @ self.H.T
            WHH_T = self.W @ self.H @ self.H.T + self.eps
            for i in range(np.size(self.W, 0)):
                for j in range(np.size(self.W, 1)):
                    self.W[i, j] = self.W[i, j] * AH_T[i, j] / WHH_T[i, j]

            norm = np.linalg.norm(A - self.W @ self.H, 'fro')
            self.norms.append(norm)
            t.update(1)
            t.set_description_str(desc = f'Iter [{epoch + 1} / {maxiter}]')
            t.set_postfix_str(s = f'Frobenious Norm: {norm:.5f}')