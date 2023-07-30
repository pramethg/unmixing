import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


class NNICA:
    def __init__(self, n_components, tol = 1e-4, maxiter = 200):
        self.n_components = n_components
        self.components_ = None
        self.mixing_ = None
        self.maxiter = maxiter
        self.tol = tol

    @staticmethod
    def whiten(X, n_components):
        X_mean = X.mean(axis = -1)
        X -= X_mean[:, np.newaxis]
        U, D = linalg.svd(X, full_matrices = False, check_finite = False)[:2]
        U *= np.sign(U[0])
        K = (U / D).T
        del U, D
        XW = np.dot(K, X)
        XW *= np.sqrt(X.shape[-1])
        return XW

    def fit(self, X):
        XW = self.whiten(X, self.n_components)
        _, n_features = XW.shape
        p = 1
        components = []
        while p <= self.n_components:
            wp = np.random.rand(n_features)
            wp /= np.linalg.norm(wp)
            if np.max(wp @ XW) <= 0:
                wp = -wp
            converged = False
            num_iterations = 0
            while not converged and num_iterations < self.maxiter:
                if np.min(wp @ XW) >= 0:
                    for component in components:
                        wp -= (wp @ component) * component
                gz = 1 / (1 + np.exp(-wp @ XW))
                g_z = gz * XW - np.mean(gz) * wp
                g_z_prime = gz * (1 - gz)

                wp_new = np.mean(g_z, axis=1) - np.mean(g_z_prime) * wp
                wp_new /= np.linalg.norm(wp_new)
                if np.linalg.norm(wp_new - wp) < self.tol:
                    converged = True
                else:
                    wp = wp_new
                num_iterations += 1
            if converged:
                components.append(wp_new)
                p += 1
        self.components_ = np.array(components)
        self.mixing_ = np.linalg.pinv(self.components_)

    def transform(self, X):
        return X @ self.components_