import warnings
import numpy as np
from scipy import linalg
from sklearn.utils import as_float_array, check_array, check_random_state
from sklearn.utils._param_validation import Interval, Options, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.decomposition._fastica import FastICA, _ica_def, _ica_par

class ModICA(FastICA):
    def __init__(self, n_components = None, algorithm = 'parallel', whiten = True,
                 fun = 'logcosh', fun_args = None, max_iter = 200, tol = 1e-4,
                 w_init = None, random_state = None, modified = True, predefined = None):
        super().__init__(n_components = n_components, algorithm = algorithm, whiten = whiten,
                         fun = fun, fun_args = fun_args, max_iter = max_iter, tol = tol,
                         w_init = w_init, random_state = random_state)
        self.modified = modified
        self.predefined = predefined

    def _fit_transform(self, X, compute_sources=False):
        XT = self._validate_data(
            X, copy=self.whiten, dtype=[np.float64, np.float32], ensure_min_samples=2
        ).T
        fun_args = {} if self.fun_args is None else self.fun_args
        random_state = check_random_state(self.random_state)

        alpha = fun_args.get("alpha", 1.0)
        if not 1 <= alpha <= 2:
            raise ValueError("alpha must be in [1,2]")

        if self.fun == "logcosh":
            g = _logcosh
        elif self.fun == "exp":
            g = _exp
        elif self.fun == "cube":
            g = _cube
        elif callable(self.fun):

            def g(x, fun_args):
                return self.fun(x, **fun_args)

        n_features, n_samples = XT.shape
        n_components = self.n_components
        if not self.whiten and n_components is not None:
            n_components = None
            warnings.warn("Ignoring n_components with whiten=False.")

        if n_components is None:
            n_components = min(n_samples, n_features)
        if n_components > min(n_samples, n_features):
            n_components = min(n_samples, n_features)
            warnings.warn(
                "n_components is too large: it will be set to %s" % n_components
            )

        if self.whiten:
            X_mean = XT.mean(axis=-1)
            XT -= X_mean[:, np.newaxis]

            if self.whiten_solver == "eigh":
                d, u = linalg.eigh(XT.dot(X))
                sort_indices = np.argsort(d)[::-1]
                eps = np.finfo(d.dtype).eps
                degenerate_idx = d < eps
                if np.any(degenerate_idx):
                    warnings.warn(
                        "There are some small singular values, using "
                        "whiten_solver = 'svd' might lead to more "
                        "accurate results."
                    )
                d[degenerate_idx] = eps
                np.sqrt(d, out=d)
                d, u = d[sort_indices], u[:, sort_indices]
            elif self.whiten_solver == "svd":
                u, d = linalg.svd(XT, full_matrices=False, check_finite=False)[:2]

            u *= np.sign(u[0])

            K = (u / d).T[:n_components]
            del u, d
            X1 = np.dot(K, XT)
            X1 *= np.sqrt(n_samples)
        else:
            X1 = as_float_array(XT, copy=False)

        w_init = self.w_init
        if w_init is None:
            w_init = np.asarray(
                random_state.normal(size=(n_components, n_components)), dtype=X1.dtype
            )

        else:
            w_init = np.asarray(w_init)
            if w_init.shape != (n_components, n_components):
                raise ValueError(
                    "w_init has invalid shape -- should be %(shape)s"
                    % {"shape": (n_components, n_components)}
                )

        kwargs = {
            "tol": self.tol,
            "g": g,
            "fun_args": fun_args,
            "max_iter": self.max_iter,
            "w_init": w_init,
        }

        if self.modified:
            if self.algorithm == "parallel":
                W, n_iter = _ica_mod_par(X1, **kwargs)
            elif self.algorithm == "deflation":
                W, n_iter = _ica_mod_def(X1, **kwargs)
        else:
            if self.algorithm == "parallel":
                W, n_iter = _ica_par(X1, **kwargs)
            elif self.algorithm == "deflation":
                W, n_iter = _ica_def(X1, **kwargs)
        del X1

        self.n_iter_ = n_iter

        if compute_sources:
            if self.whiten:
                S = np.linalg.multi_dot([W, K, XT]).T
            else:
                S = np.dot(W, XT).T
        else:
            S = None

        if self.whiten:
            if self.whiten == "unit-variance":
                if not compute_sources:
                    S = np.linalg.multi_dot([W, K, XT]).T
                S_std = np.std(S, axis=0, keepdims=True)
                S /= S_std
                W /= S_std.T

            self.components_ = np.dot(W, K)
            self.mean_ = X_mean
            self.whitening_ = K
        else:
            self.components_ = W
        
        if self.predefined is not None:
            self.components_ = self.predefined

        self.mixing_ = linalg.pinv(self.components_, check_finite=False)
        self._unmixing = W

        return S

def _ica_mod_par(X, tol, g, fun_args, max_iter, w_init):
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn(
            (
                "FastICA did not converge. Consider increasing "
                "tolerance or the maximum number of iterations."
            ),
            ConvergenceWarning,
        )

    return W, ii + 1

def _ica_mod_def(X, tol, g, fun_args, max_iter, w_init):
    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w**2).sum())
        for i in range(max_iter):
            gwtx, g_wtx = g(np.dot(w.T, X), fun_args)
            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w
            _gs_decorrelation(w1, W, j)
            w1 /= np.sqrt((w1**2).sum())
            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break
        n_iter.append(i + 1)
        W[j, :] = w
    return W, max(n_iter)

def _sym_decorrelation(W):
    s, u = linalg.eigh(np.dot(W, W.T))
    s = np.clip(s, a_min=np.finfo(W.dtype).tiny, a_max=None)
    return np.linalg.multi_dot([u * (1.0 / np.sqrt(s)), u.T, W])

def _gs_decorrelation(w, W, j):
    w -= np.linalg.multi_dot([w, W[:j].T, W[:j]])
    return w

def _logcosh(x, fun_args=None):
    alpha = fun_args.get("alpha", 1.0)

    x *= alpha
    gx = np.tanh(x, x)
    g_x = np.empty(x.shape[0], dtype=x.dtype)
    for i, gx_i in enumerate(gx):
        g_x[i] = (alpha * (1 - gx_i**2)).mean()
    return gx, g_x

def _exp(x, fun_args):
    exp = np.exp(-(x**2) / 2)
    gx = x * exp
    g_x = (1 - x**2) * exp
    return gx, g_x.mean(axis=-1)

def _cube(x, fun_args):
    return x**3, (3 * x**2).mean(axis=-1)