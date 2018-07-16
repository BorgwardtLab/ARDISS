# -----------------------------------------------------------------------------
# This is the model used for the imputation of missing summary statistics
# The ARD weights can be either associated with a specific population or
# optimized via GPflow (see ard_model.py)
#
# January 2018, M. Togninalli
# -----------------------------------------------------------------------------
import numpy as np

class GPModelARD(object):
    # Class for storing the gp model parameters (noise, ard, etc)
    def __init__(self, sigma_noise=0.1, sigma_ard=np.array([])):
        self.sigma_noise = sigma_noise
        self.sigma_ard = sigma_ard #/ np.sum(sigma_ard)
        self.K = None
        self.K_inv = None
        self.Y = None
        self.X = None
        self.alpha = None

    def set_X(self, X):
        assert X.ndim == 2
        self.X = X.copy()
        self.compute_kernel()
        self.compute_K_inv()

    def set_Y(self, Y):
        self.Y = Y.copy()
        if self.K_inv is not None:
            self.alpha = np.dot(self.K_inv, self.Y)

    def compute_kernel(self):
        assert self.X is not None
        ard_X = np.multiply(self.sigma_ard, self.X)
        k_y = np.dot(ard_X, self.X.T)
        self.K = k_y + self.sigma_noise * np.eye(self.X.shape[0])

    def compute_K_inv(self):
        assert self.K is not None
        self.K_inv = np.linalg.inv(self.K)

    def compute_k_star(self, X_star):
        assert self.X is not None
        ard_X = np.multiply(self.sigma_ard, X_star)
        k_star = np.dot(ard_X, self.X.T)
        return k_star

    def impute_Y(self, X_star, compute_var=False):
        k_star = self.compute_k_star(X_star)
        Y_imp = np.dot(k_star, self.alpha)
        if compute_var:
            variances = (np.dot(k_star, self.K_inv) * k_star).sum(-1)
            k_star_star = (np.multiply(self.sigma_ard, X_star) * X_star).sum(-1)
            if k_star_star is not None:
                variances += 1 - k_star_star
            mask = np.where(variances < 0)
            variances[mask] = 0.0
            return Y_imp, variances
        return Y_imp

    # fast update of moving window in FLASH
    def update_model(self, new_x, new_y):
        # First update the X matrix
        self.X = np.vstack([self.X[1:,:], new_x.reshape(1,-1)])
        self.Y = np.append(self.Y[1:],new_y).reshape(-1,1)
        # then update the kernel
        k, k12, k21, K22 = matrix_decompose_up(self.K)
        new_kernel_entry = np.dot(np.multiply(self.sigma_ard, self.X), new_x)
        new_kernel_entry[-1] += self.sigma_noise
        self.K = matrix_recompose(K22, new_kernel_entry[:-1], new_kernel_entry[:-1].T, new_kernel_entry[-1])
        # Finally update the inverse
        self.K_inv = update_sigma_tt_inv(self.K,self.K_inv)
        self.alpha = np.dot(self.K_inv, self.Y)


def update_sigma_tt_inv(sigma_tt_new_window, sigma_tt_inv):
    # get A22inv
    A11, ac, ar, a = matrix_decompose_down(sigma_tt_new_window)
    X11,X12,X21,X22 = matrix_decompose_up(sigma_tt_inv)
    A22inv = np.array(X22 - (1./X11)*np.dot(X21,X12))

    # construct new sigma_tt_inv
    ara22 = np.dot(ar,A22inv)
    s = 1./(a-np.dot(ara22,ac))

    sr = -s*ara22
    sc = -s*np.dot(A22inv,ac)

    S11 = A22inv - np.dot(sc,ara22)
    sigma_tt_inv_new = matrix_recompose(S11, sc, sr, s)

    return sigma_tt_inv_new

def matrix_decompose_up(X):
    # Decompose a matrix by removing its first row and first column
    x = X[0,0]
    X22 = X[1:,1:]
    xc = X[1:,0].reshape(len(X)-1,1)
    xr = np.array(X[0][1:]).reshape(1,len(X)-1)
    return x, xr, xc, X22

def matrix_decompose_down(X):
    # Decompose a matrix by removing its last row and first column
    x = X[-1,-1]
    X11 = X[0:-1,0:-1]
    xc = X[:-1,-1].reshape(len(X)-1,1)
    xr = X[-1,:-1].reshape(1,len(X)-1)
    return X11, xc, xr, x

def matrix_recompose(X11, xc, xr, x):
    # Recompose a matrix by adding last row and column
    X = np.zeros((len(X11) + 1, len(X11) + 1))
    X[:-1,:-1] = X11
    X[:-1,-1] = xc.reshape(len(xc))
    X[-1,:-1] = xr
    X[-1,-1] = x
    return X