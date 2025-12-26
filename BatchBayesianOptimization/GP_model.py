import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import scipy
import random
import sobol_seq
import time
from datetime import datetime


class GP:
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True):
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim = X.shape[0], X.shape[1]
        self.ny_dim = Y.shape[1]
        self.multi_hyper = multi_hyper
        self.var_out = var_out

        self.X_mean, self.X_std = np.mean(X, axis=0), np.std(X, axis=0) + 1e-8
        self.Y_mean, self.Y_std = np.mean(Y, axis=0), np.std(Y, axis=0) + 1e-8
        self.X_norm, self.Y_norm = (X - self.X_mean) / self.X_std, (Y - self.Y_mean) / self.Y_std

        self.hypopt, self.invKopt = self.determine_hyperparameters()

    # covariance matrix
    def cov_mat(self, kernel, X_norm, W, sf2):
        if kernel == 'RBF':
            dist = scipy.spatial.distance.cdist(X_norm, X_norm, 'seuclidean', V=W) ** 2
            cov_matrix = sf2 * np.exp(-0.5 * dist)
            return cov_matrix

    # single set xnorm against the dataset Xnorm
    def cal_cov_matrix(self, xnorm, Xnorm, ell, sf2):

        dist = scipy.spatial.distance.cdist(xnorm, Xnorm, 'seuclidean', V=ell) ** 2
        cov_matrix = sf2 * np.exp(-0.5 * dist)

        return cov_matrix

    def negative_loglikelihood(self, hyper, X, Y):
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel = self.kernel

        W = np.exp(2 * hyper[:nx_dim])
        sf2 = np.exp(2 * hyper[nx_dim])
        sn2 = np.exp(2 * hyper[nx_dim + 1])

        K = self.cov_mat(kernel, X, W, sf2)
        K = K + (sn2 + 1e-8) * np.eye(n_point)
        K = (K + K.T) * 0.5
        L = np.linalg.cholesky(K)
        logdetK = 2 * np.sum(np.log(np.diag(L)))
        invLY = np.linalg.solve(L, Y)
        alpha = np.linalg.solve(L.T, invLY)
        NLL = 0.5 * (np.dot(Y.T, alpha) + logdetK)

        return NLL

    def determine_hyperparameters(self):
        multi_start = 2**3
        num_hyperparameters = self.nx_dim + 2

        # lb = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-3, 1e-6])
        lb = np.array([-3, -3, -3, -3, -3, -3, -3, -3, -3, -6])

        ub = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 0])

        sampler = scipy.stats.qmc.Sobol(d=num_hyperparameters, scramble=False, seed=42)
        multi_startvec = sampler.random(n=multi_start)

        localsol = np.empty((multi_start, num_hyperparameters))
        localval = np.empty(multi_start)
        hypopt = np.zeros((10, 1))
        options = {
            'disp': False,
            'maxiter': 1000,
        }
        invKopt = []
        for i in range(self.ny_dim):
            for j in range(multi_start):
                print('multistart hyper parameter optimisation iteration= ', j, ' input= ', i)
                hyp_init = lb + (ub - lb) * multi_startvec[j, :]
                res = scipy.optimize.minimize(self.negative_loglikelihood, hyp_init,
                                              args=(self.X_norm, self.Y_norm[:, i]), method='SLSQP',
                                              options=options, bounds=scipy.optimize.Bounds(lb, ub), tol=1e-8)
                localsol[j] = res.x
                localval[j] = res.fun

            minindex = np.argmin(localval)
            hypopt[:, i] = localsol[minindex]
            ellopt = np.exp(2. * hypopt[:self.nx_dim, i])
            sf2opt = np.exp(2. * hypopt[self.nx_dim, i])
            sn2opt = np.exp(2. * hypopt[self.nx_dim + 1, i]) + 1e-8

            Kopt = self.cov_mat(self.kernel, self.X_norm, ellopt, sf2opt) + sn2opt * np.eye(self.n_point)
            Kopt = (Kopt + Kopt.T) * 0.5  # Symmetrize
            invKopt += [np.linalg.solve(Kopt, np.eye(self.n_point))]

        return hypopt, invKopt

    def GP_inference_np(self, x):
        xnorm = (x - self.X_mean) / self.X_std

        if xnorm.ndim == 1:
            xnorm = xnorm.reshape(-1, 1)

        n_test = xnorm.shape[0]
        mean = np.zeros((n_test, self.ny_dim))
        var = np.zeros((n_test, self.ny_dim))

        for i in range(self.ny_dim):
            invK = self.invKopt[i]
            hyper = self.hypopt[:, i]
            ellopt, sf2opt = np.exp(2 * hyper[:self.nx_dim]), np.exp(2 * hyper[self.nx_dim])
            sn2opt = np.exp(2 * hyper[self.nx_dim + 1])

            # k = self.cal_cov_matrix(xnorm, self.X_norm, ellopt, sf2opt)
            # mean[i] = np.matmul(np.matmul(k, invK), self.Y_norm[:,i]).item()
            # var[i] = max(0, sf2opt - (np.matmul(np.matmul(k, invK), k.T).item()))

            k_star = self.cal_cov_matrix(xnorm, self.X_norm, ellopt, sf2opt)
            k_star_star = np.full(n_test, sf2opt)  # For RBF kernel k(x,x) = sf2
            mean_norm = k_star @ invK @ self.Y_norm[:, i]
            var_latent_norm = k_star_star - np.einsum('ij,ji->i', k_star @ invK, k_star.T)

            if self.var_out:
                var_norm = var_latent_norm + sn2opt
            else:
                var_norm = var_latent_norm

            mean[:, i] = mean_norm
            var[:, i] = np.maximum(0, var_norm)

        mean_sample = mean * self.Y_std + self.Y_mean
        var_sample = var * self.Y_std ** 2

        if self.var_out:
            return mean_sample.squeeze(), var_sample.squeeze()
        else:
            return mean_sample.squeeze()
