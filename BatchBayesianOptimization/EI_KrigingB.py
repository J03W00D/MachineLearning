import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import scipy
import random
import sobol_seq
import time
from datetime import datetime

import matplotlib.pyplot as plt

group_names = ['Idrees Kholwadia', 'Joe Wood', 'Joseph Wright', 'Kundan Mahitkumar']
cid_numbers = ['000000', '02301095', '22222', '02219861']
oral_assessment = [0, 1]

category_map_1 = {
    'celltype_1': [1, 0, 0],
    'celltype_2': [0, 1, 0],
    'celltype_3': [0, 0, 1]
}

def objective_func(x):
    return np.array(virtual_lab.conduct_experiment(x))

def one_hot_to_category_name(x_next, category_map=category_map_1):
    one_hot_length = len(next(iter(category_map.values())))
    reverse_map = {tuple(value): key for key, value in category_map.items()}
    processed_rows = []
    for row in x_next:
        numerical_part = row[:-one_hot_length]
        one_hot_part = row[-one_hot_length:]
        if isinstance(one_hot_part, np.ndarray):
            one_hot_part = one_hot_part.tolist()
        one_hot_tuple = tuple(one_hot_part)
        category_name = reverse_map.get(one_hot_tuple)

        if category_name:
            if isinstance(numerical_part, np.ndarray):
                numerical_part = numerical_part.tolist()
            processed_rows.append(numerical_part + [category_name])

    return processed_rows

def category_to_one_hot(x_next, category_map=category_map_1):
    x_continuous = np.array([row[:-1] for row in x_next])
    x_categorical = [row[-1] for row in x_next]
    x_categorical_value = np.array([category_map[i] for i in x_categorical])
    x = np.hstack([x_continuous, x_categorical_value])
    return x

def stable_cholesky(K, jitter=1e-8, max_tries=5):
    """Robust Cholesky with increasing jitter."""
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10
    raise np.linalg.LinAlgError("Cholesky failed even with jitter.")


class GP:
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True, hypopt_init=None):
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim = X.shape[0], X.shape[1]
        self.ny_dim = Y.shape[1]
        self.multi_hyper = multi_hyper
        self.var_out = var_out

        self.X_mean, self.X_std = np.mean(X, axis=0), np.std(X, axis=0) + 1e-8
        self.Y_mean, self.Y_std = np.mean(Y, axis=0), np.std(Y, axis=0) + 1e-8
        self.X_norm, self.Y_norm = (X - self.X_mean) / self.X_std, (Y - self.Y_mean) / self.Y_std

        self.hypopt, self.invKopt = self.determine_hyperparameters(hypopt_init=hypopt_init)

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

        L = stable_cholesky(K)  # or np.linalg.cholesky(K)
        logdetK = 2 * np.sum(np.log(np.diag(L)))
        invLY = np.linalg.solve(L, Y)
        alpha = np.linalg.solve(L.T, invLY)
        NLL = 0.5 * (Y.T @ alpha + logdetK)

        return float(NLL)

    def determine_hyperparameters(self, hypopt_init=None):
        if hypopt_init is None:
            multi_start = 2**3  # e.g. 8 starts for the first iteration
        else:
            multi_start = 2**2

        num_hyperparameters = self.nx_dim + 2

        lb = np.array([-3] * self.nx_dim + [-3, -6])
        ub = np.array([3] * self.nx_dim + [3, 0])

        sampler = scipy.stats.qmc.Sobol(d=num_hyperparameters, scramble=False, seed=42)
        multi_startvec = sampler.random(n=multi_start)

        localsol = np.empty((multi_start + 1, num_hyperparameters))  # +1 for warm-start
        localval = np.empty(multi_start + 1)
        hypopt = np.zeros((num_hyperparameters, self.ny_dim))
        options = {
            'disp': False,
            'maxiter': 1000,
        }
        Lopt_list = []

        for i in range(self.ny_dim):
            # 1) standard Sobol starts
            for j in range(multi_start):
                hyp_init = lb + (ub - lb) * multi_startvec[j, :]
                res = scipy.optimize.minimize(
                    self.negative_loglikelihood,
                    hyp_init,
                    args=(self.X_norm, self.Y_norm[:, i]),
                    method='SLSQP',
                    options=options,
                    bounds=scipy.optimize.Bounds(lb, ub),
                    tol=1e-8,
                )
                localsol[j] = res.x
                localval[j] = res.fun

            # 2) OPTIONAL warm-start from previous hypopt (if provided)
            if hypopt_init is not None:
                # hypopt_init shape: (num_hyperparameters, ny_dim)
                hyp_init_prev = hypopt_init[:, i]

                # clip to bounds (just in case)
                hyp_init_prev = np.minimum(np.maximum(hyp_init_prev, lb), ub)

                res_prev = scipy.optimize.minimize(
                    self.negative_loglikelihood,
                    hyp_init_prev,
                    args=(self.X_norm, self.Y_norm[:, i]),
                    method='SLSQP',
                    options=options,
                    bounds=scipy.optimize.Bounds(lb, ub),
                    tol=1e-8,
                )

                localsol[multi_start] = res_prev.x
                localval[multi_start] = res_prev.fun
            else:
                # if no warm-start, just copy the best of the Sobol runs into the extra slot
                best_idx_tmp = np.argmin(localval[:multi_start])
                localsol[multi_start] = localsol[best_idx_tmp]
                localval[multi_start] = localval[best_idx_tmp]

            # 3) pick the best among all starting points (Sobol + warm-start)
            minindex = np.argmin(localval)
            hypopt[:, i] = localsol[minindex]

            ellopt = np.exp(2.0 * hypopt[:self.nx_dim, i])
            sf2opt = np.exp(2.0 * hypopt[self.nx_dim, i])
            sn2opt = np.exp(2.0 * hypopt[self.nx_dim + 1, i]) + 1e-8

            Kopt = self.cov_mat(self.kernel, self.X_norm, ellopt, sf2opt) + sn2opt * np.eye(self.n_point)
            Kopt = (Kopt + Kopt.T) * 0.5
            Lopt = stable_cholesky(Kopt)
            Lopt_list.append(Lopt)

        return hypopt, Lopt_list

    def GP_inference_np(self, x):
        xnorm = (x - self.X_mean) / self.X_std

        if xnorm.ndim == 1:
            xnorm = xnorm[None, :]  # (1, nx_dim)

        n_test = xnorm.shape[0]
        mean = np.zeros((n_test, self.ny_dim))
        var = np.zeros((n_test, self.ny_dim))

        for i in range(self.ny_dim):
            L = self.invKopt[i]  # Cholesky factor of K (n_point x n_point)
            hyper = self.hypopt[:, i]
            ellopt = np.exp(2 * hyper[:self.nx_dim])
            sf2opt = np.exp(2 * hyper[self.nx_dim])
            sn2opt = np.exp(2 * hyper[self.nx_dim + 1])

            # Compute k_* = k(x*, X)
            k_star = self.cal_cov_matrix(xnorm, self.X_norm, ellopt, sf2opt)  # (n_test, n_point)

            # GP mean: k_* K^{-1} y = k_* alpha
            # First compute alpha = K^{-1} y using L
            y_i = self.Y_norm[:, i]  # (n_point,)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_i))  # (n_point,)

            mean_norm = k_star @ alpha  # (n_test,)

            # GP variance
            # v = L^{-1} k_*^T  → shape: (n_point, n_test)
            v = np.linalg.solve(L, k_star.T)
            # latent variance: sf2 - sum(v^2, axis=0)
            k_star_star = np.full(n_test, sf2opt)
            var_latent_norm = k_star_star - np.sum(v ** 2, axis=0)

            if self.var_out:
                var_norm = var_latent_norm + sn2opt
            else:
                var_norm = var_latent_norm

            mean[:, i] = mean_norm
            var[:, i] = np.maximum(0, var_norm)

        mean_sample = mean * self.Y_std + self.Y_mean
        var_sample = var * (self.Y_std ** 2)

        if self.var_out:
            return mean_sample.squeeze(), var_sample.squeeze()
        else:
            return mean_sample.squeeze()

    @classmethod
    def from_existing_hypers(cls, X, Y, kernel, hypopt, var_out=True):
        """
        Build a GP with fixed hyperparameters hypopt.
        This does NOT re-optimise hyperparameters; it just computes posterior.
        """
        obj = cls.__new__(cls)  # create uninitialised instance

        # Basic attributes
        obj.X, obj.Y, obj.kernel = X, Y, kernel
        obj.n_point, obj.nx_dim = X.shape
        obj.ny_dim = Y.shape[1]
        obj.multi_hyper = True
        obj.var_out = var_out

        # Normalisation (same as in __init__)
        obj.X_mean = np.mean(X, axis=0)
        obj.X_std = np.std(X, axis=0) + 1e-8
        obj.Y_mean = np.mean(Y, axis=0)
        obj.Y_std = np.std(Y, axis=0) + 1e-8
        obj.X_norm = (X - obj.X_mean) / obj.X_std
        obj.Y_norm = (Y - obj.Y_mean) / obj.Y_std

        # Use provided hypopt; compute Lopt_list only (no optimisation)
        obj.hypopt = hypopt
        obj.invKopt = obj._build_cholesky_from_hypers()

        return obj

    def _build_cholesky_from_hypers(self):
        """
        Internal helper: given self.X_norm, self.Y_norm, self.hypopt,
        compute Cholesky factors Lopt_list for each output dim.
        """
        Lopt_list = []
        for i in range(self.ny_dim):
            hyper = self.hypopt[:, i]
            ellopt = np.exp(2.0 * hyper[:self.nx_dim])
            sf2opt = np.exp(2.0 * hyper[self.nx_dim])
            sn2opt = np.exp(2.0 * hyper[self.nx_dim + 1]) + 1e-8

            Kopt = self.cov_mat(self.kernel, self.X_norm, ellopt, sf2opt) + sn2opt * np.eye(self.n_point)
            Kopt = (Kopt + Kopt.T) * 0.5
            Lopt = stable_cholesky(Kopt)
            Lopt_list.append(Lopt)

        return Lopt_list


class BO:
    def __init__(self, iterations, batch_size, search_space):
        self.start_time = datetime.timestamp(datetime.now())
        self.Y = []
        self.time = []

        self.best_y_history = []
        self.current_best_y = -np.inf

        self.X_searchspace, self.X_1, self.X_2, self.X_3 = self.create_searchspace(search_space)

        self.X_initial, self.Y_initial = self.create_initial_samples()
        self.time = [datetime.timestamp(datetime.now()) - self.start_time]
        self.time += [0] * (len(self.X_initial) - 1)
        self.start_time = datetime.timestamp(datetime.now())

        self.batch_bayesian(iterations, batch_size)

    def create_initial_samples(self):
        sampler = scipy.stats.qmc.LatinHypercube(d=5, seed=42)
        continuous_samples = sampler.random(n=6)

        bounds = np.array([[30., 40.], [6., 8.], [0., 50.], [0., 50.], [0., 50.]])
        scaled_samples = scipy.stats.qmc.scale(continuous_samples, bounds[:, 0], bounds[:, 1])

        cell_type = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        X_initial = []
        for i in range(len(scaled_samples)):
            X_initial.append(list(scaled_samples[i]) + cell_type[i])

        X_initial = one_hot_to_category_name(X_initial)

        Y_initial = objective_func(X_initial)

        for row in Y_initial:
            if row > self.current_best_y:
                self.current_best_y = row
            self.best_y_history.append(self.current_best_y)
            self.Y.append(row)

        Y_initial = np.array(Y_initial)
        Y_initial = Y_initial.reshape(-1, 1)
        print(X_initial, Y_initial)
        X_initial = category_to_one_hot(X_initial)

        return X_initial, Y_initial

    def create_searchspace(self, num_points):

        bounds = np.array([
            [30., 40.],  # temp
            [6., 8.],  # pH
            [0., 50.],  # f1
            [0., 50.],  # f2
            [0., 50.]  # f3
        ])

        num_continuous_dims = 5

        sampler = scipy.stats.qmc.Sobol(d=num_continuous_dims, scramble=True)
        sobol_points_01 = sampler.random(n=num_points)

        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]

        scaled_points = scipy.stats.qmc.scale(sobol_points_01, lower_bounds, upper_bounds)

        x_initial_1 = []
        x_initial_2 = []
        x_initial_3 = []

        X_searchspace = []
        for cell_type in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            for point in scaled_points:
                X_searchspace.append(list(point) + cell_type)

        for i in [30, 40]:
            for j in [6, 8]:
                for k in [0, 50]:
                    for l in [0, 50]:
                        for m in [0, 50]:
                            for n in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
                                X_searchspace.append([i, j, k, l, m] + n)

        return X_searchspace, x_initial_1, x_initial_2, x_initial_3

    def find_indices_of_rows(self, x_data, x_searchspace):
        indices = set()
        for row in x_data:
            index = np.where(np.all(x_searchspace == row, axis=1))[0]
            if index.size > 0:
                indices.add(index[0])
        return indices

    def expected_improvement(self, mean, std, y_best, xi, epsilon=1e-8):
        std_safe = np.maximum(std, epsilon)
        # *** CORRECTED TERM FOR MAXIMIZATION ***
        diff = mean - y_best - xi

        Z = diff / std_safe
        Z = np.clip(Z, -10, 10)
        pdf_Z = scipy.stats.norm.pdf(Z)
        cdf_Z = scipy.stats.norm.cdf(Z)

        # *** CORRECTED EI FORMULA FOR MAXIMIZATION ***
        ei = diff * cdf_Z + std_safe * pdf_Z
        ei = np.maximum(ei, 0.0)

        return ei

    def plot_graphs(self, max_ei_history):
        # Create a figure with a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # 1. Top Left: Best Solution Found
        num_iterations = len(self.best_y_history)
        iterations = list(range(1, num_iterations + 1))
        axes[0, 0].plot(iterations, self.best_y_history, marker='o', linestyle='-')
        axes[0, 0].set_title('Best Maximum Solution Found vs. Iterations')
        axes[0, 0].set_xlabel('No. of Iterations')
        axes[0, 0].set_ylabel('Best Current Maximum Solution')
        axes[0, 0].grid(True)

        # 2. Top Right: Maximum Expected Improvement
        # Note: Using len(max_ei_history) is safer than hardcoding 15
        ei_iterations = list(range(1, len(max_ei_history) + 1))
        axes[0, 1].plot(ei_iterations, max_ei_history, marker='o', linestyle='-', color='red')
        axes[0, 1].set_title('Maximum Expected Improvement vs. Iterations')
        axes[0, 1].set_xlabel('No. of Iterations')
        axes[0, 1].set_ylabel('Maximum EI')
        axes[0, 1].grid(True)

        # 3. Bottom Left: Time Taken
        iteration_times = [t for t in self.time if t > 0]
        axes[1, 0].plot(range(len(iteration_times)), iteration_times, marker='x', color='purple')
        axes[1, 0].set_title('Time Taken per Optimization Step')
        axes[1, 0].set_xlabel('Step (0=Initial, 1+=Batch Iterations)')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)

        # 4. Bottom Right: Score vs Iterations
        # Note: Using len(self.Y) is safer than hardcoding 81
        y_iterations = list(range(1, len(self.Y) + 1))
        axes[1, 1].scatter(y_iterations, self.Y, marker='o', linestyle='-', color='green')
        axes[1, 1].set_title('Score vs. Iterations')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Score for each iteration')
        axes[1, 1].grid(True)

        # Adjust layout to prevent title/label overlap
        plt.tight_layout()
        plt.show()

    def _compute_ei_on_searchspace(self, gp_model, y_best, iteration_index):
        """
        Compute EI for all points in X_searchspace using the given GP model.
        Uses your xi-schedule based on iteration_index.
        """
        Xs = np.array(self.X_searchspace)
        means, vars_ = gp_model.GP_inference_np(Xs)
        stds = np.sqrt(vars_)

        if iteration_index in [0, 1]:
            xi = 0.5
        elif iteration_index in [2, 3, 4, 5, 6, 7, 8]:
            xi = 0.1
        else:
            xi = 0.01

        ei = self.expected_improvement(means, stds, y_best, xi)
        return ei, means, stds

    def batch_bayesian(self, no_iterations, batch_size):

        X_data = self.X_initial.copy()
        Y_data = self.Y_initial.copy()

        X_search = np.array(self.X_searchspace)
        sampled_indices = self.find_indices_of_rows(self.X_initial, X_search)

        max_ei_history = []
        prev_hypopt = None  # store hyperparameters from previous outer iteration

        for it in range(no_iterations):
            print(f"\n--- Iteration {it + 1}/{no_iterations} ---")

            # 1) Fit GP on real data, with warm-start
            gp_model_0 = GP(X_data, Y_data, 'RBF', 5, True, hypopt_init=prev_hypopt)
            prev_hypopt = gp_model_0.hypopt.copy()
            hypopt_fixed = gp_model_0.hypopt.copy()

            print("Hyperparameters:\n", gp_model_0.hypopt)

            # 2) EI on full search space under current real-data GP
            y_best_real = np.max(Y_data)
            ei_values, all_means, all_stds = self._compute_ei_on_searchspace(
                gp_model_0, y_best_real, it
            )

            # mask already evaluated points
            for idx in sampled_indices:
                if idx < len(ei_values):
                    ei_values[idx] = -1e9

            # log max EI from the *initial* EI (before KB)
            max_ei_history.append(np.max(ei_values))

            max_idx = np.argmax(ei_values)
            print(f"Initial Max EI Point - Mean: {all_means[max_idx]}, Std: {all_stds[max_idx]}")

            # 3) Kriging Believer batch selection
            print(f"  Selecting {batch_size} points for batch using Kriging Believer...")
            x_batch_to_evaluate = []

            # Local KB copies
            X_kb = X_data.copy()
            Y_kb = Y_data.copy()

            for j in range(batch_size):
                if j == 0:
                    # Use the GP already fitted on real data
                    gp_kb = gp_model_0
                else:
                    # Build GP on fantasy-augmented data with fixed hyperparameters
                    gp_kb = GP.from_existing_hypers(X_kb, Y_kb, 'RBF', hypopt_fixed, var_out=True)

                # EI with respect to fantasy-augmented best y
                y_best_kb = np.max(Y_kb)
                ei_kb, means_kb, stds_kb = self._compute_ei_on_searchspace(gp_kb, y_best_kb, it)

                # mask already truly sampled points
                for idx in sampled_indices:
                    if idx < len(ei_kb):
                        ei_kb[idx] = -1e9

                # mask points already chosen in this batch
                for chosen in x_batch_to_evaluate:
                    mask_idx = np.where(np.all(X_search == chosen, axis=1))[0]
                    if mask_idx.size > 0:
                        ei_kb[mask_idx[0]] = -1e9

                # pick next batch point
                next_index = np.argmax(ei_kb)
                x_next = X_search[next_index]
                x_batch_to_evaluate.append(x_next)
                print(f"    Point {j + 1}: index {next_index}, EI {ei_kb[next_index]:.4f}")

                # KB fantasy observation = GP mean at this point
                mean_next, _ = gp_kb.GP_inference_np(x_next)
                fantasy_y = np.array(mean_next).reshape(1, 1)

                X_kb = np.vstack([X_kb, x_next.reshape(1, -1)])
                Y_kb = np.vstack([Y_kb, fantasy_y])

            # mark batch points as truly sampled indices for future iterations
            for chosen in x_batch_to_evaluate:
                idx = np.where(np.all(X_search == chosen, axis=1))[0]
                if idx.size > 0:
                    sampled_indices.add(idx[0])

            # 4) Evaluate true objective on the selected batch
            print(f"  Evaluating {batch_size} points...")
            x_batch_array = np.vstack(x_batch_to_evaluate)
            y_batch_real_results = []

            for k in range(batch_size):
                x_point = x_batch_array[k].reshape(1, -1)
                x_readable_list = one_hot_to_category_name([x_point.squeeze()])
                x_readable = x_readable_list[0]
                y_next_array = objective_func([x_readable])
                y_next_raw = float(np.max(y_next_array))

                if y_next_raw > self.current_best_y:
                    self.current_best_y = y_next_raw
                self.best_y_history.append(self.current_best_y)
                self.Y.append(y_next_raw)

                print(f"Sampling: {x_readable} --> {y_next_raw:.4f}")
                y_batch_real_results.append([y_next_raw])

            # track time
            self.time += [datetime.timestamp(datetime.now()) - self.start_time]
            self.time += [0] * (len(y_batch_real_results) - 1)
            self.start_time = datetime.timestamp(datetime.now())

            y_batch_array = np.array(y_batch_real_results)

            # 5) Update real dataset with batch
            X_data = np.vstack([X_data, x_batch_array])
            Y_data = np.vstack([Y_data, y_batch_array])

        # 6) Final reporting
        print("\nOptimization finished.")
        best_index = np.argmax(Y_data)
        best_y = Y_data[best_index]
        best_x_numeric = X_data[best_index]
        best_x_readable_list = one_hot_to_category_name([best_x_numeric])
        best_x_readable = best_x_readable_list[0]

        print(f"Best titre found: {best_y.item():.4f} g/L")
        print(f"Best parameters: {best_x_readable}")

        self.plot_graphs(max_ei_history)


bayesian = BO(15, 5, 2**15)

# best_y_values = []
# for i in range(10):
#     bayesian = BO(15, 5, 2**15)
#     best_y_iter = bayesian.best_y_history[-1]
#     best_y_values.append(best_y_iter)

# average = np.average(best_y_values)

# print(best_y_values)
# print(average)

# Best titre found: 576.4849 g/L
# Best parameters: [30.358146196231246, 6.447080001235008, 48.79570505581796, 46.79364333860576, 14.37097997404635, 'celltype_2']
