import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import scipy
import random
import sobol_seq
import time
from datetime import datetime

import matplotlib.pyplot as plt

group_names = ['Idrees Kholwadia', 'Joe Wood', 'Joseph Wright', 'Kundan Mahitkumar']
cid_numbers = ['02286584', '02301095', '02214563', '02219861']
oral_assessment = [0, 1]

category_map_1 = {
    'celltype_1': [1., 0., 0.],
    'celltype_2': [0., 1., 0.],
    'celltype_3': [0., 0., 1.]
}

def objective_func(x):
    return np.array(virtual_lab.conduct_experiment(x))

def one_hot_to_category(x_next, category_map=category_map_1):
    x_next = np.atleast_2d(x_next)
    names = list(category_map.keys())
    one_hot_len = len(category_map[names[0]])

    return [
        list(row[:-one_hot_len]) + [names[np.argmax(row[-one_hot_len:])]]
        for row in x_next
    ]

def category_to_one_hot(x_next, category_map=category_map_1):
    x_continuous = np.array([row[:-1] for row in x_next])
    x_categorical = [row[-1] for row in x_next]
    x_categorical_value = np.array([category_map[i] for i in x_categorical])
    x = np.hstack([x_continuous, x_categorical_value])
    return x

def stable_cholesky(K, jitter=1e-8, max_tries=5):
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10
    raise np.linalg.LinAlgError("Cholesky failed even with jitter.")


class GP:
    def __init__(self, X, Y, var_out=True, hypopt_init=None):
        self.X, self.Y = X, Y
        self.n_point, self.nx_dim = X.shape[0], X.shape[1]
        self.ny_dim = Y.shape[1]
        self.var_out = var_out

        self.X_mean, self.X_std = np.mean(X, axis=0), np.std(X, axis=0) + 1e-8
        self.Y_mean, self.Y_std = np.mean(Y, axis=0), np.std(Y, axis=0) + 1e-8
        self.X_norm, self.Y_norm = (X - self.X_mean) / self.X_std, (Y - self.Y_mean) / self.Y_std

        self.hypopt, self.invKopt = self.determine_hyperparameters(hypopt_init=hypopt_init)

    def cal_cov_matrix(self, xnorm, Xnorm, ell, sf2):
        dist = scipy.spatial.distance.cdist(xnorm, Xnorm, 'seuclidean', V=ell) ** 2
        cov_matrix = sf2 * np.exp(-0.5 * dist)

        return cov_matrix

    def negative_loglikelihood(self, hyper, X, Y):
        n_point, nx_dim = self.n_point, self.nx_dim

        log_ell = hyper[:nx_dim]
        ell2 = np.exp(2 * log_ell)
        sf2 = np.exp(2 * hyper[nx_dim])
        sn2 = np.exp(2 * hyper[nx_dim + 1])

        K = self.cal_cov_matrix(X, X, ell2, sf2)
        K = K + (sn2 + 1e-8) * np.eye(n_point)
        K = (K + K.T) * 0.5

        L = stable_cholesky(K)  # or np.linalg.cholesky(K)
        logdetK = 2 * np.sum(np.log(np.diag(L)))
        invLY = np.linalg.solve(L, Y)
        alpha = np.linalg.solve(L.T, invLY)
        NLL = 0.5 * (Y.T @ alpha + logdetK)

        return float(NLL)

    def determine_hyperparameters(self, hypopt_init=None, iteration_index=0):
        if hypopt_init is None or iteration_index < 3:
            multi_start = 2 ** 3
        elif iteration_index < 7:
            multi_start = 2 ** 2
        else:
            multi_start = 1

        num_hyperparameters = self.nx_dim + 2

        lb = np.array([-2.5] * self.nx_dim + [-2.5, -6])
        ub = np.array([2.5] * self.nx_dim + [2.5, 0])

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

            log_ell = hypopt[:self.nx_dim, i]
            ell2 = np.exp(2.0 * log_ell)
            sf2opt = np.exp(2.0 * hypopt[self.nx_dim, i])
            sn2opt = np.exp(2.0 * hypopt[self.nx_dim + 1, i]) + 1e-8

            Kopt = self.cal_cov_matrix(self.X_norm, self.X_norm, ell2, sf2opt) + sn2opt * np.eye(self.n_point)
            Kopt = (Kopt + Kopt.T) * 0.5
            Lopt = stable_cholesky(Kopt)
            Lopt_list.append(Lopt)

        return hypopt, Lopt_list

    def GP_inference_np(self, x):
        xnorm = (x - self.X_mean) / self.X_std

        if xnorm.ndim == 1:
            xnorm = xnorm[None, :]

        n_test = xnorm.shape[0]
        mean = np.zeros((n_test, self.ny_dim))
        var = np.zeros((n_test, self.ny_dim))

        for i in range(self.ny_dim):
            L = self.invKopt[i]
            hyper = self.hypopt[:, i]
            log_ell = hyper[:self.nx_dim]
            ell2 = np.exp(2 * log_ell)
            sf2opt = np.exp(2 * hyper[self.nx_dim])
            sn2opt = np.exp(2 * hyper[self.nx_dim + 1])

            # 1. Compute Cross-Covariance for ALL test points at once
            # Result k_star shape: (n_test, n_point)
            k_star = self.cal_cov_matrix(xnorm, self.X_norm, ell2, sf2opt)

            # 2. Vectorized Mean Calculation
            y_i = self.Y_norm[:, i]
            # alpha is (n_point, 1), k_star is (n_test, n_point)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_i))
            mean[:, i] = k_star @ alpha

            # 3. Vectorized Variance Calculation
            # v = L^-1 * k_star.T -> shape: (n_point, n_test)
            v = np.linalg.solve(L, k_star.T)

            # Predictive variance: k(x*,x*) - k_* K^-1 k_*^T
            # np.sum(v**2, axis=0) gives a (n_test,) vector of the quadratic term
            var_latent_norm = sf2opt - np.sum(v ** 2, axis=0)

            if self.var_out:
                var_norm = var_latent_norm + sn2opt
            else:
                var_norm = var_latent_norm

            var[:, i] = np.maximum(1e-10, var_norm)

        mean_sample = mean * self.Y_std + self.Y_mean
        var_sample = var * (self.Y_std ** 2)

        return mean_sample.squeeze(), var_sample.squeeze()

    @classmethod
    def from_existing_hypers(cls, X, Y, hypopt, var_out=True):
        """
        Build a GP with fixed hyperparameters hypopt.
        This does NOT re-optimise hyperparameters; it just computes posterior.
        """
        obj = cls.__new__(cls)  # create uninitialised instance

        # Basic attributes
        obj.X, obj.Y = X, Y
        obj.n_point, obj.nx_dim = X.shape
        obj.ny_dim = Y.shape[1]
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

            Kopt = self.cal_cov_matrix(self.X_norm, self.X_norm, ellopt, sf2opt) + sn2opt * np.eye(self.n_point)
            Kopt = (Kopt + Kopt.T) * 0.5
            Lopt = stable_cholesky(Kopt)
            Lopt_list.append(Lopt)

        return Lopt_list


class BO:
    def __init__(self, iterations, batch_size, initial_inputs, initial_outputs, search_space):
        self.start_time = datetime.timestamp(datetime.now())
        self.Y = []
        self.time = []

        self.best_y_history = []
        self.current_best_y = -np.inf
        self.cumulative_time = 0
        self.cumulative_y = []
        self.best_y_batch = []

        self.X_searchspace = np.array(search_space)

        self.X_initial, self.Y_initial = initial_inputs, initial_outputs

        for row in self.Y_initial:
            val = float(np.max(row))
            if val > self.current_best_y:
                self.current_best_y = val
            self.best_y_history.append(self.current_best_y)
            self.Y.append(val)

        self.time = [datetime.timestamp(datetime.now()) - self.start_time]
        self.time += [0] * (len(self.X_initial) - 1)
        self.start_time = datetime.timestamp(datetime.now())

        self.batch_bayesian(iterations, batch_size)

    # def find_indices_of_rows(self, x_data, x_searchspace):
    #     indices = set()
    #     for row in x_data:
    #         index = np.where(np.all(x_searchspace == row, axis=1))[0]
    #         if index.size > 0:
    #             indices.add(index[0])
    #     return indices

    def find_indices_of_rows(self, x_data, x_searchspace):
        lookup = {tuple(row): i for i, row in enumerate(x_searchspace)}
        indices = {lookup[tuple(row)] for row in x_data if tuple(row) in lookup}
        return indices

    def expected_improvement(self, mean, std, y_best, xi, epsilon=1e-8):
        std_safe = np.maximum(std, epsilon)
        diff = mean - y_best - xi

        Z = diff / std_safe
        Z = np.clip(Z, -10, 10)
        pdf_Z = scipy.stats.norm.pdf(Z)
        cdf_Z = scipy.stats.norm.cdf(Z)
        ei = diff * cdf_Z + std_safe * pdf_Z

        return np.maximum(ei, 0.0)

    def _compute_ei_on_searchspace(self, gp_model, y_best, iteration_index):
        Xs = self.X_searchspace
        means, vars_ = gp_model.GP_inference_np(Xs)
        stds = np.sqrt(vars_)

        if iteration_index < 2:
            xi = 0.5
        elif iteration_index < 9:
            xi = 0.1   # 0.1
        else:
            xi = 0.01      # 0.01

        ei = self.expected_improvement(means, stds, y_best, xi)

        return ei, means, stds

    def batch_bayesian(self, no_iterations, batch_size):

        X_data = self.X_initial.copy()
        Y_data = self.Y_initial.copy()
        X_search = self.X_searchspace
        sampled_indices = self.find_indices_of_rows(self.X_initial, X_search)
        lookup = {tuple(row): i for i, row in enumerate(X_search)}

        max_ei_history = []
        prev_hypopt = None  # store hyperparameters from previous outer iteration

        for it in range(no_iterations):
            print(f"\n--- Iteration {it + 1}/{no_iterations} ---")

            # 1) Fit GP on real data, with warm-start
            gp_model_0 = GP(X_data, Y_data, True, hypopt_init=prev_hypopt)
            prev_hypopt = gp_model_0.hypopt.copy()
            hypopt_fixed = gp_model_0.hypopt.copy()

            print("Hyperparameters:\n", gp_model_0.hypopt)

            # 2) EI on full search space under current real-data GP
            y_best_real = np.max(Y_data)
            ei_values, all_means, all_stds = self._compute_ei_on_searchspace(
                gp_model_0, y_best_real, it
            )

            # mask already evaluated points
            ei_values[list(sampled_indices)] = -1e9

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
                    gp_kb = GP.from_existing_hypers(X_kb, Y_kb, hypopt_fixed, var_out=True)

                # EI with respect to fantasy-augmented best y
                y_best_kb = np.max(Y_kb)
                ei_kb, means_kb, stds_kb = self._compute_ei_on_searchspace(gp_kb, y_best_kb, it)

                # mask already truly sampled points
                for idx in sampled_indices:
                    if idx < len(ei_kb):
                        ei_kb[idx] = -1e9

                # # mask points already chosen in this batch
                # for chosen in x_batch_to_evaluate:
                #     mask_idx = np.where(np.all(X_search == chosen, axis=1))[0]
                #     if mask_idx.size > 0:
                #         ei_kb[mask_idx[0]] = -1e9
                # 1. Pre-calculate the lookup once (do this outside your batch loop if possible)

                # 2. Optimized masking
                for chosen in x_batch_to_evaluate:
                    idx = lookup.get(tuple(chosen))
                    if idx is not None:
                        ei_kb[idx] = -1e9

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
            # for chosen in x_batch_to_evaluate:
            #     idx = np.where(np.all(X_search == chosen, axis=1))[0]
            #     if idx.size > 0:
            #         sampled_indices.add(idx[0])

            for chosen in x_batch_to_evaluate:
                idx = lookup.get(tuple(chosen))
                if idx is not None:
                    sampled_indices.add(idx)

            # 4) Evaluate true objective on the selected batch
            print(f"  Evaluating {batch_size} points...")
            x_batch_array = np.vstack(x_batch_to_evaluate)
            y_batch_real_results = []

            for k in range(batch_size):
                x_point = x_batch_array[k].reshape(1, -1)
                x_readable_list = one_hot_to_category([x_point.squeeze()])
                x_readable = x_readable_list[0]
                y_next_array = objective_func([x_readable])
                y_next_raw = float(np.max(y_next_array))

                if y_next_raw > self.current_best_y:
                    self.current_best_y = y_next_raw
                self.best_y_history.append(self.current_best_y)
                self.Y.append(y_next_raw)

                print(f"Sampling: {x_readable} --> {y_next_raw:.4f}")
                y_batch_real_results.append([y_next_raw])

            if it == 0 or it == 1:
                pass
            else:
                self.best_y_batch.append(float(np.max(self.Y)))

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
        best_x_readable_list = one_hot_to_category([best_x_numeric])
        best_x_readable = best_x_readable_list[0]
        self.cumulative_time = np.cumsum(self.time)

        print(f"Best titre found: {best_y.item():.5f} g/L")
        print(f"Best parameters: {best_x_readable}")
        print(f"Cumulative Score: {np.cumsum(self.best_y_batch)[-1]}")
        print(f"Total time: {self.cumulative_time[-1]:.5f} seconds")

        self.plot_graphs()

    def plot_graphs(self):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        num_iterations = len(self.best_y_history)
        iterations = list(range(1, num_iterations + 1))
        axes[0, 0].plot(iterations, self.best_y_history, marker='o', linestyle='-')
        axes[0, 0].set_title('Best Maximum Solution Found vs. Iterations')
        axes[0, 0].set_xlabel('No. of Iterations')
        axes[0, 0].set_ylabel('Best Current Maximum Solution')
        axes[0, 0].grid(True)

        iteration_times = [t for t in self.time if t > 0]
        axes[1, 0].plot(range(len(iteration_times)), iteration_times, marker='x', color='purple')
        axes[1, 0].set_title('Time Taken per Optimization Step')
        axes[1, 0].set_xlabel('Step (0=Initial, 1+=Batch Iterations)')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)

        y_iterations = list(range(1, len(self.Y) + 1))
        axes[1, 1].scatter(y_iterations, self.Y, marker='o', linestyle='-', color='green', s=5)
        axes[1, 1].set_title('Score vs. Iterations')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Score for each iteration')
        axes[1, 1].grid(True)

        it_bit = np.linspace(2, 15, 13)
        axes[0, 1].plot(it_bit, self.best_y_batch, marker='o', linestyle='-', markersize=3)
        axes[0, 1].set_title(f'Cumulative Batch Y: {np.cumsum(self.best_y_batch)[-1]}')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Best Titre Yet')
        axes[0, 1].grid(True)
        plt.tight_layout()
        plt.show()


def create_initial_samples():
    sampler = scipy.stats.qmc.LatinHypercube(d=5, seed=42)
    continuous_samples = sampler.random(n=6)

    bounds = np.array([[30., 40.], [6., 8.], [0., 50.], [0., 50.], [0., 50.]])
    scaled_samples = scipy.stats.qmc.scale(continuous_samples, bounds[:, 0], bounds[:, 1])

    cell_type = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]

    X_initial = []
    for i in range(len(scaled_samples)):
        X_initial.append(list(scaled_samples[i]) + cell_type[i])

    X_initial = one_hot_to_category(X_initial)

    Y_initial = objective_func(X_initial)

    Y_initial = np.array(Y_initial)
    Y_initial = Y_initial.reshape(-1, 1)
    print(X_initial, Y_initial)
    X_initial = category_to_one_hot(X_initial)

    return X_initial, Y_initial

def create_searchspace(num_points):

    bounds = np.array([
        [30., 40.],  # temp
        [6., 8.],  # pH
        [0., 50.],  # f1
        [0., 50.],  # f2
        [0., 50.]  # f3
    ])

    sampler = scipy.stats.qmc.Sobol(d=5, scramble=True)
    sobol_points_01 = sampler.random(n=num_points)

    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    scaled_points = scipy.stats.qmc.scale(sobol_points_01, lower_bounds, upper_bounds)

    types = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    X_sobol = np.hstack([np.repeat(scaled_points, 3, axis=0), np.tile(types, (num_points, 1))])
    corners = np.array(np.meshgrid(*bounds)).T.reshape(-1, 5)
    X_corners = np.hstack([np.repeat(corners, 3, axis=0), np.tile(types, (len(corners), 1))])

    X_searchspace = np.vstack([X_sobol, X_corners])

    return X_searchspace

x_initial, y_initial = create_initial_samples()
x_searchspace = create_searchspace(2**14)
BO_m = BO(15, 5, x_initial, y_initial, x_searchspace)

# best_y_values = []
# best_cumulative_y = []
# for i in range(15):
#     x_initial, y_initial = create_initial_samples()
#     x_searchspace = create_searchspace(2 ** 14)
#     bayesian = BO(15, 5, x_initial, y_initial, x_searchspace)
#     best_y_iter = bayesian.best_y_history[-1]
#     best_y_values.append(best_y_iter)
#     best_cumulative_y.append(np.cumsum(bayesian.best_y_batch)[-1])
#
# average = np.average(best_y_values)
# average_cumsum = np.average(best_cumulative_y)
#
# print(best_y_values)
# print(average)
# print(average_cumsum)
# print(best_cumulative_y)
