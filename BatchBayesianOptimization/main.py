import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import scipy
import random
import sobol_seq
import time
from datetime import datetime

import matplotlib.pyplot as plt
from GP_model import GP

group_names     = ['Idrees Kholwadia', 'Joe Wood', 'Joseph Wright', 'Kundan Mahitkumar']
cid_numbers     = ['000000','02301095', '22222', '02219861']
oral_assessment = [0, 1]

category_map_1 = {
    'celltype_1': [1, 0, 0],
    'celltype_2': [0, 1, 0],
    'celltype_3': [0, 0, 1]
    }

xsample = create_sobol_initial_samples(3,True)
print(xsample)

def objective_func(x):
    return np.array(virtual_lab.conduct_experiment(x))

def create_sobol_initial_samples(num_points, is_initial):
    bounds = np.array([
        [30., 40.],  # temp
        [6., 8.],  # pH
        [0., 50.],  # f1
        [0., 50.],  # f2
        [0., 50.]  # f3
    ])

    cell_types = ['celltype_1', 'celltype_2', 'celltype_3', 'celltype_1', 'celltype_2', 'celltype_3']
    random.shuffle(cell_types)
    cell = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    num_continuous_dims = 5

    sampler = scipy.stats.qmc.Sobol(d=num_continuous_dims, scramble=True)
    sobol_points_01 = sampler.random(n=num_points)

    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    scaled_points = scipy.stats.qmc.scale(sobol_points_01, lower_bounds, upper_bounds)

    x_initial_1 = []

    if not is_initial:
        for i in range(num_points):
            continuous_vars = list(scaled_points[i])
            categorical_var = random.choice(cell)
            x_initial_1.append(continuous_vars + categorical_var)
    else:
        for i in range(num_points):
            continuous_vars = list(scaled_points[i])
            categorical_var = cell_types[i]
            x_initial_1.append(continuous_vars + [categorical_var])
    return x_initial_1

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


class BO:
    def __init__(self, iterations, batch_size, initial_points, search_space):
        start_time = datetime.timestamp(datetime.now())
        self.Y = []
        self.time = []

        self.best_y_history = []
        self.current_best_y = -np.inf

        self.X_initial, self.Y_initial = self.create_initial_samples(initial_points, True)

        self.X_searchspace = self.create_searchspace(search_space, False)

        self.batch_bayesian(iterations, batch_size)

#IF is_initial = True:
#Creates first samples randomly (using create_sobol_initial_samples) and calculates the output at these points.
#Then, updates the best output if any of the new ouotputs is better than the previous best
#Reformats Y_initial to be a single column vector.

#IF is_initial = False:
#Creates a new set of [n_points] random samples, appends them onto any existing X values.
#Calculates all of the corresponding outputs, reformats into a single column vector
#Points generated are only random

    def create_initial_samples(self, n_points, is_initial):

        X_initial = create_sobol_initial_samples(n_points, is_initial)
        Y_initial = objective_func(X_initial)

        for row in Y_initial:
            if row > self.current_best_y:
                self.current_best_y = row
            self.best_y_history.append(self.current_best_y)

        X_initial = category_to_one_hot(X_initial)
        Y_initial = np.array(Y_initial)
        Y_initial = Y_initial.reshape(-1, 1)
        print(X_initial, Y_initial)

        return X_initial, Y_initial


#Gets all of the extrmemities of the valid searchspace - every 'corner' in 6D anda set of random samples
    def create_searchspace(self, n_points, is_initial):
        X_searchspace = create_sobol_initial_samples(n_points, is_initial)
        for i in [30, 40]:
            for j in [6, 8]:
                for k in [0, 50]:
                    for l in [0, 50]:
                        for m in [0, 50]:
                            for n in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
                                X_searchspace.append([i, j, k, l, m] + n)

        return X_searchspace

#A LITTLE CONFUSING
#Since x_searchspace is only the extremities of the bounds (as defined above), the chances that a sample
#or value in x_data is actually at one of the extremities is pretty much 0. This would result in indices being an empty set
#Also, why only if index.size>0? Python starts counting indices at 0, so if the row in question were the first row, it wouldnt be counted
#even though I'm assuming it probably should be.
#PLEASE CORRECT ME IF IM WRONG


    def find_indices_of_rows(self, x_data, x_searchspace):
        indices = set()
        for row in x_data:
            index = np.where(np.all(x_searchspace == row, axis=1))[0]
            if index.size > 0:
                indices.add(index[0])
        return indices

    def expected_improvement(self, mean, std, y_best, epsilon=1e-8, xi=2):
        std_safe = std + epsilon
    
        # *** CORRECTED TERM FOR MAXIMIZATION ***
        diff = mean - y_best - xi 

        Z = diff / std_safe
        pdf_Z = scipy.stats.norm.pdf(Z)
        cdf_Z = scipy.stats.norm.cdf(Z)

    # *** CORRECTED EI FORMULA FOR MAXIMIZATION ***
        ei = diff * cdf_Z + std_safe * pdf_Z

        ei[std < epsilon] = 0.0
        return ei

#Not sure that calculating the mean of the x values that have been used is necessary. I think the mean is only needed
#for the output, and in the equations that are used its just using the arbitrary mean function that we define.
#When you iterate again and recondition, the change in the output mean will be considered using the covariances
#May be that there are alternate ways of writing out and getting the same thing.
    def batch_bayesian(self, no_iterations, batch_size):

        X_data = self.X_initial.copy()
        Y_data = self.Y_initial.copy()

        sampled_indices = self.find_indices_of_rows(self.X_initial, self.X_searchspace)

        max_ei_history = []

        for i in range(no_iterations):
            print(f"\n--- Iteration {i + 1}/{no_iterations} ---")
            gp_model_0 = GP(X_data, Y_data, 'RBF', 5, True)
            print(gp_model_0.hypopt)
            y_best = np.max(Y_data)
            all_means, all_vars = gp_model_0.GP_inference_np(self.X_searchspace)
            all_stds = np.sqrt(all_vars)

            ei_values = self.expected_improvement(all_means, all_stds, y_best)

            for index in sampled_indices:
                if index < len(ei_values):
                    ei_values[index] = -1e9

            x_batch_to_evaluate = []

            max_ei_history.append(np.max(ei_values))

            print(f"  Selecting {batch_size} points for batch...")
            for j in range(batch_size):
                next_index = np.argmax(ei_values)

                x_next_numeric_1D = self.X_searchspace[next_index]

                x_batch_to_evaluate.append(x_next_numeric_1D)

                print(f"    Point {j + 1}: index {next_index}, EI {ei_values[next_index]:.4f}")
                sampled_indices.add(next_index)
                ei_values[next_index] = -1e9

            print(f"  Evaluating {batch_size} points...")

            x_batch_array = np.vstack(x_batch_to_evaluate)

            y_batch_real_results = []
            for k in range(batch_size):
                x_point_to_eval = x_batch_array[k].reshape(1, -1)
                x_readable_list = one_hot_to_category_name([x_point_to_eval.squeeze()])
                x_readable = x_readable_list[0]
                y_next_array = objective_func([x_readable])
                y_next_raw = np.max(y_next_array)
                if y_next_raw > self.current_best_y:
                    self.current_best_y = y_next_raw
                self.best_y_history.append(self.current_best_y)

                y_next_2D = y_next_raw.reshape(1, 1)
                print(f"Sampling: {x_readable} --> {y_next_2D.item():.4f}")
                y_batch_real_results.append(y_next_2D)

            y_batch_array = np.vstack(y_batch_real_results)

            X_data = np.vstack([X_data, x_batch_array])
            Y_data = np.vstack([Y_data, y_batch_array])

        print("\nOptimization finished.")
        best_index = np.argmax(Y_data)
        best_y = Y_data[best_index]
        best_x_numeric = X_data[best_index]
        best_x_readable_list = one_hot_to_category_name([best_x_numeric])
        best_x_readable = best_x_readable_list[0]

        print(f"Best titre found: {best_y.item():.4f} g/L")
        print(f"Best parameters: {best_x_readable}")

        num_iterations = len(self.best_y_history)
        iterations = list(range(1, num_iterations + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.best_y_history, marker='o', linestyle='-')
        plt.title('Best Maximum Solution Found vs. Iterations')
        plt.xlabel('No. of Iterations')
        plt.ylabel('Best Current Maximum Solution')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        iterations_axis = list(range(1, no_iterations + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(iterations_axis, max_ei_history, marker='o', linestyle='-', color='red')
        plt.title('Maximum Expected Improvement vs. Iterations')
        plt.xlabel('No. of Iterations')
        plt.ylabel('Maximum EI')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

bayesian = BO(15, 5, 6, 2**11)