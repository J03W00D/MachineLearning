import copy
import time

import numpy as np
from streamlit import progress
import torch
import tqdm
from tqdm import tqdm

from common import PolicyNetwork
from ML4CE_RL_environment import MESCEnv
from utils import setup_model_saving


def your_optimization_alg(
    env: MESCEnv,
    policy_net: PolicyNetwork,
    *,
    max_episodes=2000,
    max_time=5 * 60,  # seconds
):
    # Create file to store model weigths
    save_f_path = setup_model_saving(algorithm="Your algorithm")

    # Initialize buffers to store data for plotting
    plot_data = {"reward_history": [], "episodes": []}

    start_time = time.time()
    best_reward = -np.inf
    best_policy = policy_net.state_dict()
    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE ABOVE THIS LINE
    # -----------------------------------------------------------------------------------

# 1. Enhanced CMA-ES with Strategic Improvements
    if not hasattr(your_optimization_alg, "initialized"):
        # Flatten parameters for easier manipulation
        def get_params(net):
            return torch.cat([p.data.view(-1) for p in net.parameters()])

        def set_params(net, new_params):
            start = 0
            for p in net.parameters():
                length = p.numel()
                p.data.copy_(new_params[start:start + length].view(p.shape))
                start += length

        your_optimization_alg.get_params = get_params
        your_optimization_alg.set_params = set_params
        
        # Get initial parameters
        initial_params = get_params(policy_net)
        n_params = len(initial_params)
        
        # CMA-ES Hyperparameters - Optimised for 5-minute budget
        your_optimization_alg.pop_size = 60  # Balance between speed and quality
        your_optimization_alg.mu = your_optimization_alg.pop_size // 2
        your_optimization_alg.mean = initial_params.clone()
        
        # Diagonal covariance with higher initial exploration
        your_optimization_alg.sigma_vec = torch.ones(n_params) * 0.6 #0.6 from 0.5
        
        # Evolution paths
        your_optimization_alg.ps = torch.zeros(n_params)
        your_optimization_alg.pc = torch.zeros(n_params)  # Add for rank-one update
        your_optimization_alg.cs = 0.3
        your_optimization_alg.cc = 0.3
        your_optimization_alg.damps = 1 #2 from 1
        
        # Selection weights (log-weighted)
        weights = torch.tensor([np.log(your_optimization_alg.mu + 0.5) - np.log(i + 1) 
                               for i in range(your_optimization_alg.mu)])
        your_optimization_alg.weights = weights / weights.sum()
        
        # Learning rates
        your_optimization_alg.c_cov = 0.25  # Faster covariance learning #0.08 from 0.25
        your_optimization_alg.c1 = 0.15     # Rank-one update rate
        
        # Tracking for restarts
        your_optimization_alg.best_ever_params = initial_params.clone()
        your_optimization_alg.best_ever_reward = -np.inf
        your_optimization_alg.no_improvement_count = 0
        
        your_optimization_alg.initialized = True
        your_optimization_alg.generation = 0

    # Main Loop
    num_iterations = int(max_episodes // your_optimization_alg.pop_size)
    
    for iteration in tqdm(range(num_iterations)):
        your_optimization_alg.generation += 1
        population_params = []
        population_deltas = []
        population_rewards = []
        
        # A. ANTITHETIC SAMPLING: Mirrored pairs for variance reduction
        half_pop = your_optimization_alg.pop_size // 2
        for _ in range(half_pop):
            # Sample from N(mean, diag(sigma_vec^2))
            z = torch.randn(len(your_optimization_alg.mean))
            delta_pos = your_optimization_alg.sigma_vec * z
            delta_neg = your_optimization_alg.sigma_vec * (-z)
            
            x_pos = your_optimization_alg.mean + delta_pos
            x_neg = your_optimization_alg.mean + delta_neg
            
            population_params.append(x_pos)
            population_deltas.append(delta_pos)
            population_params.append(x_neg)
            population_deltas.append(delta_neg)
        
        # B. Evaluate Population
        for x in population_params:
            your_optimization_alg.set_params(policy_net, x)
            
            reset_ret = env.reset()
            state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            done = False
            ep_reward = 0
            
            while not done:
                # Standardise state for network input
                state_np = np.array(state, dtype=np.float32).flatten()
                if state_np.size != 4:
                    tmp = np.zeros(4, dtype=np.float32)
                    tmp[:min(state_np.size, 4)] = state_np[:min(state_np.size, 4)]
                    state_np = tmp
                
                state_t = torch.from_numpy(state_np).float()
                # Normalise state, based roughly on env limits, NOT Fully Normalisation, just standardisation?
                state_t[0] /= 125 #125 from 150
                state_t[1] /= 75 #should be 125
                state_t[2] /= 75 #should be 350 if based on maxima
                state_t[3] /= 6
                state_input = state_t.unsqueeze(0)
                
                with torch.no_grad():
                    action_raw = policy_net(state_input)
                
                action_np = torch.tanh(action_raw).squeeze().cpu().numpy()
                if action_np.ndim == 0: 
                    action_np = np.array([action_np])
                
                action_scaled = action_np * 100
                action_scaled = np.clip(action_scaled, [0, 0, 0], [50, 50, 200])
                
                step_ret = env.step(action_scaled)
                state, reward, done = step_ret[0], step_ret[1], step_ret[2]
                if len(step_ret) == 5: 
                    done = done or step_ret[3]
                ep_reward += reward
            
            population_rewards.append(ep_reward)
            plot_data["reward_history"].append(ep_reward)
            plot_data["episodes"].append(len(plot_data["episodes"]))
            
            if ep_reward > best_reward:
                best_reward = ep_reward
                best_policy = copy.deepcopy(policy_net.state_dict())
                torch.save(best_policy, save_f_path)
            
            # Track absolute best for restarts
            if ep_reward > your_optimization_alg.best_ever_reward:
                your_optimization_alg.best_ever_reward = ep_reward
                your_optimization_alg.best_ever_params = x.clone()
                your_optimization_alg.no_improvement_count = 0
        

        # Update mean (weighted recombination)
        old_mean = your_optimization_alg.mean.clone()

        # --- C. MIRRORED SELECTION & UTILITY ---
        # We look at pairs: (population_params[0], population_params[1]), etc.
        half_pop = len(population_params) // 2
        reward_diffs = []
        pair_deltas = []

        for i in range(half_pop):
            r_pos = population_rewards[2*i]
            r_neg = population_rewards[2*i + 1]
            # The "gradient" direction is the difference between the two
            reward_diffs.append(r_pos - r_neg)
            # Use the positive delta as the reference direction
            pair_deltas.append(population_deltas[2*i])

        reward_diffs = np.array(reward_diffs)
        
        # Rank-based utilities for the differences (Standardizes the magnitude)
        # This prevents a single pair from dominating the update
        diff_indices = np.argsort(reward_diffs)
        utilities = np.zeros(half_pop)
        for i, idx in enumerate(diff_indices):
            # Map ranks to [-0.5, 0.5] range
            utilities[idx] = (i / (half_pop - 1)) - 0.5
        
        # D. DUAL EVOLUTION PATHS
        mean_shift = (your_optimization_alg.mean - old_mean) / (your_optimization_alg.sigma_vec + 1e-8)
        
        # Update ps (for step-size control)
        your_optimization_alg.ps = (1 - your_optimization_alg.cs) * your_optimization_alg.ps + \
                                    np.sqrt(your_optimization_alg.cs * (2 - your_optimization_alg.cs) * your_optimization_alg.mu) * mean_shift
        
        # Update pc (for covariance learning)
        your_optimization_alg.pc = (1 - your_optimization_alg.cc) * your_optimization_alg.pc + \
                                     np.sqrt(your_optimization_alg.cc * (2 - your_optimization_alg.cc) * your_optimization_alg.mu) * mean_shift
        
        # --- E. MIRRORED GRADIENT UPDATE (NES STYLE) ---
        # Update mean using the utility-weighted sum of deltas
        update_direction = torch.zeros_like(your_optimization_alg.mean)
        for i in range(half_pop):
            update_direction += utilities[i] * pair_deltas[i]
            
        # Move the mean (Learning Rate is implicitly controlled by utilities)
        # 0.5 is a safe, stable step size for this logic
        your_optimization_alg.mean += 0.5 * update_direction

        # Update Sigma (Diagonal Covariance)
        # We increase variance in dimensions that improved the reward
        for i in range(half_pop):
            # If the absolute difference is high, this direction is sensitive
            sensitivity = abs(utilities[i]) 
            improvement_dir = (pair_deltas[i]**2) / (your_optimization_alg.sigma_vec**2 + 1e-8)
            
            # Weighted moving average for the diagonal covariance
            cov_lr = your_optimization_alg.c_cov * sensitivity
            your_optimization_alg.sigma_vec = (1 - cov_lr) * your_optimization_alg.sigma_vec + \
                                              cov_lr * torch.sqrt(improvement_dir + 1e-8)
        
        # F. ADAPTIVE STEP-SIZE with CSA
        ps_norm = torch.norm(your_optimization_alg.ps)
        expected_norm = np.sqrt(len(your_optimization_alg.mean))
        your_optimization_alg.sigma_vec *= np.exp((your_optimization_alg.cs / your_optimization_alg.damps) * 
                                                   (ps_norm / expected_norm - 1))
        
        # Adaptive clamping based on progress
        if iteration < num_iterations * 0.4:
            your_optimization_alg.sigma_vec = torch.clamp(your_optimization_alg.sigma_vec, 0.02, 3.0)  # Wide range early
        else:
            your_optimization_alg.sigma_vec = torch.clamp(your_optimization_alg.sigma_vec, 0.01, 1.5)  # Tighter late

        # -----------------------------------------------------------------------------------
        # PLEASE DO NOT MODIFY THE CODE BELOW THIS LINE
        # -----------------------------------------------------------------------------------
        # Check time
        if (time.time() - start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    print(f"Policy model weights saved in: {save_f_path}")
    print(f"Best reward: {best_reward}")

    # Names of the team members (Imperial format)
    team_names = [
        "Kholwadia, Idrees",
        "Mahitkumar, Kundan","Wood,Joe","Wright,Joseph"
    ]
    # CID (University Identifier)
    cids = ["02286584", "02219861","02301095","02214563"] #Same order as for names
    # Would you like to be asked about this coursework in the final exam? 1 if YES, 0 else
    question = [0, 0, 0, 0] #No one wants to be asked about this coursework, TBC

    return best_policy, plot_data