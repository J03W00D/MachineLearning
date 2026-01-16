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

# Enhanced CMA-ES Implementation with Antithetic Sampling, NES and Adaptive Clamping
    if not hasattr(your_optimization_alg, "initialized"): #Check if algorithm is initalised (set-up step)
        # Flatten network parameters for easier manipulation..
        def get_params(net):
            return torch.cat([p.data.view(-1) for p in net.parameters()]) #Flattened parameters into single vector

        def set_params(net, new_params): #Set network parameters from flattened vector
            start = 0
            for p in net.parameters():
                length = p.numel()
                p.data.copy_(new_params[start:start + length].view(p.shape))
                start += length
        
        #Attach utility functions to algorithm object..
        your_optimization_alg.get_params = get_params
        your_optimization_alg.set_params = set_params
        
        # Get initial parameter vector w. dimensionality
        initial_params = get_params(policy_net)
        n_params = len(initial_params)
        
        # Population parameters - Somewhat optimised for 5-minute budget
        your_optimization_alg.pop_size = 60  # Total population size
        your_optimization_alg.mu = your_optimization_alg.pop_size // 2 
        your_optimization_alg.mean = initial_params.clone() 
        
        # Diagonal covariance matrix with higher initial exploration..
        your_optimization_alg.sigma_vec = torch.ones(n_params) * 0.6 #initial step-size (increased from 0.5)
        
        # Evolution paths for tracking algorithm momentum for adaptation
        your_optimization_alg.ps = torch.zeros(n_params) # Step-size control.. (CSA later)

        # Adaptation learning rates
        your_optimization_alg.cs = 0.3  # Step-size learning rate
        your_optimization_alg.cc = 0.3 #Covar learning rate
        your_optimization_alg.damps = 1 #Damping factor on the step-size (reduced from 2)
        
        # Covariance Learning rate
        your_optimization_alg.c_cov = 0.25  # Diagonal covariance learning rate from 0.08 from 0.25
        
        # Mark it as initialised so don't repeat
        your_optimization_alg.initialized = True
        your_optimization_alg.generation = 0

    # Main Optimisation Loop
    num_iterations = int(max_episodes // your_optimization_alg.pop_size)
    
    for iteration in tqdm(range(num_iterations)):
        your_optimization_alg.generation += 1
        #Initialise containers for this generation
        population_params = [] #Parameter vectors for each individual
        population_deltas = [] #Mutation vectors for grad estimation
        population_rewards = [] #Fitness rewards
        
        # A Population Generation w. ANTITHETIC SAMPLING: Generate mirrored pairs for variance reduction & improved efficiency
        half_pop = your_optimization_alg.pop_size // 2 #half pop size as mirrored..
        for _ in range(half_pop):
            # Sample from N(mean, diag(sigma_vec^2)) [normal dist] in random direction
            z = torch.randn(len(your_optimization_alg.mean))
            #Create mirrored vectors
            delta_pos = your_optimization_alg.sigma_vec * z #+ve direction
            delta_neg = your_optimization_alg.sigma_vec * (-z) #-ve direction (the mirror)
            # Generate candidate solutions..
            x_pos = your_optimization_alg.mean + delta_pos
            x_neg = your_optimization_alg.mean + delta_neg
            # Store both candidaates and mutations by appending
            population_params.append(x_pos)
            population_deltas.append(delta_pos)
            population_params.append(x_neg)
            population_deltas.append(delta_neg)
        
        # B. Evaluate Population w. Fitness Assessment (run policy in the environment)
        for x in population_params:
            your_optimization_alg.set_params(policy_net, x) #Load candidate parameters into network
            #Initialise environment episode
            reset_ret = env.reset()
            state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            done = False
            ep_reward = 0
            
            while not done: #Run the episode w. current policy
                # Preprocess state for network input to ensure right dimensionality and data type
                state_np = np.array(state, dtype=np.float32).flatten()
                if state_np.size != 4:
                    tmp = np.zeros(4, dtype=np.float32)
                    tmp[:min(state_np.size, 4)] = state_np[:min(state_np.size, 4)]
                    state_np = tmp
                
                state_t = torch.from_numpy(state_np).float() #Convert to tensor
                # Normalise state, based roughly on env limits, NOT Fully Normalisation, just standardisation? Dodgiest part of algorithm
                state_t[0] /= 125 #125 from 150
                state_t[1] /= 75 #should be 125
                state_t[2] /= 75 #should be 350 if based on env limit max but 75 works better?
                state_t[3] /= 6 #days of the week? (0-6)
                state_input = state_t.unsqueeze(0)
                
                with torch.no_grad(): #generate action through using current policy
                    action_raw = policy_net(state_input)
                #Process the action through applying tanh activation and scaling 
                action_np = torch.tanh(action_raw).squeeze().cpu().numpy()
                if action_np.ndim == 0: 
                    action_np = np.array([action_np])
                #Arbitrary scaling to roughly match environment action space
                action_scaled = action_np * 100
                action_scaled = np.clip(action_scaled, [0, 0, 0], [50, 50, 200])
                #Execute action in environment
                step_ret = env.step(action_scaled)
                state, reward, done = step_ret[0], step_ret[1], step_ret[2]
                if len(step_ret) == 5: 
                    done = done or step_ret[3]
                ep_reward += reward
            
            #Store results
            population_rewards.append(ep_reward)
            plot_data["reward_history"].append(ep_reward)
            plot_data["episodes"].append(len(plot_data["episodes"]))
            
            #Update best policy if current solution turned out better
            if ep_reward > best_reward:
                best_reward = ep_reward
                best_policy = copy.deepcopy(policy_net.state_dict())
                torch.save(best_policy, save_f_path)
        
        # C.  Mirrored Selection & Utility Computation, compute grad information from mirrored pairings
        # Look at pairs: (population_params[0], population_params[1]), etc.
        half_pop = len(population_params) // 2
        reward_diffs = []
        pair_deltas = []

        for i in range(half_pop): #Process for each mirrored pair
            r_pos = population_rewards[2*i] #Reward for +ve 
            r_neg = population_rewards[2*i + 1] #Reward for -ve
            # The "gradient" direction estimate is the fitness difference between the two??
            reward_diffs.append(r_pos - r_neg)
            # Use the positive delta as the reference direction
            pair_deltas.append(population_deltas[2*i])

        reward_diffs = np.array(reward_diffs)
        
        # Rank-based utilities for the differences (Standardises the magnitude)
        # This also prevents a single outlier pair from dominating too much
        diff_indices = np.argsort(reward_diffs)
        utilities = np.zeros(half_pop)
        for i, idx in enumerate(diff_indices):
            # Map ranks to [-0.5, 0.5] range for balanced updates
            utilities[idx] = (i / (half_pop - 1)) - 0.5
        
        # D. Evolution Paths Update for Momentum Tracking, track movement patterns for adaptive control
        # Update mean (weighted recombination)
        old_mean = your_optimization_alg.mean.clone() #Perhaps D and E should be swapped? Although somehow it works better this way round..
        mean_shift = (your_optimization_alg.mean - old_mean) / (your_optimization_alg.sigma_vec + 1e-8) #Normalised mean shift in the normal way
        
        # Update ps (as explained earlier) (for step-size control [CSA])
        your_optimization_alg.ps = (1 - your_optimization_alg.cs) * your_optimization_alg.ps + \
                                    np.sqrt(your_optimization_alg.cs * (2 - your_optimization_alg.cs) * your_optimization_alg.mu) * mean_shift
        
        # E. Mean Update (NES [Natural Evolution Strategy] Style), move search centre towards promising regions using weighted gradient
        # Update mean using the utility-weighted sum of deltas
        update_direction = torch.zeros_like(your_optimization_alg.mean)
        for i in range(half_pop): #utility weighted sum
            update_direction += utilities[i] * pair_deltas[i]
            
        # Update the mean (remember Learning Rate is implicitly controlled by utilities)
        # 0.5 was chosen as a safe, stable step size for this logic 
        your_optimization_alg.mean += 0.5 * update_direction #0.5 is a stable learning rate here

        # F. Update Sigma (Diagonal Covariance), Adapt the strength of mutation per parameter
        # We increase variance in dimensions that improved the reward
        for i in range(half_pop):
            # If the absolute difference is high, this direction is sensitive..
            sensitivity = abs(utilities[i]) #Measuring sensitivity as a measure of how much this direction affected fitness
            improvement_dir = (pair_deltas[i]**2) / (your_optimization_alg.sigma_vec**2 + 1e-8) #Compute improvement direction (as normalised squared delta)
            
            # Weighted moving average for the diagonal covariance
            cov_lr = your_optimization_alg.c_cov * sensitivity #adaptive learning rate
            your_optimization_alg.sigma_vec = (1 - cov_lr) * your_optimization_alg.sigma_vec + \
                                              cov_lr * torch.sqrt(improvement_dir + 1e-8)
        # G.  Adaptive Step-Size with CSA [Cumulative Step-size Adaptation] to adjust overall mutation strength based on search progress
        ps_norm = torch.norm(your_optimization_alg.ps) #Determine length of evolution path
        expected_norm = np.sqrt(len(your_optimization_alg.mean)) #Expected length under random selection
        your_optimization_alg.sigma_vec *= np.exp((your_optimization_alg.cs / your_optimization_alg.damps) * 
                                                   (ps_norm / expected_norm - 1)) #CSA... increases sigma if consistently progressing, decreases if random
        # H. Adaptive clamping of sigma based on whether exploring or exploiting
        if iteration < num_iterations * 0.4:
            your_optimization_alg.sigma_vec = torch.clamp(your_optimization_alg.sigma_vec, 0.02, 3.0)  # Wide range early for exploration
        else:
            your_optimization_alg.sigma_vec = torch.clamp(your_optimization_alg.sigma_vec, 0.01, 1.5)  # Tighter late to encourage exploitation

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
    question = [1, 0, 0, 1] #IDREES and JOSEPH (Wright) want to be asked, Kundan and Joe (Wood) don't

    return best_policy, plot_data