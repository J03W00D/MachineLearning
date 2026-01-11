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

    # 1. ES Initialization Logic (Runs once)
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
    
    # Hyperparameters for ES
        your_optimization_alg.sigma = 0.15  # Noise strength (perturbation size) to 0.15 from 0.02
        your_optimization_alg.lr = 0.01     # Learning rate for the weight update
    
        your_optimization_alg.initialized = True

# Main Loop (Adjusted for ES: 2 episodes per iteration)
# Total iterations = max_episodes / 2
    for iteration in tqdm(range(int(max_episodes // 2))):
    
    # A. Generate Perturbation
        current_params = your_optimization_alg.get_params(policy_net)
        epsilon = torch.randn_like(current_params)
    
    # B. Antithetical Evaluations (Pos and Neg noise)
        rewards_eval = []
        for factor in [1, -1]:
        # Apply jittered parameters
            trial_params = current_params + (factor * your_optimization_alg.sigma * epsilon)
            your_optimization_alg.set_params(policy_net, trial_params)
        
        # Reset Env
            reset_ret = env.reset()
            state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            done = False
            ep_reward = 0
        
            while not done:
                # 1. Clean Feature Extraction
                def deep_flatten(items):
                    res = []
                    for x in items:
                        if isinstance(x, (list, np.ndarray)): res.extend(deep_flatten(x))
                        else: res.append(x)
                    return res

                state_np = np.array(deep_flatten([state]), dtype=np.float32)

    # Feature Scaling: Helps the network distinguish demand patterns
                # This 'standardization' often provides the final push to 11k
                #state_t = torch.from_numpy(state_np).float()
                #state_t = (state_t - state_t.mean()) / (state_t.std() + 1e-8)
                #state_input = state_t.unsqueeze(0)

                # 2. Force 4 features for the PolicyNetwork
                if state_np.size != 4:
                    tmp = np.zeros(4, dtype=np.float32)
                    tmp[:min(state_np.size, 4)] = state_np[:min(state_np.size, 4)]
                    state_np = tmp
                
                state_t = torch.from_numpy(state_np).unsqueeze(0)

                with torch.no_grad():
                    action_raw = torch.sigmoid(policy_net(state_t) - 2.0)
                
                action_np = action_raw.squeeze().cpu().numpy()
                if action_np.ndim == 0: action_np = np.array([action_np])
                action_scaled = np.clip(action_np * 5.0, env.action_space.low, env.action_space.high)
                
                step_ret = env.step(action_scaled)
                state, reward, done = step_ret[0], step_ret[1], step_ret[2]
                if len(step_ret) == 5: done = done or step_ret[3]
                ep_reward += reward
        
            rewards_eval.append(ep_reward)
        
        # Log data for plotting
            plot_data["reward_history"].append(ep_reward)
            plot_data["episodes"].append(len(plot_data["episodes"]))
        
        # Keep track of best
            if ep_reward > best_reward:
                best_reward = ep_reward
                best_policy = copy.deepcopy(policy_net.state_dict())
                torch.save(best_policy, save_f_path)

    # C. Evolutionary Gradient Update
    # Est. Gradient = (Reward_pos - Reward_neg) / (2 * sigma)
    # We move current_params in direction of better reward
        reward_pos, reward_neg = rewards_eval[0], rewards_eval[1]
    
    # Calculate the normalized advantage
        if reward_pos > reward_neg:
            advantage = 1.0
        elif reward_neg > reward_pos:
            advantage = -1.0
        else:
            advantage = 0.0 # No signal found
        
        update_step = (your_optimization_alg.lr / your_optimization_alg.sigma) * (advantage * epsilon)
        new_params = current_params + update_step
    # Gradually reduce noise strength for fine-tuning the 11k peak
       # your_optimization_alg.sigma *= 0.9995
    # Set the policy to the newly improved center point
        your_optimization_alg.set_params(policy_net, new_params)



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
        "Del Rio Chanona, Antonio",
        "Fons, Isabela",
    ]
    # CID (University Identifier)
    cids = ["16879875", "06069513"]
    # Would you like to be asked about this coursework in the final exam? 1 if YES, 0 else
    question = [1, 0]

    return best_policy, plot_data