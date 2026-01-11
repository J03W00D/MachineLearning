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


# 1. Initialization Logic (Runs once on the first episode)
    if not hasattr(your_optimization_alg, "initialized"):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # The Critic: Predicts V(s) to reduce variance
        your_optimization_alg.critic_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.ReLU(),
          #  torch.nn.Linear(64, 64),
           # torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        # Learnable exploration parameter
        your_optimization_alg.log_std = torch.tensor(np.ones(action_dim) * -0.5, requires_grad=True)
        
        # PPO usually performs better with a slightly lower learning rate
        your_optimization_alg.optimizer = torch.optim.Adam([
            {'params': policy_net.parameters(), 'lr': 2.5e-4}, #try shift to 3e-4 for stability?
            {'params': your_optimization_alg.critic_net.parameters(), 'lr': 1e-3},
            {'params': [your_optimization_alg.log_std], 'lr': 2.5e-4}
        ])
        
        your_optimization_alg.initialized = True

    # Main PPO Loop over episodes
    for episode in tqdm(range(int(max_episodes))):

        # --- A. Data Collection (Rollout) ---
        states, actions, rewards, old_log_probs, values = [], [], [], [], []

        # FIX: Initialize the raw tracker here
        episode_reward_raw = 0

        
        # Reset and handle different environment return types
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else (reset_ret if reset_ret is not None else env.state)
        done = False
        
        while not done:
            state_t = torch.from_numpy(np.array(state)).float()
            

            state_t=(state_t-state_t.mean())/(state_t.std() + 1e-8) #most recent
            # Policy forward pass
            # -2.0 bias helps start with smaller, low-cost inventory orders
            with torch.no_grad():
                action_mean = torch.sigmoid(policy_net(state_t) - 2.0)
                std = your_optimization_alg.log_std.exp()
                dist = torch.distributions.Normal(action_mean, std)
            
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                value = your_optimization_alg.critic_net(state_t)

            # Scale action for the environment
            action_np = action.detach().cpu().numpy()
            # If your prev best was 11k without scaling, keep scaling factor low (e.g., 2.0 - 5.0)
            action_scaled = np.clip(action_np * 5.0, env.action_space.low, env.action_space.high) #switched to 7 from 5?
            
            step_ret = env.step(action_scaled)
            next_state, reward, done = step_ret[0], step_ret[1], step_ret[2]
            # Handle truncated/info if 5-tuple returned
            if len(step_ret) == 5: done = done or step_ret[3]


            scaled_reward = reward / 100.0  # Scale reward to keep returns manageable

            states.append(state_t)
            actions.append(action)
            rewards.append(scaled_reward) #edited
            values.append(value.item())
            old_log_probs.append(log_prob)
            state = next_state

            episode_reward_raw += reward  # Keep track of unscaled reward for logging

        # --- B. Advantage Calculation ---
        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        old_log_probs_t = torch.stack(old_log_probs)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)

        # Compute discounted returns
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            # TD Error: reward + gamma * V(next) - V(current)
            delta = rewards[t] + 0.98 * next_value - values[t]
            gae = delta + 0.98 * 0.95 * gae # Gamma=0.98, Lambda=0.95 switch to 97
            advantages.insert(0, gae)
            next_value = values[t]
            returns.insert(0, gae + values[t])
            
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # CRITICAL: Normalize advantages ONCE before the K-epochs loop
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # --- C. PPO Clipped Update ---
        # Update K times per episode (standard for PPO)
        for _ in range(4):
            
            # Re-evaluate current policy
            new_mean = torch.sigmoid(policy_net(states_t) - 2.0)
            new_dist = torch.distributions.Normal(new_mean, your_optimization_alg.log_std.exp())
            new_log_probs = new_dist.log_prob(actions_t).sum(dim=-1)
            new_values = your_optimization_alg.critic_net(states_t).squeeze()

            # Calculate Probability Ratio
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            
            # PPO Clipped Objective
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_t # 0.2 clipping

            # Total Loss: Policy + Value (MSE) - Entropy (Bonus for exploration)
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(new_values, returns_t)
            entropy_loss = -0.01 * new_dist.entropy().mean() #MORE , from 0.01 to 0.05
            
            loss = policy_loss + 0.5 * value_loss + entropy_loss
            
            your_optimization_alg.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(your_optimization_alg.critic_net.parameters(), 0.5)
            your_optimization_alg.optimizer.step()

        # Update plotting and save best model
        #episode_reward = sum(rewards)
        plot_data["reward_history"].append(episode_reward_raw)
        plot_data["episodes"].append(episode)
        
        if episode_reward_raw > best_reward:
            best_reward = episode_reward_raw
            best_policy = copy.deepcopy(policy_net.state_dict())
            torch.save(best_policy, save_f_path)



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