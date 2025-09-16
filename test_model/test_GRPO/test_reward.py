import torch
from AAAI2025.GRPO.GRPO import compute_trajectory_reward


if __name__ == "__main__":
    # Example usage of compute_trajectory_reward
    # Assuming predicted_trajectories is a tensor of shape [B, N, T, 2 or 3]
    # where B is the batch size, N is the number of agents, and T is the trajectory length.

    
    # Example predicted trajectori  es (batch size 2, 3 agents, 5 time steps, 2D trajectories)
    predicted_trajectories = torch.randn(2, 32, 18, 5)  # Randomly generated for demonstration
    
    # Compute the reward for the predicted trajectories
    reward = compute_trajectory_reward(predicted_trajectories)
    print("Computed Reward:", reward.shape)