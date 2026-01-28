import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Hyperparameters
ENV_NAME = "CartPole-v1"
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LR = 1e-4                  # Learning Rate decreased to observe stability
MEMORY_SIZE = 10000
MIN_MEMORY = 1000          # Warmup samples before training
TARGET_UPDATE = 10         # Hard update frequency
REWARD_THRESHOLD = 500     # Required reward to consider the agent "solved"
BOUNDARY_LIMIT = 1.0       # Define a virtual "small boundary"
HARD_LIMIT = 2.0           # Actual physical limit of the environment

# Force CPU for CartPole (usually faster due to low overhead)
device = torch.device("cpu")

# Replay Buffer Class for speed
class ReplayBuffer:
    """
    Pre-allocated numpy-backed replay buffer for speed.
    Stores transitions (state, action, reward, next_state, done).
    """
    def __init__(self, size, state_dim):
        self.states  = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, 1), dtype=np.int64)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.next_s  = np.zeros((size, state_dim), dtype=np.float32)
        self.dones   = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def push(self, state, action, reward, next_state, done):
        """Store a transition at the circular buffer pointer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_s[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Randomly sample a batch and return tensors on the configured device."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_s[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device)
        )


# Networks
class DQN(nn.Module):
    """
    Standard MLP for Q-function approximation.
    Input -> Hidden(128) -> ReLU -> Hidden(128) -> ReLU -> Output(Actions)
    """
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# Training Loop
def main():
    """
    Trains the DQN agent with a modified reward function to enforce
    balancing within a small boundary.
    """
    env = gym.make(ENV_NAME)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize Networks
    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Optimizer and Loss
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss() # More stable than MSE

    # Memory and Tracking
    memory = ReplayBuffer(MEMORY_SIZE, n_states)
    epsilon = EPSILON_START
    
    print(f"Training Started on {device}...")
    print(f"Constraint: Reward penalty applied if Cart > {BOUNDARY_LIMIT} units from center.")

    for episode in range(1, 601):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Action Selection (Epsilon-Greedy)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()

            # Environment Step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Reward Shaping for Boundary Constraint
            # 1. Base survival reward
            r = 1.0
            
            # 2. Distance Penalty: Penalize deviation from center (x=0)
            cart_pos = next_state[0]
            dist_penalty = abs(cart_pos) * 0.5 
            r -= dist_penalty

            # 3. Virtual Boundary Enforcement
            # If the cart goes beyond the small BOUNDARY_LIMIT, apply heavy penalty
            if abs(cart_pos) > BOUNDARY_LIMIT:
                r -= 2.0
            
            # Setup kill zones
            if abs(cart_pos) > HARD_LIMIT:
                r -= 10.0
            
            # 4. Terminal Penalty (Actual Crash)
            if terminated:
                r = -10.0

            # Store transition
            memory.push(state, action, r, next_state, int(done))
            state = next_state
            episode_reward += reward # Log the modified reward to see training impact

            # --- Optimization Step ---
            if memory.size >= MIN_MEMORY:
                s, a, r_batch, ns, d = memory.sample(BATCH_SIZE)

                # Compute Current Q
                current_q = policy_net(s).gather(1, a)

                # Compute Target Q
                with torch.no_grad():
                    max_next_q = target_net(ns).max(1)[0].unsqueeze(1)
                    target_q = r_batch + (GAMMA * max_next_q * (1 - d))

                loss = criterion(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # Updates
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Update Target Network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Logging (every 10 episodes)
        if episode % 10 == 0:
            print(f"Ep {episode:4d} | Score: {episode_reward:6.1f} | Epsilon: {epsilon:.2f} | Mem: {memory.size}")

        # Stop condition (modified score due to penalties)
        if episode_reward >= REWARD_THRESHOLD: 
            print(f"Solved (Stable within boundary) in {episode} episodes!")
            break

    env.close()
    
    # Visualization of the learned policy
    print("\nVisualizing Solution...")
    env = gym.make(ENV_NAME, render_mode="human")
    state, _ = env.reset()
    done = False
    
    while not done:
        env.render()
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = policy_net(state_t).argmax().item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    print("Simulation Closed.")


if __name__ == "__main__":
    main()