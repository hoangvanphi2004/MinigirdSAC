import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

# SAC Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-3, gamma=0.99, tau=0.01, alpha=0.2, auto_entropy_tuning=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store number of discrete actions, but use continuous action_dim=1
        self.n_actions = action_dim
        continuous_action_dim = 1
        
        self.actor = Actor(state_dim, continuous_action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, continuous_action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, continuous_action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -continuous_action_dim  # Heuristic value
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        self.action_dim = continuous_action_dim
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            _, _, continuous_action = self.actor.sample(state)
        else:
            continuous_action, _, _ = self.actor.sample(state)
        # Convert continuous action to discrete
        continuous_action = continuous_action.detach().cpu().numpy()[0]
        discrete_action = int((continuous_action[0] + 1) / 2 * (self.n_actions - 1))
        discrete_action = np.clip(discrete_action, 0, self.n_actions - 1)
        return discrete_action
    
    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        # Convert discrete actions to continuous for the actor-critic
        action_continuous = torch.FloatTensor([[(a / (self.n_actions - 1)) * 2 - 1] for a in action]).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(state, action_continuous)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        new_action, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        # Update alpha if automatic entropy tuning is enabled
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_value = alpha_loss.item()
        else:
            alpha_loss_value = 0
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Calculate average Q values for monitoring
        avg_q1 = current_q1.mean().item()
        avg_q2 = current_q2.mean().item()
        avg_q = (avg_q1 + avg_q2) / 2
        
        return actor_loss.item(), critic_loss.item(), avg_q, alpha_loss_value

# MiniGrid Wrapper to flatten observation
class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space['image'].shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[0] * obs_shape[1] * obs_shape[2],),
            dtype='uint8'
        )
    
    def observation(self, obs):
        return obs['image'].flatten()

def train():
    # Create 5x5 MiniGrid environment
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode=None)
    env = FlattenObsWrapper(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    agent = SAC(state_dim, action_dim)
    replay_buffer = ReplayBuffer(capacity=100000)
    
    max_episodes = 200
    max_steps = 100
    batch_size = 256
    start_steps = 1000
    
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    episode_lengths = []
    q_values = []
    alpha_values = []
    alpha_losses = []
    
    total_steps = 0
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state.astype(np.float32) / 255.0  # Normalize
        episode_reward = 0
        episode_length = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        episode_q_value = 0
        episode_alpha_loss = 0
        update_count = 0
        
        for step in range(max_steps):
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.astype(np.float32) / 255.0
            done = terminated or truncated
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            if len(replay_buffer) > batch_size and total_steps >= start_steps:
                actor_loss, critic_loss, q_val, alpha_loss = agent.update(replay_buffer, batch_size)
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                episode_q_value += q_val
                episode_alpha_loss += alpha_loss
                update_count += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if update_count > 0:
            actor_losses.append(episode_actor_loss / update_count)
            critic_losses.append(episode_critic_loss / update_count)
            q_values.append(episode_q_value / update_count)
            alpha_losses.append(episode_alpha_loss / update_count)
        
        # Track alpha value
        alpha_values.append(agent.alpha)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_actor_loss = np.mean(actor_losses[-10:]) if len(actor_losses) >= 10 else (np.mean(actor_losses) if actor_losses else 0)
            avg_critic_loss = np.mean(critic_losses[-10:]) if len(critic_losses) >= 10 else (np.mean(critic_losses) if critic_losses else 0)
            avg_q_value = np.mean(q_values[-10:]) if len(q_values) >= 10 else (np.mean(q_values) if q_values else 0)
            avg_alpha_loss = np.mean(alpha_losses[-10:]) if len(alpha_losses) >= 10 else (np.mean(alpha_losses) if alpha_losses else 0)
            
            print(f"Episode {episode + 1}/{max_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f} | Avg Length: {avg_length:.1f}")
            print(f"  Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")
            print(f"  Q-Value: {avg_q_value:.4f} | Alpha: {agent.alpha:.4f} | Alpha Loss: {avg_alpha_loss:.4f}")
            print(f"  Buffer Size: {len(replay_buffer)} | Total Steps: {total_steps}")
            print(f"  Latest Reward: {episode_reward:.3f} | Latest Length: {episode_length}")
            print("-" * 60)
    
    env.close()
    
    # Print training summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Total Steps: {total_steps}")
    print(f"Average Reward (all): {np.mean(episode_rewards):.3f}")
    print(f"Average Reward (last 100): {np.mean(episode_rewards[-100:]):.3f}")
    print(f"Best Reward: {np.max(episode_rewards):.3f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    if actor_losses:
        print(f"Final Actor Loss: {actor_losses[-1]:.4f}")
        print(f"Final Critic Loss: {critic_losses[-1]:.4f}")
    if q_values:
        print(f"Final Q-Value: {q_values[-1]:.4f}")
        print(f"Average Q-Value: {np.mean(q_values):.4f}")
    print(f"Alpha (Entropy Coefficient): {agent.alpha:.4f}")
    print(f"Initial Alpha: {alpha_values[0]:.4f} | Final Alpha: {alpha_values[-1]:.4f}")
    if alpha_losses:
        print(f"Average Alpha Loss: {np.mean(alpha_losses):.4f}")
    print("=" * 60 + "\n")
    
    # Test the trained agent
    print("Testing trained agent...")
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')
    env = FlattenObsWrapper(env)
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(5):
        state, _ = env.reset()
        state = state.astype(np.float32) / 255.0
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.astype(np.float32) / 255.0
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.3f}, Length = {episode_length}")
    
    print(f"\nTest Average Reward: {np.mean(test_rewards):.3f}")
    print(f"Test Average Length: {np.mean(test_lengths):.1f}")
    
    env.close()
    
    return agent

def test_agent(agent=None, num_episodes=10, render=True):
    """Test a trained agent"""
    if agent is None:
        print("No agent provided. Please train an agent first.")
        return
    
    render_mode = 'human' if render else None
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode=render_mode)
    env = FlattenObsWrapper(env)
    
    test_rewards = []
    test_lengths = []
    success_count = 0
    
    print("\n" + "=" * 60)
    print(f"TESTING AGENT - {num_episodes} Episodes")
    print("=" * 60)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.astype(np.float32) / 255.0
        episode_reward = 0
        episode_length = 0
        
        for step in range(100):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.astype(np.float32) / 255.0
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                if reward > 0:  # Success
                    success_count += 1
                break
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        status = "✓ Success" if episode_reward > 0 else "✗ Failed"
        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:6.3f} | Length = {episode_length:3d} | {status}")
    
    print("-" * 60)
    print(f"Test Results:")
    print(f"  Average Reward: {np.mean(test_rewards):.3f} ± {np.std(test_rewards):.3f}")
    print(f"  Average Length: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")
    print(f"  Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"  Best Reward: {np.max(test_rewards):.3f}")
    print(f"  Worst Reward: {np.min(test_rewards):.3f}")
    print("=" * 60)
    
    env.close()
    
    return {
        'rewards': test_rewards,
        'lengths': test_lengths,
        'success_rate': success_count / num_episodes,
        'avg_reward': np.mean(test_rewards),
        'avg_length': np.mean(test_lengths)
    }

if __name__ == "__main__":
    # Train the agent
    trained_agent = train()
    
    # Additional testing with more episodes
    print("\n\nRunning extended testing...")
    test_results = test_agent(trained_agent, num_episodes=20, render=False)
    
    print("\n✓ Training and testing completed!")
