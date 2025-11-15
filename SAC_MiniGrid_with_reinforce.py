import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque
import random
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class EpisodeMemory:
    """Memory for storing complete episodes for REINFORCE"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def push(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def __len__(self):
        return len(self.rewards)


class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        
        # CNN for image observations
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        h = conv2d_size_out(conv2d_size_out(state_shape[0]))
        w = conv2d_size_out(conv2d_size_out(state_shape[1]))
        linear_input_size = h * w * 32
        
        self.fc1 = nn.Linear(linear_input_size, action_dim)
        
    def forward(self, state):
        # Normalize pixel values
        x = state / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        action_probs = F.softmax(self.fc1(x), dim=-1)
        return action_probs
    
    def sample(self, state):
        action_probs = self.forward(state)
        # Add small epsilon for numerical stability
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, action_probs


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        
        # Q1 CNN
        self.conv1_q1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2_q1 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        
        # Q2 CNN
        self.conv1_q2 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2_q2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        h = conv2d_size_out(conv2d_size_out(state_shape[0]))
        w = conv2d_size_out(conv2d_size_out(state_shape[1]))
        linear_input_size = h * w * 32
        
        # Q1 architecture
        self.fc1 = nn.Linear(linear_input_size, action_dim)
        
        # Q2 architecture
        self.fc2 = nn.Linear(linear_input_size, action_dim)
        
    def forward(self, state):
        # Normalize pixel values
        x = state / 255.0
        
        # Q1
        q1 = F.relu(self.conv1_q1(x))
        q1 = F.relu(self.conv2_q1(q1))
        q1 = q1.reshape(q1.size(0), -1)
        q1 = self.fc1(q1)
        
        # Q2
        q2 = F.relu(self.conv1_q2(x))
        q2 = F.relu(self.conv2_q2(q2))
        q2 = q2.reshape(q2.size(0), -1)
        q2 = self.fc2(q2)
        
        return q1, q2


class SAC_REINFORCE:
    """SAC with REINFORCE policy gradient for on-policy updates"""
    def __init__(self, state_shape, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99, 
                 tau=0.01, alpha=0.2, auto_entropy_tuning=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Actor network
        self.actor = Actor(state_shape, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic networks (for SAC part)
        self.critic = Critic(state_shape, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_shape, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -0.98 * np.log(1.0 / action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
    
    def select_action(self, state, evaluate=False):
        """Select action and return log_prob for REINFORCE"""
        with torch.no_grad():
            state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            if evaluate:
                action_probs = self.actor(state)
                action = torch.argmax(action_probs, dim=-1)
                return action.item(), None
            else:
                action, log_prob, _ = self.actor.sample(state)
                return action.item(), log_prob.item()
    
    def compute_returns(self, rewards, gamma=0.99):
        """Compute discounted returns for REINFORCE"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update_reinforce(self, episode_memory):
        """REINFORCE policy gradient update"""
        if len(episode_memory) == 0:
            return {}
        
        # Compute returns
        returns = self.compute_returns(episode_memory.rewards, self.gamma)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(episode_memory.states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(episode_memory.actions).to(self.device)
        log_probs = torch.FloatTensor(episode_memory.log_probs).to(self.device)
        
        # Recompute action probabilities
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # REINFORCE loss with entropy bonus
        policy_loss = -(new_log_probs * returns).mean() - 0.01 * entropy
        
        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'avg_return': returns.mean().item()
        }
    
    def update_sac(self, replay_buffer, batch_size=64):
        """SAC critic update using replay buffer"""
        if len(replay_buffer) < batch_size:
            return {}
        
        # Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Convert states: (B, H, W, C) -> (B, C, H, W)
        state = torch.FloatTensor(state).permute(0, 3, 1, 2).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).permute(0, 3, 1, 2).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_action_probs = self.actor(next_state)
            next_q1_target, next_q2_target = self.critic_target(next_state)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # Calculate the expected Q-value weighted by action probabilities
            next_q_value = (next_action_probs * (next_q_target - self.alpha * torch.log(next_action_probs + 1e-8))).sum(dim=1, keepdim=True)
            target_q_value = reward + (1 - done) * self.gamma * next_q_value
        
        q1, q2 = self.critic(state)
        q1_selected = q1.gather(1, action.unsqueeze(1))
        q2_selected = q2.gather(1, action.unsqueeze(1))
        
        critic_loss = F.mse_loss(q1_selected, target_q_value) + F.mse_loss(q2_selected, target_q_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'q_value': q1_selected.mean().item()
        }
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


def train_sac_reinforce():
    # Environment - MiniGrid with image observations
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='rgb_array')
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    
    state_shape = env.observation_space.shape  # (H, W, C)
    action_dim = env.action_space.n
    
    print(f"State shape: {state_shape}")
    print(f"Action dimension: {action_dim}")
    print(f"Algorithm: SAC with REINFORCE policy gradient")
    
    # Hyperparameters
    max_episodes = 5000
    max_steps = 50
    batch_size = 32
    replay_buffer_size = 5000
    update_critic_freq = 4  # Update critic every N episodes
    
    # Initialize agent and buffers
    agent = SAC_REINFORCE(state_shape, action_dim, hidden_dim=64, lr=3e-4, gamma=0.95)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    episode_memory = EpisodeMemory()
    
    # Training loop
    episode_rewards = []
    training_stats = []
    
    for episode in range(max_episodes):
        obs, _ = env.reset()
        state = obs
        episode_reward = 0
        episode_memory.clear()
        
        episode_stats = {
            'policy_loss': [],
            'critic_loss': [],
            'q_value': [],
            'entropy': []
        }
        
        for step in range(max_steps):
            # Select action (with log_prob for REINFORCE)
            action, log_prob = agent.select_action(state)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs
            done = terminated or truncated
            
            # Store in both buffers
            replay_buffer.push(state, action, reward, next_state, done)
            episode_memory.push(state, action, reward, log_prob)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update policy using REINFORCE (on-policy)
        reinforce_stats = agent.update_reinforce(episode_memory)
        if reinforce_stats:
            episode_stats['policy_loss'].append(reinforce_stats['policy_loss'])
            episode_stats['entropy'].append(reinforce_stats['entropy'])
        
        # Update critic using SAC (off-policy) - less frequent
        if episode % update_critic_freq == 0 and len(replay_buffer) > batch_size:
            for _ in range(4):  # Multiple critic updates
                sac_stats = agent.update_sac(replay_buffer, batch_size)
                if sac_stats:
                    episode_stats['critic_loss'].append(sac_stats['critic_loss'])
                    episode_stats['q_value'].append(sac_stats['q_value'])
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            
            # Calculate average stats
            avg_stats = {
                'policy_loss': np.mean(episode_stats['policy_loss']) if episode_stats['policy_loss'] else 0,
                'critic_loss': np.mean(episode_stats['critic_loss']) if episode_stats['critic_loss'] else 0,
                'q_value': np.mean(episode_stats['q_value']) if episode_stats['q_value'] else 0,
                'entropy': np.mean(episode_stats['entropy']) if episode_stats['entropy'] else 0,
            }
            
            print(f"\nEpisode {episode} | Reward: {episode_reward:.2f} | Avg(20): {avg_reward:.2f}")
            print(f"  Policy Loss: {avg_stats['policy_loss']:.4f} | Critic Loss: {avg_stats['critic_loss']:.4f}")
            print(f"  Q-value: {avg_stats['q_value']:.3f} | Entropy: {avg_stats['entropy']:.3f}")
            
            training_stats.append(avg_stats)
        
        # Save best model
        if episode > 100 and episode_reward > np.percentile(episode_rewards[-100:], 90):
            agent.save('best_sac_reinforce_model.pth')
    
    env.close()
    return agent, episode_rewards, training_stats


if __name__ == "__main__":
    print("Training SAC with REINFORCE on MiniGrid-Empty-5x5-v0")
    print("=" * 60)
    agent, rewards, stats = train_sac_reinforce()
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Max reward achieved: {max(rewards):.2f}")
    print(f"{'='*60}")
