import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(ActorNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        x = self.network(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def log_prob(self, state, action):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=[256, 256, 128]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state).squeeze(-1)


class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
    def store(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def get(self):
        return (np.array(self.states), np.array(self.actions), 
                np.array(self.rewards), np.array(self.values), 
                np.array(self.log_probs))
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
    
    @property
    def size(self):
        return len(self.states)


class PPOAgent:
    def __init__(self, state_dim=26, action_dim=5, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, lambda_gae=0.95, clip_ratio=0.2):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        
    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                action_mean, _ = self.actor(state_tensor)
                action = action_mean
                log_prob = torch.zeros(1)
            else:
                action, log_prob = self.actor.sample(state_tensor)
            value = self.critic(state_tensor)
        
        return action.numpy()[0], log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns, 
               train_policy_iters=80, train_value_iters=80, minibatch_size=64):
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        
        for _ in range(train_policy_iters):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                new_log_probs = self.actor.log_prob(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(ratio * batch_advantages, 
                                        clipped_ratio * batch_advantages).mean()
                
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
        
        for _ in range(train_value_iters):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]
                
                values = self.critic(batch_states)
                value_loss = nn.MSELoss()(values, batch_returns)
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
