import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from modules.physics import TrajectoryPhysics

class HybridController:
    """
    Combines classical trajectory planning with RL optimization
    
    Philosophy:
    1. Use classical planning to generate a "reference trajectory"
    2. Train RL agent to follow + optimize this trajectory
    3. Much faster learning with guaranteed feasibility
    """
    
    def __init__(self, state_dim=26, action_dim=5):
        self.physics = TrajectoryPhysics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Small correction network (not full policy)
        self.correction_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),  # state + reference action
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Corrections in [-1, 1]
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.correction_network.parameters(), lr=1e-3)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def generate_reference_action(self, current_state, target, step_count):
        """
        Generate classical trajectory planning action
        Uses simple geometric planning with drilling constraints
        """
        # Extract current position from state
        current_MD = current_state[0] * 40000
        current_TVD = current_state[1] * 20000
        current_I = current_state[2] * 90
        current_A = current_state[3] * 360
        current_N = current_state[4] * 10000
        current_E = current_state[5] * 10000
        
        target_N = target.get('N', 4000)
        target_E = target.get('E', 4000)
        target_TVD = target.get('TVD', 15000)
        
        # Calculate what's needed
        delta_N = target_N - current_N
        delta_E = target_E - current_E
        delta_TVD = target_TVD - current_TVD
        
        horiz_dist = np.sqrt(delta_N**2 + delta_E**2)
        vert_dist = abs(delta_TVD)
        
        # PHASE 1: Build angle (KOP to ~30% of depth)
        # PHASE 2: Hold angle (~30% to ~80% of depth)
        # PHASE 3: Drop angle (~80% to target)
        
        progress = current_TVD / target_TVD
        
        # Determine required inclination
        if horiz_dist > 100 and vert_dist > 100:
            required_I = np.degrees(np.arctan2(horiz_dist, vert_dist))
        else:
            required_I = 0  # Drop to vertical near target
        
        # Phase-based inclination target
        if progress < 0.3:
            # Build phase
            target_I = min(required_I, 75)
            dI_ref = np.clip((target_I - current_I) * 0.1, -2, 2)
        elif progress < 0.8:
            # Hold phase
            target_I = required_I
            dI_ref = np.clip((target_I - current_I) * 0.05, -1, 1)
        else:
            # Drop phase
            target_I = max(0, required_I - 20)
            dI_ref = np.clip((target_I - current_I) * 0.15, -3, 1)
        
        # Azimuth control
        if horiz_dist > 200:
            required_A = np.degrees(np.arctan2(delta_E, delta_N)) % 360
            azimuth_error = required_A - current_A
            if azimuth_error > 180:
                azimuth_error -= 360
            elif azimuth_error < -180:
                azimuth_error += 360
            dA_ref = np.clip(azimuth_error * 0.05, -3, 3)
        else:
            dA_ref = 0
        
        # Create reference action (normalized to [-1, 1])
        ref_action = np.array([
            dI_ref / 3.0,      # Normalize by max change
            dA_ref / 8.0,      # Normalize by max change
            0.0,               # Bias terms
            0.0,
            0.0
        ], dtype=np.float32)
        
        return ref_action
    
    def select_action(self, state, target, step_count, use_correction=True):
        """
        Generate action: reference + learned correction
        """
        # Get reference action from classical planning
        ref_action = self.generate_reference_action(state, target, step_count)
        
        if not use_correction:
            return ref_action
        
        # Get learned correction
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ref_action_tensor = torch.FloatTensor(ref_action).unsqueeze(0).to(self.device)
        
        combined_input = torch.cat([state_tensor, ref_action_tensor], dim=1)
        
        with torch.no_grad():
            correction = self.correction_network(combined_input).cpu().numpy()[0]
        
        # Small corrections (max 30% deviation from reference)
        correction = correction * 0.3
        
        final_action = ref_action + correction
        final_action = np.clip(final_action, -1, 1)
        
        return final_action
    
    def update(self, states, actions, rewards, ref_actions):
        """
        Update correction network to improve upon reference trajectory
        Uses supervised learning + reward shaping
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        ref_actions = torch.FloatTensor(ref_actions).to(self.device)
        
        # Predict corrections
        combined_input = torch.cat([states, ref_actions], dim=1)
        predicted_corrections = self.correction_network(combined_input)
        
        # Actual corrections taken
        actual_corrections = actions - ref_actions
        
        # Loss: MSE weighted by rewards
        # Good outcomes → learn these corrections
        # Bad outcomes → minimize corrections (stick to reference)
        reward_weights = torch.sigmoid(rewards / 1000)  # Normalize rewards
        
        mse_loss = ((predicted_corrections - actual_corrections) ** 2).mean(dim=1)
        weighted_loss = (mse_loss * reward_weights).mean()
        
        # Also add regularization to keep corrections small
        l2_reg = (predicted_corrections ** 2).mean() * 0.01
        
        total_loss = weighted_loss + l2_reg
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.correction_network.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def save(self, path):
        torch.save({
            'correction_network': self.correction_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.correction_network.load_state_dict(checkpoint['correction_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def train_hybrid_agent(env, agent, target, n_episodes=300, max_steps=1000):
    """
    Training loop for hybrid agent
    Much faster convergence than pure RL
    """
    episode_rewards = []
    best_reward = -float('inf')
    
    print("\nTraining Hybrid RL + Planning Agent...")
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        # Storage for batch update
        states, actions, rewards, ref_actions = [], [], [], []
        
        for step in range(max_steps):
            # Generate action (reference + correction)
            # Early episodes: mostly reference, later: more correction
            use_correction = (episode > 10) and (np.random.random() > 0.1)
            
            ref_action = agent.generate_reference_action(state, target, step)
            action = agent.select_action(state, target, step, use_correction)
            
            # Execute
            next_state, reward, done, _ = env.step(action)
            
            # Store
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            ref_actions.append(ref_action)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update correction network
        if len(states) > 10:
            loss = agent.update(
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(ref_actions)
            )
        
        # Save best
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('models/best_hybrid_agent.pth')
        
        # Logging
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(20): {avg_reward:.2f} | "
                  f"Best: {best_reward:.2f}")
    
    return episode_rewards, agent