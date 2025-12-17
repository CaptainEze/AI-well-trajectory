"""
    python train.py --algorithm hybrid --episodes 150
    python train.py --algorithm sac --episodes 500 --verbose
    python train.py --algorithm td3 --episodes 500
    python train.py --algorithm ppo --episodes 300
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from modules.environment import WellPlanningEnv
from modules.reservoir import ReservoirModel
from modules.visualization import Visualizer


class TrainingSession:
    """Handles training session for any algorithm"""
    
    def __init__(self, algorithm, output_dir='training_outputs'):
        self.algorithm = algorithm
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create organized directory structure
        self.session_dir = f'{output_dir}/{algorithm}_{self.timestamp}'
        self.models_dir = f'{self.session_dir}/models'
        self.plots_dir = f'{self.session_dir}/plots'
        self.logs_dir = f'{self.session_dir}/logs'
        
        for dir_path in [self.session_dir, self.models_dir, self.plots_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Also create standard directories for main.py compatibility
        os.makedirs(f'models/{algorithm}', exist_ok=True)
        os.makedirs(f'plots/{algorithm}', exist_ok=True)
        os.makedirs(f'results/{algorithm}', exist_ok=True)
        
        self.visualizer = Visualizer(self.plots_dir)
        
        print(f"\n{'='*70}")
        print(f"TRAINING SESSION: {algorithm.upper()}")
        print(f"{'='*70}")
        print(f"Session ID: {self.timestamp}")
        print(f"Output directory: {self.session_dir}")
        
    def setup_environment(self, target_config=None):
        """Setup reservoir and environment"""
        print("\n[1/4] Setting up environment...")
        
        # Create reservoir
        reservoir = ReservoirModel(grid_size=(100, 100, 150), cell_size=(50, 50, 100))
        reservoir.generate_synthetic_reservoir(
            mean_porosity=0.18,
            std_porosity=0.05,
            pore_pressure_grad=0.52,
            frac_gradient=0.85
        )
        print("  ✓ Reservoir model created")
        
        # Define target
        if target_config is None:
            target_config = {
                'KOP': 3000,
                'TVD': 15000,
                'N': 4000,
                'E': 4000,
                'initial_azimuth': 45
            }
        
        # Create environment
        env = WellPlanningEnv(
            reservoir_model=reservoir,
            target_location=target_config,
            constraints={
                'DLS_max': 8.0,
                'friction_factor': 0.25,
                'min_separation': 500,
                'max_inclination': 88.0
            }
        )
        print("  ✓ Environment initialized")
        print(f"  ✓ Target: N={target_config['N']}, E={target_config['E']}, "
              f"TVD={target_config['TVD']}")
        
        self.env = env
        self.reservoir = reservoir
        self.target = target_config
        
        return env, reservoir, target_config
    
    def create_agent(self):
        """Create agent based on algorithm"""
        print(f"\n[2/4] Creating {self.algorithm.upper()} agent...")
        
        if self.algorithm == 'hybrid':
            from modules.hybrid_agent import HybridController
            agent = HybridController(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim
            )
            print("  ✓ Hybrid RL + Planning controller created")
            
        elif self.algorithm == 'sac':
            from modules.sac_agent import SACAgent
            agent = SACAgent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                lr=3e-4,
                gamma=0.99,
                tau=0.005,
                automatic_entropy_tuning=True
            )
            print("  ✓ SAC agent created")
            print("  ✓ Automatic entropy tuning enabled")
            
        elif self.algorithm == 'td3':
            from modules.td3_agent import TD3Agent
            agent = TD3Agent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                max_action=1.0,
                lr=3e-4,
                gamma=0.99,
                tau=0.005
            )
            print("  ✓ TD3 agent created")
            
        elif self.algorithm == 'ppo':
            from modules.ppo_agent import PPOAgent
            agent = PPOAgent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                lr_actor=3e-4,
                lr_critic=1e-3
            )
            print("  ✓ PPO agent created")
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.agent = agent
        return agent
    
    def train(self, n_episodes, verbose=False):
        """Training loop with monitoring"""
        print(f"\n[3/4] Training {self.algorithm.upper()} agent...")
        print(f"  Episodes: {n_episodes}")
        
        if self.algorithm == 'hybrid':
            rewards = self._train_hybrid(n_episodes, verbose)
        elif self.algorithm in ['sac', 'td3']:
            rewards = self._train_offpolicy(n_episodes, verbose)
        elif self.algorithm == 'ppo':
            rewards = self._train_ppo(n_episodes, verbose)
        
        print("\n  ✓ Training completed!")
        
        # Save training log
        log_df = pd.DataFrame({
            'episode': range(1, len(rewards) + 1),
            'reward': rewards
        })
        log_df.to_csv(f'{self.logs_dir}/training_log.csv', index=False)
        
        return rewards
    
    def _train_hybrid(self, n_episodes, verbose):
        """Train hybrid agent"""
        from modules.hybrid_agent import train_hybrid_agent
        
        rewards, _ = train_hybrid_agent(
            self.env, self.agent, self.target, n_episodes
        )
        
        # Save to both session and standard directory
        self.agent.save(f'{self.models_dir}/best_model.pth')
        self.agent.save(f'models/{self.algorithm}/best_{self.algorithm}_agent.pth')
        
        return rewards
    
    def _train_offpolicy(self, n_episodes, verbose):
        """Train SAC or TD3"""
        start_steps = 10000
        max_steps = 1000
        
        print(f"  Random exploration: {start_steps} steps")
        print(f"  Max steps per episode: {max_steps}")
        
        rewards = []
        best_reward = -float('inf')
        total_steps = 0
        
        # Tracking metrics
        success_count = 0
        recent_distances = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps):
                # Select action
                if total_steps < start_steps:
                    action = np.random.uniform(-1, 1, size=self.env.action_dim)
                else:
                    if self.algorithm == 'sac':
                        action = self.agent.select_action(state, evaluate=False)
                    else:  # td3
                        action = self.agent.select_action(state, noise=0.1)
                
                # Execute
                next_state, reward, done, _ = self.env.step(action)
                self.agent.replay_buffer.push(state, action, reward, next_state, float(done))
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Update networks
                if total_steps >= start_steps:
                    self.agent.update(batch_size=256)
                
                if done:
                    break
            
            rewards.append(episode_reward)
            
            # Track success
            trajectory = self.env.get_trajectory()
            final_pos = trajectory[-1]
            distance = np.sqrt(
                (final_pos['N'] - self.target['N'])**2 +
                (final_pos['E'] - self.target['E'])**2 +
                (final_pos['TVD'] - self.target['TVD'])**2
            )
            recent_distances.append(distance)
            if len(recent_distances) > 20:
                recent_distances.pop(0)
            
            if distance < 500:
                success_count += 1
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save(f'{self.models_dir}/best_model.pth')
                self.agent.save(f'models/{self.algorithm}/best_{self.algorithm}_agent.pth')
            
            # Logging
            if (episode + 1) % 10 == 0 or verbose:
                avg_reward = np.mean(rewards[-10:])
                avg_dist = np.mean(recent_distances[-10:]) if recent_distances else 0
                success_rate = (success_count / (episode + 1)) * 100
                
                print(f"  Ep {episode+1:4d}/{n_episodes} | "
                      f"R: {episode_reward:8.0f} | "
                      f"Avg: {avg_reward:8.0f} | "
                      f"Best: {best_reward:8.0f} | "
                      f"Dist: {distance:6.0f}ft | "
                      f"Success: {success_rate:5.1f}%")
        
        return rewards
    
    def _train_ppo(self, n_episodes, verbose):
        """Train PPO agent"""
        from modules.ppo_agent import ExperienceBuffer
        import torch
        
        rewards = []
        best_reward = -float('inf')
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            buffer = ExperienceBuffer()
            
            for step in range(1000):
                action, log_prob, value = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                buffer.store(state, action, reward, value, log_prob)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update
            if buffer.size > 0:
                states, actions, rewards_buf, values, log_probs = buffer.get()
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_value = self.agent.critic(state_tensor).item()
                dones = np.zeros(len(rewards_buf))
                dones[-1] = 1 if done else 0
                
                advantages, returns = self.agent.compute_gae(
                    rewards_buf, values, next_value, dones
                )
                self.agent.update(
                    states, actions, log_probs, advantages, returns,
                    train_policy_iters=20, train_value_iters=20
                )
            
            rewards.append(episode_reward)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save(f'{self.models_dir}/best_model.pth')
                self.agent.save(f'models/{self.algorithm}/best_{self.algorithm}_agent.pth')
            
            if (episode + 1) % 10 == 0 or verbose:
                avg_reward = np.mean(rewards[-10:])
                print(f"  Ep {episode+1:4d}/{n_episodes} | "
                      f"R: {episode_reward:8.0f} | "
                      f"Avg: {avg_reward:8.0f} | "
                      f"Best: {best_reward:8.0f}")
        
        return rewards
    
    def evaluate(self, n_eval=5):
        """Evaluate trained agent"""
        print(f"\n[4/4] Evaluating {self.algorithm.upper()} agent...")
        print(f"  Evaluation runs: {n_eval}")
        
        eval_trajectories = []
        eval_distances = []
        
        for i in range(n_eval):
            state = self.env.reset()
            
            for step in range(1500):
                if self.algorithm == 'hybrid':
                    action = self.agent.select_action(state, self.target, step, True)
                elif self.algorithm == 'sac':
                    action = self.agent.select_action(state, evaluate=True)
                elif self.algorithm == 'td3':
                    action = self.agent.select_action(state, noise=0.0)
                elif self.algorithm == 'ppo':
                    action, _, _ = self.agent.select_action(state, deterministic=True)
                
                state, _, done, _ = self.env.step(action)
                
                if done:
                    break
            
            trajectory = self.env.get_trajectory()
            eval_trajectories.append(trajectory)
            
            final = trajectory[-1]
            distance = np.sqrt(
                (final['N'] - self.target['N'])**2 +
                (final['E'] - self.target['E'])**2 +
                (final['TVD'] - self.target['TVD'])**2
            )
            eval_distances.append(distance)
            
            print(f"  Run {i+1}: Distance to target = {distance:.1f} ft")
        
        # Use best trajectory for visualization
        best_idx = np.argmin(eval_distances)
        best_trajectory = eval_trajectories[best_idx]
        
        print(f"\n  Best run: #{best_idx + 1}")
        print(f"  Average distance: {np.mean(eval_distances):.1f} ft")
        print(f"  Best distance: {np.min(eval_distances):.1f} ft")
        
        # Generate visualizations
        print("\n  Generating visualizations...")
        self.visualizer.plot_trajectory_3d(
            best_trajectory,
            filename='trajectory_3d.png'
        )
        self.visualizer.plot_inclination_azimuth(
            best_trajectory,
            filename='inclination_azimuth.png'
        )
        self.visualizer.plot_dogleg_severity(
            best_trajectory,
            filename='dogleg_severity.png'
        )
        
        # Save best trajectory
        traj_df = pd.DataFrame(best_trajectory)
        traj_df.to_csv(f'{self.logs_dir}/best_trajectory.csv', index=False)
        
        # Copy to standard location
        traj_df.to_csv(f'results/{self.algorithm}/optimized_trajectory.csv', index=False)
        
        print(f"  ✓ Visualizations saved to: {self.plots_dir}")
        
        return best_trajectory, eval_distances
    
    def generate_report(self, training_rewards, eval_distances):
        """Generate training report"""
        report = {
            'algorithm': self.algorithm,
            'timestamp': self.timestamp,
            'episodes_trained': len(training_rewards),
            'best_training_reward': max(training_rewards),
            'final_avg_reward': np.mean(training_rewards[-50:]),
            'eval_best_distance_ft': min(eval_distances),
            'eval_avg_distance_ft': np.mean(eval_distances),
            'eval_success_rate_%': (sum(1 for d in eval_distances if d < 500) / len(eval_distances)) * 100,
        }
        
        report_df = pd.DataFrame([report])
        report_df.to_csv(f'{self.logs_dir}/training_report.csv', index=False)
        
        # Print report
        print("\n" + "="*70)
        print("TRAINING REPORT")
        print("="*70)
        for key, value in report.items():
            print(f"  {key}: {value}")
        print("="*70)
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description='Train well trajectory optimization agent'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='hybrid',
        choices=['hybrid', 'sac', 'td3', 'ppo'],
        help='Algorithm to train (default: hybrid)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed training logs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='training_outputs',
        help='Output directory for training session'
    )
    
    args = parser.parse_args()
    
    # Set default episodes based on algorithm
    if args.episodes is None:
        default_episodes = {
            'hybrid': 150,
            'sac': 500,
            'td3': 500,
            'ppo': 300
        }
        args.episodes = default_episodes[args.algorithm]
    
    # Create training session
    session = TrainingSession(args.algorithm, args.output_dir)
    
    # Setup
    session.setup_environment()
    session.create_agent()
    
    # Train
    training_rewards = session.train(args.episodes, args.verbose)
    
    # Evaluate
    best_trajectory, eval_distances = session.evaluate(n_eval=5)
    
    # Generate report
    session.generate_report(training_rewards, eval_distances)
    
    # Training progress visualization
    session.visualizer.plot_training_progress(
        training_rewards,
        filename='training_progress.png'
    )
    
    print("\n✓ Training session completed successfully!")
    print(f"✓ All outputs saved to: {session.session_dir}")


if __name__ == "__main__":
    main()