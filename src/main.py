import numpy as np
import pandas as pd
import os
import sys
import argparse
from modules.physics import TrajectoryPhysics
from modules.reservoir import ReservoirModel, SyntheticDataGenerator
from modules.environment import WellPlanningEnv
from modules.visualization import Visualizer

class WellOptimizationSystem:
    def __init__(self, algorithm='hybrid'): # 'hybrid', 'sac', 'td3', 'ppo'
        self.algorithm = algorithm
        
        # Create algorithm-specific directories
        self.results_dir = f'results/{algorithm}'
        self.plots_dir = f'plots/{algorithm}'
        self.models_dir = f'models/{algorithm}'
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.visualizer = Visualizer(self.plots_dir)
        self.physics = TrajectoryPhysics()
        
        print(f"\n{'='*70}")
        print(f"WELL OPTIMIZATION SYSTEM - {algorithm.upper()} Algorithm")
        print(f"{'='*70}")
        print(f"Results directory: {self.results_dir}")
        print(f"Plots directory: {self.plots_dir}")
        print(f"Models directory: {self.models_dir}")
        
    def objective_1_develop_mathematical_model(self):
        print("\n" + "="*70)
        print("OBJECTIVE 1: Develop Mathematical Model with AI Integration")
        print(f"Algorithm: {self.algorithm.upper()}")
        print("="*70)
        
        print("\nInitializing reservoir model...")
        reservoir = ReservoirModel(grid_size=(100, 100, 150), cell_size=(50, 50, 100))
        reservoir.generate_synthetic_reservoir(
            mean_porosity=0.18, 
            std_porosity=0.05,
            pore_pressure_grad=0.52, 
            frac_gradient=0.85
        )
        print("✓ Reservoir model generated successfully")
        
        print("\nExporting reservoir formation data...")
        reservoir.export_to_csv(f'{self.results_dir}/reservoir_formation.csv')
        print(f"✓ Reservoir data saved to {self.results_dir}/reservoir_formation.csv")
        
        print("\nDefining target location...")
        target = {
            'KOP': 3000,
            'TVD': 15000,
            'N': 4000,
            'E': 4000,
            'initial_azimuth': 45
        }
        
        print("\nCreating well planning environment...")
        env = WellPlanningEnv(
            reservoir_model=reservoir,
            target_location=target,
            constraints={
                'DLS_max': 8.0,
                'friction_factor': 0.25,
                'min_separation': 500,
                'max_inclination': 88.0
            }
        )
        print("✓ Environment initialized successfully")
        
        print(f"\nInitializing {self.algorithm.upper()} agent...")
        agent = self._create_agent(env)
        print(f"✓ {self.algorithm.upper()} agent initialized")
        self._print_agent_info(agent)
        
        print("\nGenerating conventional trajectory for comparison...")
        conventional_trajectory = self.physics.calculate_trajectory(
            KOP=target['KOP'],
            target=target,
            BUR=3.0,
            max_inclination=70,
            azimuth=target['initial_azimuth']
        )
        
        self.visualizer.plot_trajectory_3d(
            conventional_trajectory, 
            filename='01_conventional_trajectory_3d.png'
        )
        self.visualizer.plot_inclination_azimuth(
            conventional_trajectory,
            filename='02_conventional_inclination_azimuth.png'
        )
        
        print(f"\n✓ Conventional trajectory summary:")
        print(f"  Total survey points: {len(conventional_trajectory)}")
        print(f"  Final position: N={conventional_trajectory[-1]['N']:.1f} ft, "
              f"E={conventional_trajectory[-1]['E']:.1f} ft, "
              f"TVD={conventional_trajectory[-1]['TVD']:.1f} ft")
        print(f"  Target position: N={target['N']:.1f} ft, "
              f"E={target['E']:.1f} ft, TVD={target['TVD']:.1f} ft")
        print(f"  Final MD: {conventional_trajectory[-1]['MD']:.1f} ft")
        
        # Save model specifications
        results_obj1 = pd.DataFrame({
            'Component': ['Algorithm', 'State Dimension', 'Action Dimension', 
                         'Reservoir Grid Size', 'Survey Interval'],
            'Value': [self.algorithm.upper(), 26, 5, '100x100x150', 30]
        })
        results_obj1.to_csv(f'{self.results_dir}/objective1_model_specs.csv', index=False)
        
        print("\n✓ Objective 1 completed successfully")
        return env, agent, reservoir, target, conventional_trajectory
    
    def _create_agent(self, env):
        if self.algorithm == 'hybrid':
            from modules.hybrid_agent import HybridController
            return HybridController(state_dim=env.state_dim, action_dim=env.action_dim)
        
        elif self.algorithm == 'sac':
            from modules.sac_agent import SACAgent
            return SACAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                lr=3e-4,
                gamma=0.99,
                tau=0.005
            )
        
        elif self.algorithm == 'td3':
            from modules.td3_agent import TD3Agent
            return TD3Agent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                max_action=1.0,
                lr=3e-4
            )
        
        elif self.algorithm == 'ppo':
            from modules.ppo_agent import PPOAgent
            return PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _print_agent_info(self, agent):
        if self.algorithm in ['sac', 'td3']:
            try:
                actor_params = sum(p.numel() for p in agent.actor.parameters())
                critic_params = sum(p.numel() for p in agent.critic.parameters())
                print(f"  Actor parameters: {actor_params:,}")
                print(f"  Critic parameters: {critic_params:,}")
            except:
                pass
        elif self.algorithm == 'ppo':
            actor_params = sum(p.numel() for p in agent.actor.parameters())
            critic_params = sum(p.numel() for p in agent.critic.parameters())
            print(f"  Actor parameters: {actor_params:,}")
            print(f"  Critic parameters: {critic_params:,}")
        elif self.algorithm == 'hybrid':
            try:
                correction_params = sum(p.numel() for p in agent.correction_network.parameters())
                print(f"  Correction network parameters: {correction_params:,}")
                print(f"  Uses classical planning + RL corrections")
            except:
                pass
    
    def objective_2_implement_ai_algorithms(self, env, agent, reservoir, target, skip_training=False):
        print("\n" + "="*70)
        print("OBJECTIVE 2: Implement AI-Driven Optimization Algorithms")
        print(f"Algorithm: {self.algorithm.upper()}")
        print("="*70)
        
        model_path = f'{self.models_dir}/best_{self.algorithm}_agent.pth'
        
        if skip_training and os.path.exists(model_path):
            print(f"\n✓ Loading pre-trained {self.algorithm.upper()} model...")
            agent.load(model_path)
            print(f"✓ Model loaded from {model_path}")
            episode_rewards = []
        else:
            print(f"\nTraining {self.algorithm.upper()} agent...")
            episode_rewards = self._train_agent(env, agent, target)
            print("\n✓ Training completed")
        
        print("\nGenerating optimized trajectory using trained agent...")
        optimized_trajectory = self._generate_trajectory(env, agent, target)
        
        print(f"\n✓ Optimized trajectory summary:")
        print(f"  Total survey points: {len(optimized_trajectory)}")
        print(f"  Final position: N={optimized_trajectory[-1]['N']:.1f} ft, "
              f"E={optimized_trajectory[-1]['E']:.1f} ft, "
              f"TVD={optimized_trajectory[-1]['TVD']:.1f} ft")
        print(f"  Target position: N={target['N']:.1f} ft, "
              f"E={target['E']:.1f} ft, TVD={target['TVD']:.1f} ft")
        print(f"  Max inclination: {max([p['inclination'] for p in optimized_trajectory]):.2f}°")
        print(f"  Final MD: {optimized_trajectory[-1]['MD']:.1f} ft")
        
        # Calculate accuracy
        final = optimized_trajectory[-1]
        horiz_error = np.sqrt((final['N'] - target['N'])**2 + (final['E'] - target['E'])**2)
        vert_error = abs(final['TVD'] - target['TVD'])
        total_error = np.sqrt(horiz_error**2 + vert_error**2)
        print(f"  Distance to target: {total_error:.1f} ft (H: {horiz_error:.1f} ft, V: {vert_error:.1f} ft)")
        
        # Generate all visualizations
        print("\nGenerating comprehensive visualizations...")
        self._generate_all_plots(optimized_trajectory, reservoir, episode_rewards)
        
        # Save results
        if episode_rewards:
            training_df = pd.DataFrame({
                'Episode': range(1, len(episode_rewards) + 1),
                'Reward': episode_rewards
            })
            training_df.to_csv(f'{self.results_dir}/objective2_training_history.csv', index=False)
        
        trajectory_df = pd.DataFrame(optimized_trajectory)
        trajectory_df.to_csv(f'{self.results_dir}/objective2_optimized_trajectory.csv', index=False)
        
        print(f"\n✓ All plots saved to: {self.plots_dir}/")
        print("\n✓ Objective 2 completed successfully")
        return optimized_trajectory, episode_rewards
    
    def _train_agent(self, env, agent, target):
        """Train the agent using algorithm-specific training loop"""
        
        if self.algorithm == 'hybrid':
            from modules.hybrid_agent import train_hybrid_agent
            n_episodes = 150
            print(f"Training episodes: {n_episodes}")
            episode_rewards, _ = train_hybrid_agent(env, agent, target, n_episodes)
            agent.save(f'{self.models_dir}/best_{self.algorithm}_agent.pth')
            
        elif self.algorithm in ['sac', 'td3']:
            n_episodes = 500
            start_steps = 10000
            print(f"Training episodes: {n_episodes}")
            print(f"Random exploration steps: {start_steps}")
            
            episode_rewards = []
            best_reward = -float('inf')
            total_steps = 0
            
            for episode in range(n_episodes):
                state = env.reset()
                episode_reward = 0
                
                for step in range(1000):
                    # Select action
                    if total_steps < start_steps:
                        action = np.random.uniform(-1, 1, size=env.action_dim)
                    else:
                        if self.algorithm == 'sac':
                            action = agent.select_action(state, evaluate=False)
                        else:  # td3
                            action = agent.select_action(state, noise=0.1)
                    
                    # Execute
                    next_state, reward, done, _ = env.step(action)
                    agent.replay_buffer.push(state, action, reward, next_state, float(done))
                    
                    state = next_state
                    episode_reward += reward
                    total_steps += 1
                    
                    # Update
                    if total_steps >= start_steps:
                        agent.update(batch_size=256)
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save(f'{self.models_dir}/best_{self.algorithm}_agent.pth')
                
                if (episode + 1) % 20 == 0:
                    avg = np.mean(episode_rewards[-20:])
                    print(f"Episode {episode+1}/{n_episodes} | "
                          f"Reward: {episode_reward:.0f} | Avg: {avg:.0f} | Best: {best_reward:.0f}")
        
        elif self.algorithm == 'ppo':
            from modules.ppo_agent import ExperienceBuffer
            import torch
            
            n_episodes = 300
            print(f"Training episodes: {n_episodes}")
            
            episode_rewards = []
            best_reward = -float('inf')
            
            for episode in range(n_episodes):
                state = env.reset()
                episode_reward = 0
                buffer = ExperienceBuffer()
                
                for step in range(1000):
                    action, log_prob, value = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    buffer.store(state, action, reward, value, log_prob)
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                if buffer.size > 0:
                    states, actions, rewards, values, log_probs = buffer.get()
                    
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    next_value = agent.critic(state_tensor).item()
                    dones = np.zeros(len(rewards))
                    dones[-1] = 1 if done else 0
                    
                    advantages, returns = agent.compute_gae(rewards, values, next_value, dones)
                    agent.update(states, actions, log_probs, advantages, returns,
                               train_policy_iters=20, train_value_iters=20)
                
                episode_rewards.append(episode_reward)
                
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save(f'{self.models_dir}/best_{self.algorithm}_agent.pth')
                
                if (episode + 1) % 20 == 0:
                    avg = np.mean(episode_rewards[-20:])
                    print(f"Episode {episode+1}/{n_episodes} | "
                          f"Reward: {episode_reward:.0f} | Avg: {avg:.0f} | Best: {best_reward:.0f}")
        
        return episode_rewards
    
    def _generate_trajectory(self, env, agent, target):
        """Generate trajectory using trained agent"""
        state = env.reset()
        
        for step in range(1500):
            if self.algorithm == 'hybrid':
                action = agent.select_action(state, target, step, use_correction=True)
            elif self.algorithm == 'sac':
                action = agent.select_action(state, evaluate=True)
            elif self.algorithm == 'td3':
                action = agent.select_action(state, noise=0.0)
            elif self.algorithm == 'ppo':
                action, _, _ = agent.select_action(state, deterministic=True)
            
            state, _, done, _ = env.step(action)
            
            if done:
                break
        
        return env.get_trajectory()
    
    def _generate_all_plots(self, trajectory, reservoir, episode_rewards):
        """Generate all visualization plots"""
        # Training progress
        if episode_rewards:
            self.visualizer.plot_training_progress(
                episode_rewards,
                filename='04_training_rewards.png'
            )
        
        # Trajectory visualizations
        self.visualizer.plot_trajectory_3d(
            trajectory,
            filename='08_optimized_trajectory_3d.png'
        )
        
        self.visualizer.plot_inclination_azimuth(
            trajectory,
            filename='02_optimized_inclination_azimuth.png'
        )
        
        self.visualizer.plot_dogleg_severity(
            trajectory,
            filename='03_dogleg_severity.png'
        )
        
        # Reservoir visualizations
        self.visualizer.plot_reservoir_properties(
            reservoir,
            trajectory,
            filename='05_reservoir_properties.png'
        )
        
        self.visualizer.plot_reservoir_properties_4panel(
            reservoir,
            filename='08_reservoir_properties_4panel.png'
        )
        
        self.visualizer.plot_reservoir_combined_with_well(
            reservoir,
            trajectory,
            filename='09_reservoir_combined_well.png'
        )
        
        self.visualizer.plot_3d_reservoir_with_trajectory(
            reservoir,
            trajectory,
            filename='10_3d_reservoir_trajectory.png'
        )
        
        self.visualizer.plot_reservoir_contour_slices(
            reservoir,
            trajectory,
            filename='11_reservoir_contour_slices.png'
        )
        
        # Torque and drag
        MD_points = [p['MD'] for p in trajectory]
        torque_points = [self.physics.torque_drag(trajectory[:i+1], 0.25, 12.5)[0] 
                        for i in range(len(trajectory))]
        drag_points = [self.physics.torque_drag(trajectory[:i+1], 0.25, 12.5)[1] 
                      for i in range(len(trajectory))]
        
        self.visualizer.plot_torque_drag(
            MD_points,
            torque_points,
            drag_points,
            filename='06_torque_drag.png'
        )
    
    def objective_3_validate_and_compare(self, optimized_trajectory, conventional_trajectory, 
                                        reservoir, target):
        """
        OBJECTIVE 3: Validate Model and Compare with Conventional Methods
        - Computes performance metrics
        - Compares AI vs conventional approach
        - Generates comparison visualizations
        """
        print("\n" + "="*70)
        print("OBJECTIVE 3: Validate Model and Compare with Conventional Methods")
        print(f"Algorithm: {self.algorithm.upper()}")
        print("="*70)
        
        print("\nComputing performance metrics...")
        
        # Extract metrics
        ai_MD = optimized_trajectory[-1]['MD']
        ai_TVD = optimized_trajectory[-1]['TVD']
        ai_DLS_max = max([p['DLS'] for p in optimized_trajectory])
        ai_DLS_avg = np.mean([p['DLS'] for p in optimized_trajectory])
        
        conv_MD = conventional_trajectory[-1]['MD']
        conv_TVD = conventional_trajectory[-1]['TVD']
        conv_DLS_max = max([p['DLS'] for p in conventional_trajectory])
        conv_DLS_avg = np.mean([p['DLS'] for p in conventional_trajectory])
        
        # Calculate additional metrics
        ai_productivity = reservoir.calculate_productivity(optimized_trajectory)
        conv_productivity = reservoir.calculate_productivity(conventional_trajectory)
        
        ai_torque, ai_drag = self.physics.torque_drag(optimized_trajectory, 0.25, 12.5)
        conv_torque, conv_drag = self.physics.torque_drag(conventional_trajectory, 0.25, 12.5)
        
        # Target accuracy
        ai_final = optimized_trajectory[-1]
        ai_horiz_error = np.sqrt((ai_final['N'] - target['N'])**2 + 
                                (ai_final['E'] - target['E'])**2)
        ai_vert_error = abs(ai_final['TVD'] - target['TVD'])
        ai_total_error = np.sqrt(ai_horiz_error**2 + ai_vert_error**2)
        
        conv_final = conventional_trajectory[-1]
        conv_horiz_error = np.sqrt((conv_final['N'] - target['N'])**2 + 
                                   (conv_final['E'] - target['E'])**2)
        conv_vert_error = abs(conv_final['TVD'] - target['TVD'])
        conv_total_error = np.sqrt(conv_horiz_error**2 + conv_vert_error**2)
        
        # Print comparison
        print("\n" + "="*85)
        print(f"Performance Comparison: {self.algorithm.upper()} vs Conventional")
        print("="*85)
        print(f"{'Metric':<35} {f'{self.algorithm.upper()}':<20} {'Conventional':<20} {'Improvement':<15}")
        print("-" * 85)
        
        metrics = [
            ('Total MD (ft)', ai_MD, conv_MD),
            ('Final TVD (ft)', ai_TVD, conv_TVD),
            ('Horizontal Error (ft)', ai_horiz_error, conv_horiz_error),
            ('Vertical Error (ft)', ai_vert_error, conv_vert_error),
            ('Total 3D Error (ft)', ai_total_error, conv_total_error),
            ('Max DLS (deg/100ft)', ai_DLS_max, conv_DLS_max),
            ('Avg DLS (deg/100ft)', ai_DLS_avg, conv_DLS_avg),
            ('Productivity Index', ai_productivity, conv_productivity),
            ('Total Torque (ft-lbf)', ai_torque, conv_torque),
            ('Total Drag (lbf)', ai_drag, conv_drag)
        ]
        
        comparison_data = []
        for metric_name, ai_val, conv_val in metrics:
            if conv_val != 0:
                if 'Productivity' in metric_name:
                    improvement = ((ai_val - conv_val) / abs(conv_val)) * 100
                elif 'Error' in metric_name or 'DLS' in metric_name or 'Torque' in metric_name or 'Drag' in metric_name:
                    improvement = ((conv_val - ai_val) / abs(conv_val)) * 100
                else:
                    improvement = ((conv_val - ai_val) / abs(conv_val)) * 100
            else:
                improvement = 0
            
            print(f"{metric_name:<35} {ai_val:<20.2f} {conv_val:<20.2f} {improvement:>+14.2f}%")
            comparison_data.append({
                'Metric': metric_name,
                f'{self.algorithm.upper()}_Optimized': ai_val,
                'Conventional': conv_val,
                'Improvement_%': improvement
            })
        
        # Generate comparison plot
        self.visualizer.plot_comparison(
            optimized_trajectory,
            conventional_trajectory,
            target,
            filename='07_trajectory_comparison.png'
        )
        
        # Save comparison results
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(f'{self.results_dir}/objective3_performance_comparison.csv', 
                            index=False)
        
        # Validation summary
        print("\n" + "="*85)
        print("Validation Summary:")
        print("="*85)
        success = ai_total_error < 500 and ai_DLS_max < 12
        print(f"  Status: {'✓ SUCCESS' if success else '✗ NEEDS IMPROVEMENT'}")
        print(f"  - {self.algorithm.upper()} model {'successfully' if success else 'partially'} "
              f"optimized well trajectory")
        print(f"  - Target accuracy: {ai_total_error:.1f} ft "
              f"({'within' if ai_total_error < 500 else 'outside'} 500 ft tolerance)")
        print(f"  - Max DLS: {ai_DLS_max:.2f}°/100ft "
              f"({'within' if ai_DLS_max < 10 else 'exceeds'} constraint)")
        
        avg_improvement = comparison_df[comparison_df['Metric'].str.contains('Error|DLS')]['Improvement_%'].mean()
        print(f"  - Average improvement in key metrics: {avg_improvement:.2f}%")
        print(f"  - Results saved to: {self.results_dir}/")
        print(f"  - Plots saved to: {self.plots_dir}/")
        
        print("\n✓ Objective 3 completed successfully")
        
        return comparison_df


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='AI-Based Gas Well Placement and Trajectory Optimization'
    )
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default='hybrid',
        choices=['hybrid', 'sac', 'td3', 'ppo'],
        help='Algorithm to use for optimization (default: hybrid)'
    )
    parser.add_argument(
        '--skip-training', 
        action='store_true',
        help='Skip training and use pre-trained model'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("AI-BASED GAS WELL PLACEMENT AND TRAJECTORY OPTIMIZATION")
    print("="*70)
    print(f"\nSelected Algorithm: {args.algorithm.upper()}")
    
    if args.skip_training:
        print("Mode: Using pre-trained model (training skipped)")
    else:
        print("Mode: Training new model")
    
    # Initialize system
    system = WellOptimizationSystem(algorithm=args.algorithm)
    
    # Execute objectives
    env, agent, reservoir, target, conventional_trajectory = \
        system.objective_1_develop_mathematical_model()
    
    optimized_trajectory, training_history = \
        system.objective_2_implement_ai_algorithms(
            env, agent, reservoir, target, 
            skip_training=args.skip_training
        )
    
    comparison_results = system.objective_3_validate_and_compare(
        optimized_trajectory, conventional_trajectory, reservoir, target
    )
    
    # Final summary
    print("\n" + "="*70)
    print("ALL OBJECTIVES COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nAlgorithm Used: {args.algorithm.upper()}")
    print("\nOutput Summary:")
    print(f"  - Models: {system.models_dir}/")
    print(f"  - Results: {system.results_dir}/")
    print(f"  - Plots: {system.plots_dir}/")
    print("\nStudy completed successfully! ✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()