import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        sns.set_style("whitegrid")
        
    def plot_trajectory_3d(self, trajectory, filename='01_trajectory_3d.png'):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        N = [p['N'] for p in trajectory]
        E = [p['E'] for p in trajectory]
        TVD = [p['TVD'] for p in trajectory]
        
        ax.plot(N, E, TVD, 'b-', linewidth=2, label='Well Path')
        ax.scatter(N[0], E[0], TVD[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(N[-1], E[-1], TVD[-1], c='red', s=100, marker='*', label='End')
        
        ax.set_xlabel('North (ft)', fontsize=10)
        ax.set_ylabel('East (ft)', fontsize=10)
        ax.set_zlabel('TVD (ft)', fontsize=10)
        ax.set_title('3D Well Trajectory', fontsize=12, fontweight='bold')
        ax.invert_zaxis()
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_inclination_azimuth(self, trajectory, filename='02_inclination_azimuth.png'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        MD = [p['MD'] for p in trajectory]
        I = [p['inclination'] for p in trajectory]
        A = [p['azimuth'] for p in trajectory]
        
        ax1.plot(MD, I, 'b-', linewidth=2)
        ax1.set_xlabel('Measured Depth (ft)', fontsize=10)
        ax1.set_ylabel('Inclination (degrees)', fontsize=10)
        ax1.set_title('Inclination vs Measured Depth', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(MD, A, 'r-', linewidth=2)
        ax2.set_xlabel('Measured Depth (ft)', fontsize=10)
        ax2.set_ylabel('Azimuth (degrees)', fontsize=10)
        ax2.set_title('Azimuth vs Measured Depth', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dogleg_severity(self, trajectory, filename='03_dogleg_severity.png'):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        MD = [p['MD'] for p in trajectory]
        DLS = [p['DLS'] for p in trajectory]
        
        ax.plot(MD, DLS, 'g-', linewidth=2, label='DLS')
        ax.axhline(y=6, color='orange', linestyle='--', label='Moderate Limit (6°/100ft)')
        ax.axhline(y=10, color='red', linestyle='--', label='Critical Limit (10°/100ft)')
        
        ax.set_xlabel('Measured Depth (ft)', fontsize=10)
        ax.set_ylabel('Dogleg Severity (°/100ft)', fontsize=10)
        ax.set_title('Dogleg Severity Along Wellbore', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_progress(self, rewards, filename='04_training_rewards.png'):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(rewards))
        
        ax.plot(episodes, rewards, 'b-', alpha=0.3, label='Episode Reward')
        
        window = min(50, len(rewards) // 10)
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 'r-', 
                   linewidth=2, label=f'Moving Average ({window} episodes)')
        
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Total Reward', fontsize=10)
        ax.set_title('PPO Training Progress', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reservoir_properties(self, reservoir, trajectory, filename='05_reservoir_properties.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        porosity_slice = reservoir.porosity_field[:, :, 15]
        perm_slice = np.log10(reservoir.permeability_field[:, :, 15] + 1)
        
        im1 = ax1.imshow(porosity_slice.T, origin='lower', cmap='viridis', aspect='auto')
        ax1.set_title('Porosity Field (Layer 15)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('X Grid', fontsize=10)
        ax1.set_ylabel('Y Grid', fontsize=10)
        plt.colorbar(im1, ax=ax1, label='Porosity')
        
        if trajectory and len(trajectory) > 0:
            traj_points = [(int(p['N'] / 50), int(p['E'] / 50)) for p in trajectory 
                          if 0 <= int(p['N']/50) < 100 and 0 <= int(p['E']/50) < 100]
            if traj_points:
                N_indices, E_indices = zip(*traj_points)
                ax1.plot(N_indices, E_indices, 'r-', linewidth=2, alpha=0.7)
        
        im2 = ax2.imshow(perm_slice.T, origin='lower', cmap='plasma', aspect='auto')
        ax2.set_title('Log Permeability Field (Layer 15)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('X Grid', fontsize=10)
        ax2.set_ylabel('Y Grid', fontsize=10)
        plt.colorbar(im2, ax=ax2, label='Log10(Permeability)')
        
        if trajectory and len(trajectory) > 0:
            traj_points2 = [(int(p['N'] / 50), int(p['E'] / 50)) for p in trajectory 
                           if 0 <= int(p['N']/50) < 100 and 0 <= int(p['E']/50) < 100]
            if traj_points2:
                N_indices2, E_indices2 = zip(*traj_points2)
                ax2.plot(N_indices2, E_indices2, 'r-', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_torque_drag(self, MD, torque, drag, filename='06_torque_drag.png'):
        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(MD, torque, 'b-', linewidth=2)
        ax1.set_xlabel('Measured Depth (ft)', fontsize=10)
        ax1.set_ylabel('Torque (ft-lbf)', fontsize=10)
        ax1.set_title('Torque vs Measured Depth', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(MD, drag, 'r-', linewidth=2)
        ax2.set_xlabel('Measured Depth (ft)', fontsize=10)
        ax2.set_ylabel('Drag Force (lbf)', fontsize=10)
        ax2.set_title('Drag Force vs Measured Depth', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(self, ai_trajectory, conventional_trajectory, 
                       filename='07_trajectory_comparison.png'):
        fig = plt.figure(figsize=(14, 10))
        
        ax1 = fig.add_subplot(221, projection='3d')
        N_ai = [p['N'] for p in ai_trajectory]
        E_ai = [p['E'] for p in ai_trajectory]
        TVD_ai = [p['TVD'] for p in ai_trajectory]
        
        N_conv = [p['N'] for p in conventional_trajectory]
        E_conv = [p['E'] for p in conventional_trajectory]
        TVD_conv = [p['TVD'] for p in conventional_trajectory]
        
        ax1.plot(N_ai, E_ai, TVD_ai, 'b-', linewidth=2, label='AI-Optimized')
        ax1.plot(N_conv, E_conv, TVD_conv, 'r--', linewidth=2, label='Conventional')
        ax1.set_xlabel('North (ft)')
        ax1.set_ylabel('East (ft)')
        ax1.set_zlabel('TVD (ft)')
        ax1.set_title('3D Trajectory Comparison', fontweight='bold')
        ax1.invert_zaxis()
        ax1.legend()
        
        ax2 = fig.add_subplot(222)
        MD_ai = [p['MD'] for p in ai_trajectory]
        DLS_ai = [p['DLS'] for p in ai_trajectory]
        MD_conv = [p['MD'] for p in conventional_trajectory]
        DLS_conv = [p['DLS'] for p in conventional_trajectory]
        
        ax2.plot(MD_ai, DLS_ai, 'b-', linewidth=2, label='AI-Optimized')
        ax2.plot(MD_conv, DLS_conv, 'r--', linewidth=2, label='Conventional')
        ax2.set_xlabel('Measured Depth (ft)')
        ax2.set_ylabel('DLS (°/100ft)')
        ax2.set_title('Dogleg Severity Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(223)
        metrics = ['Total MD', 'Max DLS', 'Avg DLS']
        ai_vals = [MD_ai[-1], max(DLS_ai), np.mean(DLS_ai)]
        conv_vals = [MD_conv[-1], max(DLS_conv), np.mean(DLS_conv)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, ai_vals, width, label='AI-Optimized', color='blue', alpha=0.7)
        ax3.bar(x + width/2, conv_vals, width, label='Conventional', color='red', alpha=0.7)
        ax3.set_ylabel('Value')
        ax3.set_title('Performance Metrics Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = fig.add_subplot(224)
        I_ai = [p['inclination'] for p in ai_trajectory]
        I_conv = [p['inclination'] for p in conventional_trajectory]
        
        ax4.plot(MD_ai, I_ai, 'b-', linewidth=2, label='AI-Optimized')
        ax4.plot(MD_conv, I_conv, 'r--', linewidth=2, label='Conventional')
        ax4.set_xlabel('Measured Depth (ft)')
        ax4.set_ylabel('Inclination (degrees)')
        ax4.set_title('Inclination Comparison', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
