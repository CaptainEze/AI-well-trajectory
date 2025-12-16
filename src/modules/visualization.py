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
    
    def plot_reservoir_properties_4panel(self, reservoir, filename='08_reservoir_properties_4panel.png'):
        fig = plt.figure(figsize=(18, 16))
        
        nx, ny, nz = reservoir.grid_size
        sample_points = 18000
        np.random.seed(42)
        
        x_samples = np.random.randint(0, nx, sample_points)
        y_samples = np.random.randint(0, ny, sample_points)
        z_samples = np.random.randint(0, nz, sample_points)
        
        x_coords = x_samples * 50
        y_coords = y_samples * 50
        z_coords = z_samples * 100
        
        porosity_samples = reservoir.porosity_field[x_samples, y_samples, z_samples]
        perm_samples = reservoir.permeability_field[x_samples, y_samples, z_samples]
        
        ax1 = fig.add_subplot(221, projection='3d')
        sc1 = ax1.scatter(x_coords, y_coords, z_coords, 
                         c=porosity_samples, cmap='Greens', 
                         s=6, alpha=0.3, vmin=0.05, vmax=0.35)
        ax1.set_xlabel('North (ft)', fontsize=10)
        ax1.set_ylabel('East (ft)', fontsize=10)
        ax1.set_zlabel('TVD (ft)', fontsize=10)
        ax1.set_title('Porosity Distribution', fontsize=12, fontweight='bold')
        ax1.invert_zaxis()
        cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.1, shrink=0.6)
        cbar1.set_label('Porosity', fontsize=9)
        
        ax2 = fig.add_subplot(222, projection='3d')
        log_perm = np.log10(perm_samples + 1)
        sc2 = ax2.scatter(x_coords, y_coords, z_coords,
                         c=log_perm, cmap='Reds',
                         s=6, alpha=0.3, vmin=0, vmax=3)
        ax2.set_xlabel('North (ft)', fontsize=10)
        ax2.set_ylabel('East (ft)', fontsize=10)
        ax2.set_zlabel('TVD (ft)', fontsize=10)
        ax2.set_title('Permeability Distribution', fontsize=12, fontweight='bold')
        ax2.invert_zaxis()
        cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.1, shrink=0.6)
        cbar2.set_label('Log10(Perm md)', fontsize=9)
        
        ax3 = fig.add_subplot(223, projection='3d')
        pore_pressure_samples = reservoir.pore_pressure_field[x_samples, y_samples, z_samples]
        sc3 = ax3.scatter(x_coords, y_coords, z_coords,
                         c=pore_pressure_samples, cmap='Blues',
                         s=6, alpha=0.3, vmin=0.45, vmax=0.65)
        ax3.set_xlabel('North (ft)', fontsize=10)
        ax3.set_ylabel('East (ft)', fontsize=10)
        ax3.set_zlabel('TVD (ft)', fontsize=10)
        ax3.set_title('Pore Pressure Gradient', fontsize=12, fontweight='bold')
        ax3.invert_zaxis()
        cbar3 = plt.colorbar(sc3, ax=ax3, pad=0.1, shrink=0.6)
        cbar3.set_label('PPG (psi/ft)', fontsize=9)
        
        ax4 = fig.add_subplot(224, projection='3d')
        frac_grad_samples = reservoir.frac_gradient_field[x_samples, y_samples, z_samples]
        sc4 = ax4.scatter(x_coords, y_coords, z_coords,
                         c=frac_grad_samples, cmap='Purples',
                         s=6, alpha=0.3, vmin=0.75, vmax=1.05)
        ax4.set_xlabel('North (ft)', fontsize=10)
        ax4.set_ylabel('East (ft)', fontsize=10)
        ax4.set_zlabel('TVD (ft)', fontsize=10)
        ax4.set_title('Fracture Gradient', fontsize=12, fontweight='bold')
        ax4.invert_zaxis()
        cbar4 = plt.colorbar(sc4, ax=ax4, pad=0.1, shrink=0.6)
        cbar4.set_label('Frac Grad (psi/ft)', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reservoir_combined_with_well(self, reservoir, trajectory, filename='09_reservoir_combined_well.png'):
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        nx, ny, nz = reservoir.grid_size
        sample_points = 20000
        np.random.seed(42)
        
        x_samples = np.random.randint(0, nx, sample_points)
        y_samples = np.random.randint(0, ny, sample_points)
        z_samples = np.random.randint(0, nz, sample_points)
        
        x_coords = x_samples * 50
        y_coords = y_samples * 50
        z_coords = z_samples * 100
        
        porosity_samples = reservoir.porosity_field[x_samples, y_samples, z_samples]
        perm_samples = reservoir.permeability_field[x_samples, y_samples, z_samples]
        
        points_per_property = sample_points // 4
        
        por_mask = slice(0, points_per_property)
        perm_mask = slice(points_per_property, 2*points_per_property)
        pore_mask = slice(2*points_per_property, 3*points_per_property)
        frac_mask = slice(3*points_per_property, sample_points)
        
        ax.scatter(x_coords[por_mask], y_coords[por_mask], z_coords[por_mask],
                  c=porosity_samples[por_mask], cmap='Greens',
                  s=8, alpha=0.25, vmin=0.05, vmax=0.35, label='Porosity')
        
        log_perm = np.log10(perm_samples[perm_mask] + 1)
        ax.scatter(x_coords[perm_mask], y_coords[perm_mask], z_coords[perm_mask],
                  c=log_perm, cmap='Reds',
                  s=8, alpha=0.25, vmin=0, vmax=3, label='Permeability')
        
        depth_factor = z_coords[pore_mask] / np.max(z_coords)
        ax.scatter(x_coords[pore_mask], y_coords[pore_mask], z_coords[pore_mask],
                  c=depth_factor, cmap='Blues',
                  s=8, alpha=0.25, label='Pore Pressure')
        
        depth_factor_frac = z_coords[frac_mask] / np.max(z_coords)
        ax.scatter(x_coords[frac_mask], y_coords[frac_mask], z_coords[frac_mask],
                  c=depth_factor_frac, cmap='Purples',
                  s=8, alpha=0.25, label='Frac Gradient')
        
        if trajectory and len(trajectory) > 0:
            N = [p['N'] for p in trajectory]
            E = [p['E'] for p in trajectory]
            TVD = [p['TVD'] for p in trajectory]
            ax.plot(N, E, TVD, 'yellow', linewidth=4, label='Well Path', zorder=10)
        
        ax.set_xlabel('North (ft)', fontsize=11)
        ax.set_ylabel('East (ft)', fontsize=11)
        ax.set_zlabel('TVD (ft)', fontsize=11)
        ax.set_title('Combined Reservoir Properties with Well Trajectory', fontsize=14, fontweight='bold')
        ax.invert_zaxis()
        ax.legend(loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reservoir_contour_slices(self, reservoir, trajectory, filename='11_reservoir_contour_slices.png'):
        fig = plt.figure(figsize=(20, 12))
        
        nx, ny, nz = reservoir.grid_size
        depth_slices = [29, 59, 89, 119, 149]
        
        for idx, k in enumerate(depth_slices):
            ax = fig.add_subplot(2, 5, idx + 1)
            
            porosity_slice = reservoir.porosity_field[:, :, k]
            x_coords = np.arange(nx) * 50
            y_coords = np.arange(ny) * 50
            X, Y = np.meshgrid(x_coords, y_coords)
            
            contour = ax.contourf(X, Y, porosity_slice.T, levels=15, cmap='Greens')
            ax.set_xlabel('North (ft)', fontsize=9)
            ax.set_ylabel('East (ft)', fontsize=9)
            ax.set_title(f'Porosity @ TVD={k*100}ft', fontsize=10, fontweight='bold')
            ax.set_aspect('equal')
            
            if trajectory and len(trajectory) > 0:
                traj_at_depth = [p for p in trajectory if abs(p['TVD'] - k*10) < 20]
                if traj_at_depth:
                    N_vals = [p['N'] for p in traj_at_depth]
                    E_vals = [p['E'] for p in traj_at_depth]
                    ax.scatter(N_vals, E_vals, c='red', s=20, marker='o', edgecolors='black', linewidths=0.5, zorder=5)
            
            plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        
        for idx, k in enumerate(depth_slices):
            ax = fig.add_subplot(2, 5, idx + 6)
            
            perm_slice = reservoir.permeability_field[:, :, k]
            log_perm_slice = np.log10(perm_slice.T + 1)
            
            contour = ax.contourf(X, Y, log_perm_slice, levels=15, cmap='Reds')
            ax.set_xlabel('North (ft)', fontsize=9)
            ax.set_ylabel('East (ft)', fontsize=9)
            ax.set_title(f'Permeability @ TVD={k*100}ft', fontsize=10, fontweight='bold')
            ax.set_aspect('equal')
            
            if trajectory and len(trajectory) > 0:
                traj_at_depth = [p for p in trajectory if abs(p['TVD'] - k*100) < 200]
                if traj_at_depth:
                    N_vals = [p['N'] for p in traj_at_depth]
                    E_vals = [p['E'] for p in traj_at_depth]
                    ax.scatter(N_vals, E_vals, c='yellow', s=20, marker='o', edgecolors='black', linewidths=0.5, zorder=5)
            
            plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_3d_reservoir_with_trajectory(self, reservoir, trajectory, filename='10_3d_reservoir_trajectory.png'):
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = fig.add_subplot(221, projection='3d')
        
        nx, ny, nz = reservoir.grid_size
        sample_points = 2000
        np.random.seed(42)
        
        x_samples = np.random.randint(0, nx, sample_points)
        y_samples = np.random.randint(0, ny, sample_points)
        z_samples = np.random.randint(0, nz, sample_points)
        
        porosity_samples = reservoir.porosity_field[x_samples, y_samples, z_samples]
        
        x_coords = x_samples * 50
        y_coords = y_samples * 50
        z_coords = z_samples * 100
        
        scatter = ax1.scatter(x_coords, y_coords, z_coords, 
                             c=porosity_samples, cmap='viridis', 
                             s=1, alpha=0.3, vmin=0.05, vmax=0.35)
        
        if trajectory and len(trajectory) > 0:
            N = [p['N'] for p in trajectory]
            E = [p['E'] for p in trajectory]
            TVD = [p['TVD'] for p in trajectory]
            ax1.plot(N, E, TVD, 'yellow', linewidth=6, label='Well Path', alpha=0.9, zorder=10)
            ax1.scatter(N[::10], E[::10], TVD[::10], c='red', s=50, edgecolors='black', linewidths=1, zorder=11)
        
        ax1.set_xlabel('North (ft)', fontsize=10)
        ax1.set_ylabel('East (ft)', fontsize=10)
        ax1.set_zlabel('TVD (ft)', fontsize=10)
        ax1.set_title('3D Reservoir Porosity with Well Trajectory', fontsize=11, fontweight='bold')
        ax1.invert_zaxis()
        ax1.legend()
        cbar1 = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.5)
        cbar1.set_label('Porosity', fontsize=9)
        
        ax2 = fig.add_subplot(222, projection='3d')
        
        perm_samples = reservoir.permeability_field[x_samples, y_samples, z_samples]
        log_perm = np.log10(perm_samples + 1)
        
        scatter2 = ax2.scatter(x_coords, y_coords, z_coords,
                              c=log_perm, cmap='plasma',
                              s=1, alpha=0.3, vmin=0, vmax=3)
        
        if trajectory and len(trajectory) > 0:
            ax2.plot(N, E, TVD, 'yellow', linewidth=6, label='Well Path', alpha=0.9, zorder=10)
            ax2.scatter(N[::10], E[::10], TVD[::10], c='red', s=50, edgecolors='black', linewidths=1, zorder=11)
        
        ax2.set_xlabel('North (ft)', fontsize=10)
        ax2.set_ylabel('East (ft)', fontsize=10)
        ax2.set_zlabel('TVD (ft)', fontsize=10)
        ax2.set_title('3D Reservoir Permeability with Well Trajectory', fontsize=11, fontweight='bold')
        ax2.invert_zaxis()
        ax2.legend()
        cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.5)
        cbar2.set_label('Log10(Perm)', fontsize=9)
        
        ax3 = fig.add_subplot(223)
        if trajectory and len(trajectory) > 0:
            porosity_along_path = []
            perm_along_path = []
            md_along_path = []
            
            for point in trajectory:
                props = reservoir.get_properties(point['N'], point['E'], point['TVD'])
                porosity_along_path.append(props['porosity'])
                perm_along_path.append(props['permeability'])
                md_along_path.append(point['MD'])
            
            ax3.plot(md_along_path, porosity_along_path, 'b-', linewidth=2, label='Porosity')
            ax3.set_xlabel('Measured Depth (ft)', fontsize=10)
            ax3.set_ylabel('Porosity (fraction)', fontsize=10)
            ax3.set_title('Porosity Along Well Path', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        ax4 = fig.add_subplot(224)
        if trajectory and len(trajectory) > 0:
            ax4.plot(md_along_path, perm_along_path, 'r-', linewidth=2, label='Permeability')
            ax4.set_xlabel('Measured Depth (ft)', fontsize=10)
            ax4.set_ylabel('Permeability (md)', fontsize=10)
            ax4.set_title('Permeability Along Well Path', fontsize=11, fontweight='bold')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(self, ai_trajectory, conventional_trajectory, target,
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
        
        if target:
            ax1.scatter(target['N'], target['E'], target['TVD'], c='purple', s=500, marker='X', label='Target', edgecolors='black', linewidths=3, depthshade=False, zorder=100)

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
