import numpy as np
from scipy.ndimage import gaussian_filter

class ReservoirModel:
    def __init__(self, grid_size=(100, 100, 150), cell_size=(50, 50, 100)):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.porosity_field = None
        self.permeability_field = None
        self.pore_pressure_field = None
        self.frac_gradient_field = None
        
    def generate_synthetic_reservoir(self, mean_porosity=0.18, std_porosity=0.05,
                                    pore_pressure_grad=0.52, frac_gradient=0.85):
        np.random.seed(42)
        
        porosity = np.random.normal(mean_porosity, std_porosity, self.grid_size)
        porosity = gaussian_filter(porosity, sigma=3)
        self.porosity_field = np.clip(porosity, 0.05, 0.35)
        
        a = np.random.normal(10, 2)
        b = np.random.normal(1, 0.5)
        noise = np.random.normal(0, 0.2, self.grid_size)
        
        self.permeability_field = 10 ** (a * self.porosity_field + b + noise)
        self.permeability_field = np.clip(self.permeability_field, 0.01, 1000)
        
        nx, ny, nz = self.grid_size
        self.pore_pressure_field = np.zeros(self.grid_size)
        self.frac_gradient_field = np.zeros(self.grid_size)
        
        for k in range(nz):
            depth = k * self.cell_size[2]
            base_pore_press = pore_pressure_grad + (depth / 15000) * 0.05
            base_frac_grad = frac_gradient + (depth / 15000) * 0.10
            
            lateral_var_pore = np.random.normal(0, 0.02, (nx, ny))
            lateral_var_frac = np.random.normal(0, 0.03, (nx, ny))
            
            self.pore_pressure_field[:, :, k] = base_pore_press + gaussian_filter(lateral_var_pore, sigma=2)
            self.frac_gradient_field[:, :, k] = base_frac_grad + gaussian_filter(lateral_var_frac, sigma=2)
        
        self.pore_pressure_field = np.clip(self.pore_pressure_field, 0.45, 0.65)
        self.frac_gradient_field = np.clip(self.frac_gradient_field, 0.75, 1.05)
        
        return self
    
    def get_properties(self, x, y, z):
        nx, ny, nz = self.grid_size
        cell_x, cell_y, cell_z = self.cell_size
        
        i = int(np.clip(x / cell_x, 0, nx - 1))
        j = int(np.clip(y / cell_y, 0, ny - 1))
        k = int(np.clip(z / cell_z, 0, nz - 1))
        
        return {
            'porosity': float(self.porosity_field[i, j, k]),
            'permeability': float(self.permeability_field[i, j, k]),
            'pore_pressure_grad': float(self.pore_pressure_field[i, j, k]),
            'frac_gradient': float(self.frac_gradient_field[i, j, k])
        }
    
    def calculate_productivity(self, trajectory_segment, reservoir_thickness=100):
        if len(trajectory_segment) == 0:
            return 0.0
        
        total_productivity = 0.0
        
        for point in trajectory_segment:
            props = self.get_properties(point['N'], point['E'], point['TVD'])
            phi = props['porosity']
            k = props['permeability']
            
            productivity_factor = np.sqrt(phi * k)
            total_productivity += productivity_factor
        
        L_reservoir = len(trajectory_segment) * 30
        productivity_index = (L_reservoir / reservoir_thickness) * total_productivity
        
        return productivity_index
    
    def export_to_csv(self, filename):
        import pandas as pd
        data = []
        nx, ny, nz = self.grid_size
        cell_x, cell_y, cell_z = self.cell_size
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    data.append({
                        'N': i * cell_x,
                        'E': j * cell_y,
                        'TVD': k * cell_z,
                        'porosity': self.porosity_field[i, j, k],
                        'permeability': self.permeability_field[i, j, k],
                        'pore_pressure_grad': self.pore_pressure_field[i, j, k],
                        'frac_gradient': self.frac_gradient_field[i, j, k]
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return df


class SyntheticDataGenerator:
    def __init__(self, reservoir_model):
        self.reservoir = reservoir_model
    
    def generate_training_trajectories(self, n_trajectories=1000):
        trajectories = []
        
        for i in range(n_trajectories):
            KOP = np.random.uniform(2000, 6000)
            target_TVD = np.random.uniform(KOP + 5000, KOP + 15000)
            target_N = np.random.uniform(1000, 5000)
            target_E = np.random.uniform(1000, 5000)
            
            BUR = np.random.uniform(1.5, 8.0)
            max_inc = np.random.uniform(45, 88)
            azimuth = np.random.uniform(0, 360)
            
            trajectories.append({
                'id': i,
                'KOP': KOP,
                'target': {'TVD': target_TVD, 'N': target_N, 'E': target_E},
                'BUR': BUR,
                'max_inclination': max_inc,
                'azimuth': azimuth
            })
        
        return trajectories
