import numpy as np
from scipy.ndimage import gaussian_filter

class ReservoirModel:
    def __init__(self, grid_size=(100, 100, 30), cell_size=(50, 50, 10)):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.porosity_field = None
        self.permeability_field = None
        self.pressure_profile = None
        
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
        
        self.pressure_profile = {
            'pore_pressure_grad': pore_pressure_grad,
            'frac_gradient': frac_gradient
        }
        
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
            'pore_pressure_grad': self.pressure_profile['pore_pressure_grad'],
            'frac_gradient': self.pressure_profile['frac_gradient']
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
