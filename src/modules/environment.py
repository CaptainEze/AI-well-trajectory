import numpy as np
from modules.physics import TrajectoryPhysics, WellboreMechanics
from modules.reservoir import ReservoirModel

class WellPlanningEnv:
    def __init__(self, reservoir_model, target_location, constraints=None):
        self.reservoir = reservoir_model
        self.target = target_location
        self.physics = TrajectoryPhysics(survey_interval=30)
        
        self.constraints = constraints or {
            'DLS_max': 10.0,
            'friction_factor': 0.25,
            'min_separation': 500,
            'max_inclination': 90.0
        }
        
        self.state_dim = 26
        self.action_dim = 5
        
        self.reset()
    
    def reset(self):
        self.current_MD = self.target.get('KOP', 3000)
        self.current_TVD = self.current_MD
        self.current_I = 0
        self.current_A = self.target.get('initial_azimuth', 45)
        self.current_N = 0
        self.current_E = 0
        
        self.trajectory = [{
            'MD': self.current_MD, 'TVD': self.current_TVD,
            'N': self.current_N, 'E': self.current_E,
            'inclination': self.current_I, 'azimuth': self.current_A,
            'DLS': 0
        }]
        
        self.done = False
        self.steps = 0
        self.max_steps = 1500
        self.max_inclination_reached = 0
        
        return self._get_state()
    
    def _get_state(self):
        target_N = self.target.get('N', 4000)
        target_E = self.target.get('E', 4000)
        target_TVD = self.target.get('TVD', 15000)
        
        dist_to_target = np.sqrt((target_N - self.current_N)**2 + 
                                 (target_E - self.current_E)**2 + 
                                 (target_TVD - self.current_TVD)**2)
        
        HD_current = np.sqrt(self.current_N**2 + self.current_E**2)
        
        delta_N = target_N - self.current_N
        delta_E = target_E - self.current_E
        I_to_target = np.degrees(np.arctan2(HD_current, target_TVD - self.current_TVD)) if target_TVD > self.current_TVD else 0
        A_to_target = np.degrees(np.arctan2(delta_E, delta_N)) % 360
        
        # Compute azimuth error (shortest angular distance)
        azimuth_error = A_to_target - self.current_A
        if azimuth_error > 180:
            azimuth_error -= 360
        elif azimuth_error < -180:
            azimuth_error += 360
        
        props = self.reservoir.get_properties(self.current_N, self.current_E, self.current_TVD)
        
        torque, drag = self.physics.torque_drag(self.trajectory, 
                                                 self.constraints['friction_factor'], 
                                                 12.5)
        
        state = np.array([
            self.current_MD / 30000,
            self.current_TVD / 20000,
            self.current_I / 90,
            self.current_A / 360,
            self.current_N / 50000,
            self.current_E / 50000,
            self.trajectory[-1]['DLS'] / 10 if len(self.trajectory) > 0 else 0,
            HD_current / 40000,
            dist_to_target / 50000,
            I_to_target / 90,
            A_to_target / 360,
            (target_TVD - self.current_TVD) / 20000,
            delta_N / 10000,
            delta_E / 10000,
            torque / 50000,
            drag / 100000,
            0.5,
            self.constraints['friction_factor'],
            0.5,
            0.5,
            props['porosity'],
            np.log10(props['permeability'] + 1) / 3,
            props['pore_pressure_grad'] / 1.0,
            props['frac_gradient'] / 1.2,
            azimuth_error / 180,  # Azimuth error in range [-1, 1]
            self.steps / self.max_steps  # Episode progress
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        # Validate action input
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.zeros(5)
        
        dI = np.clip(action[0] * 1.5, -1.5, 1.5)
        dA = np.clip(action[1] * 3, -3, 3)
        
        new_I = np.clip(self.current_I + dI, 0, self.constraints['max_inclination'])
        new_A = (self.current_A + dA) % 360
        
        dMD = 30
        
        DLS_preview = self.physics.dogleg_severity(self.current_I, self.current_A, new_I, new_A, dMD)
        
        if DLS_preview > self.constraints['DLS_max']:
            scale_factor = self.constraints['DLS_max'] / (DLS_preview + 1e-6)
            dI_scaled = dI * scale_factor
            dA_scaled = dA * scale_factor
            new_I = np.clip(self.current_I + dI_scaled, 0, self.constraints['max_inclination'])
            new_A = (self.current_A + dA_scaled) % 360
        
        dN, dE, dTVD = self.physics.minimum_curvature(
            self.current_I, self.current_A, new_I, new_A, dMD
        )
        
        # Validate trajectory deltas
        if np.isnan(dN) or np.isnan(dE) or np.isnan(dTVD):
            dN, dE, dTVD = 0.0, 0.0, 0.0
            DLS = 0.0
        
        self.current_MD += dMD
        self.current_TVD += dTVD
        self.current_N += dN
        self.current_E += dE
        
        DLS = self.physics.dogleg_severity(self.current_I, self.current_A, new_I, new_A, dMD)
        
        self.current_I = new_I
        self.current_A = new_A
        
        self.trajectory.append({
            'MD': self.current_MD, 'TVD': self.current_TVD,
            'N': self.current_N, 'E': self.current_E,
            'inclination': self.current_I, 'azimuth': self.current_A,
            'DLS': DLS
        })
        
        reward = self._compute_reward(DLS)
        
        target_TVD = self.target.get('TVD', 15000)
        target_N = self.target.get('N', 4000)
        target_E = self.target.get('E', 4000)
        
        dist_to_target = np.sqrt((target_N - self.current_N)**2 + 
                                 (target_E - self.current_E)**2 + 
                                 (target_TVD - self.current_TVD)**2)
        
        self.steps += 1
        self.max_inclination_reached = max(self.max_inclination_reached, self.current_I)
        
        MD_TVD_ratio = self.current_MD / (self.current_TVD + 1e-6)
        
        horiz_dist = np.sqrt((target_N - self.current_N)**2 + (target_E - self.current_E)**2)
        vert_dist = abs(target_TVD - self.current_TVD)
        
        if horiz_dist < 150 and vert_dist < 150:
            self.done = True
            reward += 500000
        elif self.current_TVD > target_TVD + 500:
            self.done = True
            reward -= 100000
        elif self.steps >= self.max_steps:
            self.done = True
            reward -= 50000 + (dist_to_target * 50)
        elif DLS > self.constraints['DLS_max']:
            reward -= 200
        elif MD_TVD_ratio > 2.5:
            self.done = True
            reward -= 50000
        
        next_state = self._get_state()
        
        return next_state, reward, self.done, {}
    
    def _compute_reward(self, DLS):
        target_N = self.target.get('N', 4000)
        target_E = self.target.get('E', 4000)
        target_TVD = self.target.get('TVD', 15000)
        
        current_dist = np.sqrt((target_N - self.current_N)**2 + 
                               (target_E - self.current_E)**2 + 
                               (target_TVD - self.current_TVD)**2)
        
        R_distance = -current_dist / 10
        
        R_progress = 0
        if len(self.trajectory) > 1:
            prev_dist = np.sqrt((target_N - self.trajectory[-2]['N'])**2 + 
                                (target_E - self.trajectory[-2]['E'])**2 + 
                                (target_TVD - self.trajectory[-2]['TVD'])**2)
            R_progress = (prev_dist - current_dist) * 2
        
        R_quality = 0
        if DLS <= 3:
            R_quality += 5
        elif DLS > 8:
            R_quality -= 50
        elif DLS > 5:
            R_quality -= 20
        
        # Strong penalty for going too deep
        if self.current_TVD > target_TVD:
            tvd_overshoot = self.current_TVD - target_TVD
            R_quality -= tvd_overshoot / 10
        
        # Azimuth error penalty - guide agent toward target direction
        delta_N = target_N - self.current_N
        delta_E = target_E - self.current_E
        required_azimuth = np.degrees(np.arctan2(delta_E, delta_N)) % 360
        azimuth_error = abs(self.current_A - required_azimuth)
        if azimuth_error > 180:
            azimuth_error = 360 - azimuth_error
        
        # Strong penalty for pointing away from target
        R_quality -= azimuth_error / 10
        
        if len(self.trajectory) > 1:
            dA = abs(self.trajectory[-1]['azimuth'] - self.trajectory[-2]['azimuth'])
            if dA > 180:
                dA = 360 - dA
            if dA > 15:
                R_quality -= 5
        
        props = self.reservoir.get_properties(self.current_N, self.current_E, self.current_TVD)
        stable, _ = self.physics.wellbore_stability(
            self.current_TVD, 12.5,
            props['pore_pressure_grad'],
            props['frac_gradient']
        )
        if not stable:
            R_quality -= 10
        
        R_total = R_progress + R_distance + R_quality
        
        return R_total
    
    def get_trajectory(self):
        return self.trajectory
