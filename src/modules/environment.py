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
        self.max_steps = 3000  # Increased for deeper wells
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
        delta_TVD = target_TVD - self.current_TVD
        
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
            np.clip(self.current_MD / 40000, -1, 1),
            np.clip(self.current_TVD / 20000, -1, 1),
            np.clip(self.current_I / 90, -1, 1),
            np.clip(self.current_A / 360, -1, 1),
            np.clip(self.current_N / 10000, -1, 1),
            np.clip(self.current_E / 10000, -1, 1),
            np.clip(self.trajectory[-1]['DLS'] / 10 if len(self.trajectory) > 0 else 0, -1, 1),
            np.clip(HD_current / 10000, -1, 1),
            np.clip(dist_to_target / 30000, -1, 1),
            np.clip(I_to_target / 90, -1, 1),
            np.clip(A_to_target / 360, -1, 1),
            np.clip((target_TVD - self.current_TVD) / 20000, -1, 1),
            np.clip(delta_N / 5000, -1, 1),
            np.clip(delta_E / 5000, -1, 1),
            np.clip(torque / 50000, -1, 1),
            np.clip(drag / 100000, -1, 1),
            0.5,
            np.clip(self.constraints['friction_factor'] / 0.3, -1, 1),
            0.5,
            0.5,
            np.clip(props['porosity'] / 0.35, 0, 1),
            np.clip(np.log10(props['permeability'] + 1) / 3, 0, 1),
            np.clip(props['pore_pressure_grad'] / 0.65, 0, 1),
            np.clip(props['frac_gradient'] / 1.05, 0, 1),
            np.clip(azimuth_error / 180, -1, 1),
            np.clip(self.steps / self.max_steps, 0, 1)
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        # Validate action input
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.zeros(5)
        
        # ACTION SPACE FIX: Reduced control authority for smoother trajectories
        dI = np.clip(action[0] * 3.0, -3.0, 3.0)  # Reduced from 5.0
        dA = np.clip(action[1] * 8.0, -8.0, 8.0)  # Reduced from 12.0
        
        # Use additional action dimensions for finer control
        dI_bias = np.clip(action[2] * 0.5, -0.5, 0.5)  # Reduced from 1.0
        dA_bias = np.clip(action[3] * 1.0, -1.0, 1.0)  # Reduced from 2.0
        
        dI = dI + dI_bias
        dA = dA + dA_bias
        
        # INCLINATION CONSTRAINT: Adaptive max inclination based on target depth
        target_TVD = self.target.get('TVD', 15000)
        delta_TVD = target_TVD - self.current_TVD
        
        # Limit inclination when deep vertical progress is still needed
        if delta_TVD > 2000:
            adaptive_max_inc = min(75.0, self.constraints['max_inclination'])
        elif delta_TVD > 500:
            adaptive_max_inc = min(80.0, self.constraints['max_inclination'])
        else:
            adaptive_max_inc = self.constraints['max_inclination']
        
        new_I = np.clip(self.current_I + dI, 0, adaptive_max_inc)
        new_A = (self.current_A + dA) % 360
        
        dMD = 30
        
        DLS_preview = self.physics.dogleg_severity(self.current_I, self.current_A, new_I, new_A, dMD)
        
        if DLS_preview > self.constraints['DLS_max']:
            scale_factor = self.constraints['DLS_max'] / (DLS_preview + 1e-6)
            dI_scaled = dI * scale_factor
            dA_scaled = dA * scale_factor
            new_I = np.clip(self.current_I + dI_scaled, 0, adaptive_max_inc)
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
        
        # SUCCESS CRITERIA: More generous success zone
        if horiz_dist < 300 and vert_dist < 300:
            self.done = True
            # Scale bonus by how close we actually got
            proximity_bonus = 1.0 - (dist_to_target / 500)
            reward += 500000 * proximity_bonus
        # FAILURE CONDITIONS
        elif self.current_TVD > target_TVD + 1000:  # Overshoot significantly
            self.done = True
            reward -= 100000
        elif self.steps >= self.max_steps:
            self.done = True
            # Penalty based on distance remaining
            reward -= 50000 + (dist_to_target * 50)
        elif DLS > self.constraints['DLS_max'] * 1.2:  # Allow slight violations
            reward -= 500
        elif MD_TVD_ratio > 2.5:
            self.done = True
            reward -= 50000
        # NEW: Penalty for stopping too short
        elif self.current_TVD < target_TVD - 2000 and horiz_dist < 500:
            # We're horizontally close but way too shallow - penalize stalling
            reward -= 100
        
        next_state = self._get_state()
        
        return next_state, reward, self.done, {}
    
    def _compute_reward(self, DLS):
        target_N = self.target.get('N', 4000)
        target_E = self.target.get('E', 4000)
        target_TVD = self.target.get('TVD', 15000)
        
        # Horizontal and vertical distances
        horiz_dist = np.sqrt((target_N - self.current_N)**2 + (target_E - self.current_E)**2)
        vert_dist = abs(target_TVD - self.current_TVD)
        
        # MD/TVD efficiency
        MD_TVD_ratio = self.current_MD / (self.current_TVD + 1e-6)
        
        # IMPROVED REWARD STRUCTURE
        
        # 1. Distance-based rewards (stronger vertical emphasis)
        R_horiz = -horiz_dist / 30.0  # Reduced weight
        R_vert = -vert_dist / 50.0    # INCREASED weight for vertical progress
        
        # 2. Progress rewards (track improvement)
        R_progress = 0
        if len(self.trajectory) > 1:
            prev_N = self.trajectory[-2]['N']
            prev_E = self.trajectory[-2]['E']
            prev_TVD = self.trajectory[-2]['TVD']
            prev_horiz = np.sqrt((target_N - prev_N)**2 + (target_E - prev_E)**2)
            prev_vert = abs(target_TVD - prev_TVD)
            
            horiz_improvement = prev_horiz - horiz_dist
            vert_improvement = prev_vert - vert_dist
            
            # MUCH STRONGER vertical progress reward
            R_progress = horiz_improvement * 50 + vert_improvement * 150  # Increased from 20 to 150
        
        # 3. INCLINATION MANAGEMENT REWARDS
        R_inclination = 0
        
        # Penalize high inclination when significant vertical distance remains
        if vert_dist > 2000:
            if self.current_I > 75:
                R_inclination -= (self.current_I - 75) * 5  # Strong penalty
            elif self.current_I > 60:
                R_inclination -= (self.current_I - 60) * 2  # Moderate penalty
        elif vert_dist > 500:
            if self.current_I > 80:
                R_inclination -= (self.current_I - 80) * 3
        
        # Reward appropriate inclination for remaining work
        if vert_dist > 1000 and horiz_dist > 1000:
            # Need both horizontal and vertical progress
            ideal_inc = 45 + (horiz_dist / vert_dist) * 20
            inc_error = abs(self.current_I - ideal_inc)
            if inc_error < 10:
                R_inclination += 5
        
        # 4. Quality rewards
        R_quality = 0
        
        # DLS penalty
        if DLS <= 3.0:
            R_quality += 10
        elif DLS <= 5.0:
            R_quality += 5
        elif DLS <= 8.0:
            R_quality -= 5
        else:
            R_quality -= 50
        
        # MD/TVD efficiency
        if MD_TVD_ratio > 1.3:
            R_quality -= (MD_TVD_ratio - 1.3) * 50
        elif MD_TVD_ratio < 1.25:
            R_quality += 3
        
        # 5. Target approach bonuses
        if vert_dist < 1000:
            R_quality += (1000 - vert_dist) / 50  # Increased bonus
        elif self.current_TVD > target_TVD + 500:
            R_quality -= 20  # Increased penalty for overshoot
        
        # CRITICAL: Strong penalty for horizontal closeness but shallow depth
        if horiz_dist < 500 and vert_dist > 3000:
            R_quality -= 50  # Increased from 10
        
        # Azimuth guidance (only when needed)
        if horiz_dist > 200:
            delta_N = target_N - self.current_N
            delta_E = target_E - self.current_E
            required_azimuth = np.degrees(np.arctan2(delta_E, delta_N)) % 360
            azimuth_error = abs(self.current_A - required_azimuth)
            if azimuth_error > 180:
                azimuth_error = 360 - azimuth_error
            R_quality -= min(azimuth_error / 100, 2)
        
        # Wellbore stability
        props = self.reservoir.get_properties(self.current_N, self.current_E, self.current_TVD)
        stable, _ = self.physics.wellbore_stability(
            self.current_TVD, 12.5,
            props['pore_pressure_grad'],
            props['frac_gradient']
        )
        if not stable:
            R_quality -= 10
        
        R_total = R_progress + R_horiz + R_vert + R_quality + R_inclination
        
        return R_total
    
    def get_trajectory(self):
        return self.trajectory