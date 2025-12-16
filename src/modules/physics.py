import numpy as np

class TrajectoryPhysics:
    def __init__(self, survey_interval=30):
        self.interval = survey_interval
    
    def minimum_curvature(self, I1, A1, I2, A2, dMD):
        # Ensure angles are valid
        I1 = np.clip(I1, 0, 90)
        I2 = np.clip(I2, 0, 90)
        A1 = A1 % 360
        A2 = A2 % 360
        
        I1_rad, I2_rad = np.radians(I1), np.radians(I2)
        A1_rad, A2_rad = np.radians(A1), np.radians(A2)
        
        # Clip to prevent numerical errors in arccos
        cos_beta = np.cos(I1_rad) * np.cos(I2_rad) + \
                   np.sin(I1_rad) * np.sin(I2_rad) * np.cos(A2_rad - A1_rad)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)
        
        if beta < 1e-6:
            RF = 1.0
        else:
            RF = (2 / beta) * np.tan(beta / 2)
        
        dN = (dMD / 2) * (np.sin(I1_rad) * np.cos(A1_rad) + 
                          np.sin(I2_rad) * np.cos(A2_rad)) * RF
        dE = (dMD / 2) * (np.sin(I1_rad) * np.sin(A1_rad) + 
                          np.sin(I2_rad) * np.sin(A2_rad)) * RF
        dTVD = (dMD / 2) * (np.cos(I1_rad) + np.cos(I2_rad)) * RF
        
        # Validate outputs
        if np.isnan(dN) or np.isnan(dE) or np.isnan(dTVD):
            return 0.0, 0.0, 0.0
        
        return dN, dE, dTVD
    
    def dogleg_severity(self, I1, A1, I2, A2, dMD):
        I1_rad, I2_rad = np.radians(I1), np.radians(I2)
        A1_rad, A2_rad = np.radians(A1), np.radians(A2)
        
        cos_beta = (np.cos(I1_rad) * np.cos(I2_rad) + 
                   np.sin(I1_rad) * np.sin(I2_rad) * np.cos(A2_rad - A1_rad))
        cos_beta = np.clip(cos_beta, -1, 1)
        beta = np.arccos(cos_beta)
        
        DLS = np.degrees(beta) * (100 / dMD) if dMD > 0 else 0
        return DLS
    
    def torque_drag(self, trajectory, friction_factor, mud_weight):
        total_torque = 0
        total_drag = 0
        pipe_weight_per_ft = 19.5
        pipe_od = 5.0 / 12
        
        for i in range(1, len(trajectory)):
            I = trajectory[i]['inclination']
            dL = trajectory[i]['MD'] - trajectory[i-1]['MD']
            DLS = trajectory[i].get('DLS', 0)
            
            I_rad = np.radians(I)
            normal_force = pipe_weight_per_ft * np.sin(I_rad) * dL
            drag_force = friction_factor * normal_force
            
            total_drag += drag_force
            total_torque += friction_factor * normal_force * pipe_od
        
        return total_torque, total_drag
    
    def wellbore_stability(self, TVD, mud_weight, pore_pressure_grad, frac_gradient):
        pore_pressure = pore_pressure_grad * TVD
        frac_pressure = frac_gradient * TVD
        
        mud_pressure = mud_weight * 0.052 * TVD
        
        safe_margin = 0.5 * 0.052 * TVD
        
        if mud_pressure < pore_pressure + safe_margin:
            return False, "Underbalanced"
        if mud_pressure > frac_pressure - safe_margin:
            return False, "Overbalanced"
        
        return True, "Stable"
    
    def calculate_trajectory(self, KOP, target, BUR, max_inclination, azimuth):
        trajectory = []
        
        MD = KOP
        TVD = KOP
        N, E = 0, 0
        I = 0
        A = azimuth
        
        trajectory.append({
            'MD': MD, 'TVD': TVD, 'N': N, 'E': E, 
            'inclination': I, 'azimuth': A, 'DLS': 0
        })
        
        target_N = target['N']
        target_E = target['E']
        target_TVD = target['TVD']
        
        while TVD < target_TVD - 50:
            delta_N = target_N - N
            delta_E = target_E - E
            delta_TVD = target_TVD - TVD
            horiz_dist = np.sqrt(delta_N**2 + delta_E**2)
            
            required_A = np.degrees(np.arctan2(delta_E, delta_N)) % 360
            A_error = required_A - A
            if A_error > 180:
                A_error -= 360
            elif A_error < -180:
                A_error += 360
            
            A_new = A + np.clip(A_error, -5, 5)
            
            required_I = np.degrees(np.arctan2(horiz_dist, delta_TVD))
            required_I = min(required_I, max_inclination)
            
            if horiz_dist < 500:
                I_new = max(I - BUR * (self.interval / 100), 0)
            elif I < required_I:
                I_new = min(I + BUR * (self.interval / 100), max_inclination)
            else:
                I_new = I
            
            dN, dE, dTVD = self.minimum_curvature(I, A, I_new, A_new, self.interval)
            
            MD += self.interval
            TVD += dTVD
            N += dN
            E += dE
            
            DLS = self.dogleg_severity(I, A, I_new, A_new, self.interval)
            
            trajectory.append({
                'MD': MD, 'TVD': TVD, 'N': N, 'E': E,
                'inclination': I_new, 'azimuth': A_new, 'DLS': DLS
            })
            
            I = I_new
            A = A_new
            
            dist_to_target = np.sqrt(delta_N**2 + delta_E**2 + delta_TVD**2)
            if dist_to_target < 150:
                break
        
        return trajectory


class WellboreMechanics:
    @staticmethod
    def critical_slide_angle(friction_factor):
        return np.degrees(np.arctan(1 / friction_factor))
    
    @staticmethod
    def mud_weight_window(TVD, pore_pressure_grad, frac_gradient, safety_margin=0.5):
        MW_min = pore_pressure_grad / 0.052 + safety_margin
        MW_max = frac_gradient / 0.052 - safety_margin
        return MW_min, MW_max
    
    @staticmethod
    def build_up_rate_from_radius(radius_of_curvature):
        return 18000 / (np.pi * radius_of_curvature)
