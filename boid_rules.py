import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.spatial.transform import Rotation
from numpy.linalg import norm

class Boids():
    """This class will be a group of Bird-oid objects (Boids), 
    consisting of both preditors and preys"""
    
    def __init__(self, num_boids=10, n_dim=2, timestep=1):
        """Only the number of boids and dimensions are needed. Currently only n_dim=2 
        is supported. Leave unchanged for more natural starting conditions."""
        self.positions = np.random.uniform(-20,20, n_dim*num_boids).reshape(num_boids, n_dim)
        self.velocities = np.random.normal(1, 0.05, num_boids).reshape(-1, 1)
        self.angles = np.random.uniform(0, 2*np.pi, num_boids).reshape(-1, 1)
        self.angles = np.full(num_boids, 2*np.pi/4).reshape(-1, 1)
        self.num_boids = num_boids
        self.n_dim = n_dim
        self.timestep = timestep
        
    def add_noise_velocity(self, sd):
        return np.random.normal(0, sd, self.num_boids).reshape(-1, 1)
    
    def polar_to_vec(self, velocities, angles):   
        return velocities * np.hstack((np.sin(angles), np.cos(angles)))
    
    def next_position(self, positions, velocities, angles):
        return positions + self.timestep * self.polar_to_vec(velocities, angles)
       
    def get_local_flock_idx(self, boid_idx, radius):
        """Within a flock of birds, we distinguish local groups. These are the 
        boids that respond to each other in terms of movement. This function
        finds the boid indices of the group a specific boid responds to.
        By convention, a boid is not in own group preventing interacting with itself"""
        
        flock_idx = norm(self.positions - self.positions[boid_idx,:], axis=1) < radius 
        flock_idx[boid_idx] = False
        patat = np.where(flock_idx)[0]
        return patat
        
    def flock_alignment(self, idx, alpha, idx_group):
        """Steer towards the average heading of local flockmates with rate alpha
        Input: boid, flock of boid, effect strength
        Output: Size of velocity nudge towards average flock direction"""
        boid_angle = self.angles[idx]
        group_mean_angle = np.mean(self.angles[idx_group])
        
        angle_effect = alpha * (boid_angle - group_mean_angle)
        return angle_effect
        
    def flock_separation(self, idx, alpha, idx_group):
        """Steer to avoid crowding local flockmates. If no flockmates are within the specified
        radius, the radius is increased"""
        group_pos = self.positions[idx_group, :]
        group_vel = self.velocities[idx_group]
        group_ang = self.angles[idx_group]
        
        boid_pos = self.positions[idx, :]
        boid_vel = self.velocities[idx]
        boid_ang = self.angles[idx]
        
        # Simple Euler forward approximation of next position
        group_pos_update = self.next_position(group_pos, group_vel, group_ang)
        boid_pos_update = self.next_position(boid_pos, boid_vel, boid_ang)
        
        repulsion = norm(boid_pos_update - group_pos_update, axis=1) ** (-2)
        repulsion /= np.sum(np.abs(repulsion))
        angle_change = np.dot(repulsion, group_ang)
        
        return alpha * angle_change
        
    def flock_cohesion(self, idx, alpha, idx_group=[], centerpoint=None):
        """steer to move towards the average position (center of mass) of local flockmates
        currently, the center point is still. 
        We can choose to first move the center point one timestep forward"""
        if len(idx_group)>0:
            group_pos = self.next_position(self.positions[idx_group,:], 
                                           self.velocities[idx_group], 
                                           self.angles[idx_group])
            group_pos = np.mean(group_pos, axis=0)
        else:
            group_pos = centerpoint
            
        position = self.positions[idx,:]
        velocity = self.velocities[idx]
        angle = self.angles[idx]
        
        position_upd = self.next_position(position, velocity, angle)
        
        boid_dir = position_upd - position
        boid_to_center = group_pos - position
            
        dotproduct = np.dot(boid_dir, boid_to_center)
        magnitude_prod = norm(boid_dir) * norm(boid_to_center)
        angle = np.arccos(dotproduct/magnitude_prod)
        
        sign = np.sign(np.linalg.det((boid_dir, boid_to_center)))
        
        return alpha * sign * angle
        
    def flock_update(self, radius, c_rate, a_rate, s_rate, o_rate, center=np.array([0,0])):
        """Determines the flock for each boid, then computes the effects of alignment, cohesion
        and separation. Optionally control for speed by using the tanh contraction function."""
        store_angles = np.full(self.num_boids, 0, dtype=np.float64).reshape(-1, 1)
        
        for idx in range(self.num_boids):
            radius_local = radius
            
            flock_idx = False
            while not np.any(flock_idx):
                flock_idx = self.get_local_flock_idx(idx, radius_local)
                radius_local *= 1.1
                
            store_angles[idx] -= self.flock_separation(idx, alpha=s_rate, idx_group=flock_idx)
            store_angles[idx] -= self.flock_alignment(idx, alpha=a_rate, idx_group=flock_idx)
            store_angles[idx] -= self.flock_cohesion(idx, alpha=c_rate, idx_group=flock_idx)
            store_angles[idx] -= self.flock_cohesion(idx, alpha=o_rate, centerpoint=center)
            store_angles += self.add_noise_velocity(0.000)
            
        self.angles += store_angles
        self.positions = self.next_position(self.positions, self.velocities, self.angles)
        
        
            
# This part is for checking computation speed
if __name__ == "__main__":    
    t = time.perf_counter()
    flock = Boids(num_boids=10, n_dim=2, timestep=1)
    for i in range(2):
        print(i)
        flock.flock_update(radius=10, c_rate=0.05, a_rate=0.05, s_rate=0.01, o_rate=0.01)
        plt.quiver(flock.positions[:,1], flock.positions[:,0],
                    flock.velocities*np.cos(flock.angles), flock.velocities*np.sin(flock.angles))
    #print(time.perf_counter() - t)
    
    
    # def contract_velocities():
    #     # if contraction:
    #     #     velocities = norm(self.velocities, axis=1).reshape(-1, 1)
    #     #     velocities_cor = 1 + np.tanh(velocities-1)
    #     #     self.velocities /= norm(self.velocities, axis=1).reshape(-1, 1)
    #     #     self.velocities *= velocities_cor
    #     # else:
    #     #     self.velocities /= norm(self.velocities, axis=1).reshape(-1, 1)
    #     pass