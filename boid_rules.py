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
        #self.angles = np.full(num_boids, 2*np.pi/2).reshape(-1, 1)
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
        
    def flock_alignment(self, boid_x, boid_v, boid_a, group_x, group_v, group_a, alpha):
        """Steer towards the average heading of local flockmates with rate alpha
        Input: boid, flock of boid, effect strength
        Output: Size of velocity nudge towards average flock direction"""
        
        group_mean_angle = np.mean(group_a)
        angle_effect = alpha * (boid_a - group_mean_angle)
        return angle_effect
        
    def flock_separation(self, boid_x, boid_v, boid_a, group_x, group_v, group_a, alpha):
        """Steer to avoid crowding local flockmates. If no flockmates are within the specified
        radius, the radius is increased"""
                
        # Simple Euler forward approximation of next position
        group_pos_update = self.next_position(group_x, group_v, group_a)
        boid_pos_update = self.next_position(boid_x, boid_v, boid_a)
        
        repulsion = norm(boid_pos_update - group_pos_update, axis=1) ** (-2)
        repulsion /= np.sum(np.abs(repulsion))
        
        vector_boid = self.polar_to_vec(boid_v, boid_a)
        vector_group = self.polar_to_vec(group_v, group_a)
        
        vector_mean = np.mean(vector_group, axis=0) 
        vector_diff = vector_mean - vector_boid
        
        angles_change = np.arctan2(vector_diff[1], vector_diff[0])
        
        return alpha * angles_change
        
    def flock_cohesion2(self, idx, alpha, idx_group=[], centerpoint=None):
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

    def flock_cohesion(self, boid_x, boid_v, boid_a, group_x, group_v, group_a, alpha, centerpoint=None):
        """steer to move towards the average position (center of mass) of local flockmates
        currently, the center point is still. 
        We can choose to first move the center point one timestep forward"""
        if centerpoint is None:
            if len(group_a)>0:
                group_pos = self.next_position(group_x, group_v, group_a)
                group_pos = np.mean(group_pos, axis=0)
            else:
                return 0
        else:
            group_pos = centerpoint
            
        position_upd = self.next_position(boid_x, boid_v, boid_a)
        
        boid_dir = position_upd - boid_x
        boid_to_center = group_pos - boid_x
            
        dotproduct = np.dot(boid_dir, boid_to_center)
        magnitude_prod = norm(boid_dir) * norm(boid_to_center)
        boid_a = np.arccos(dotproduct/magnitude_prod)
        
        sign = np.sign(np.linalg.det((boid_dir, boid_to_center)))
        
        return alpha * sign * boid_a
        
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
                
            boid = (self.positions[idx, :], 
                    self.velocities[idx], 
                    self.angles[idx])
            
            group = (self.positions[flock_idx, :], 
                     self.velocities[flock_idx], 
                     self.angles[flock_idx])
                
            store_angles[idx] -= self.flock_separation(*boid, *group, alpha=s_rate)
            store_angles[idx] -= self.flock_alignment(*boid, *group, alpha=a_rate)
            store_angles[idx] -= self.flock_cohesion(*boid, *group, alpha=c_rate)
            store_angles[idx] -= self.flock_cohesion(*boid, *group, alpha=o_rate, centerpoint=center)
            store_angles      -= self.add_noise_velocity(0.001)
            
        self.angles += store_angles
        self.angles = self.angles % (2*np.pi)
        self.positions = self.next_position(self.positions, self.velocities, self.angles)
        
# This part is for checking computation speed
if __name__ == "__main__":    
    t = time.perf_counter()
    flock = Boids(num_boids=200, n_dim=2, timestep=1)
    for i in range(100):
        flock.flock_update(radius=10, c_rate=0.0, a_rate=0.00, s_rate=0.1, o_rate=0.00)
        plt.quiver(flock.positions[:,1], flock.positions[:,0],
                    flock.velocities*np.cos(flock.angles), flock.velocities*np.sin(flock.angles))
    print(time.perf_counter() - t)