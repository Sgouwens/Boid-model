import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.spatial.transform import Rotation
from numpy.linalg import norm

class Boids():
    """This class will be a group of Boids, consisting of both preditors and preys"""
    
    def __init__(self, num_boids=10, n_dim=2, timestep=1):
        
        # Testing starting conditions
        # self.positions = np.hstack((np.linspace(-50,50, num_boids).reshape(-1,1),
        #                             np.zeros(num_boids).reshape(-1,1)))
        # self.velocities = np.hstack((np.zeros(num_boids).reshape(-1,1),
        #                             np.ones(num_boids).reshape(-1,1)))
        
        # Random starting conditions
        self.positions = np.random.uniform(-30,30, n_dim*num_boids).reshape(num_boids, n_dim)
        self.velocities = np.random.normal(0, 0.1, n_dim*num_boids).reshape(num_boids, n_dim)
        self.velocities /= norm(self.velocities, axis=0, keepdims=True)
        
        self.num_boids = num_boids
        self.n_dim = n_dim
        self.timestep = timestep
                
    # @staticmethod
    # def flock_mean(arr):
    #     """Computes the center of the group as well as the group's direction"""
    #     mean = np.mean(arr, axis=0)
    #     return mean
    
    def get_local_flock_idx(self, boid_idx, radius):
        """Within a flock of birds, we distinguish local groups. These are the 
        boids that respond to each other in terms of movement. This function
        finds the boid indices of the group a specific boid responds to.
        By convention, a boid is not in own group preventing interacting with itself"""
        flock_idx = norm(self.positions - self.positions[boid_idx,:], axis=1) < radius        
        flock_idx[boid_idx] = False
        
        return np.where(flock_idx)[0]
            
    @staticmethod
    def steer_away(vel_boid, vel_group, prop):
        
        vel_boid_dir = vel_boid / norm(vel_boid)
        vel_group_dir = vel_group / norm(vel_group).reshape(-1,1)
        
        displacement = (prop.reshape(-1, 1)**2) * (vel_boid_dir - vel_group_dir) # squared
        vel_boid_update = vel_boid + np.mean(displacement, axis=0)
        vel_boid_update /= norm(vel_boid_update)
        
        return vel_boid_update
    
    def flock_alignment(self, idx, alpha, idx_group):
        """Steer towards the average heading of local flockmates with rate alpha
        Input: boid, flock of boid, effect strength
        Output: Size of velocity nudge"""
        
        group_vel = self.velocities[idx_group, :]
        group_mean_dir = np.mean(group_vel, axis=0)
        group_mean_dir /= norm(group_mean_dir)
        
        boid_dir = self.velocities[idx, :] / norm(self.velocities[idx, :])
        
        boid_dir_updated = boid_dir + alpha * group_mean_dir   
        boid_dir_updated /= norm(boid_dir_updated)
        
        return boid_dir_updated - boid_dir
    
    def flock_separation(self, idx, alpha, idx_group):
        """Steer to avoid crowding local flockmates. If no flockmates are within the specified
        radius, the radius is increased"""
        
        # Change this such that rotation matrix is used. Current implementation can cause weird issues
        # To do this, make a new function 'find_angle'
        
        group_pos = self.positions[idx_group, :]
        group_vel = self.velocities[idx_group, :]
        
        boid_pos = self.positions[idx, :]
        boid_vel = self.velocities[idx, :]
        
        # Simple Euler forward approximation of next position
        group_pos_update = group_pos + self.timestep * group_vel
        
        group_pos_distances = norm(boid_pos - group_pos_update, axis=1)
        group_prop_repulsion = 1 / group_pos_distances
        
        boid_vel = self.steer_away(boid_vel, group_vel, group_prop_repulsion) - boid_vel
        
        return alpha*boid_vel
    
    @staticmethod
    def find_angle(vec1, vec2):
        
        clockwise = np.sign(np.linalg.det((vec1, vec2)))
        dotproduct = np.dot(vec1, vec2)
        denominator = norm(vec1) * norm(vec2)
        angle = clockwise * np.arccos(dotproduct / denominator)
        
        return angle
    
    def flock_cohesion(self, idx, alpha, idx_group=[], to_center=False):
        """steer to move towards the average position (center of mass) of local flockmates
        currently, the center point is still. 
        We can choose to first move the center point one timestep forward"""
        
        if len(idx_group)>0:
            if len(idx_group)>1:
                group_pos_timestep = self.positions[idx_group, :] + self.timestep * self.velocities[idx_group, :]
                group_pos = np.mean(group_pos_timestep, axis=0)
            else:
                return 0
        else:
            group_pos = np.array([0, 0])
            
        velocity = self.velocities[idx,:]
        position = self.positions[idx,:]
        boid_to_center = group_pos - position
        
        angle_new = alpha * self.find_angle(velocity, boid_to_center)
        rotation_matrix = Rotation.from_euler('z', angle_new).as_matrix()[:2, :2]
        rotated_vector = np.dot(rotation_matrix, velocity) 
        
        return rotated_vector - velocity
        
    def flock_update(self, radius, c_rate, a_rate, s_rate):
        
        store_position = np.full((self.num_boids, self.n_dim), 0, dtype=np.float64)
        store_velocity = np.full((self.num_boids, self.n_dim), 0, dtype=np.float64)
        
        # A for-loop to find the positional/speed update for each boid.
        for idx in range(self.num_boids):
            
            radius_local = radius
            flock_idx = self.get_local_flock_idx(idx, radius_local)
            
            while len(flock_idx)==0:
                radius_local *= 1.1
                flock_idx = self.get_local_flock_idx(idx, radius_local)
                 
            store_velocity[idx, :] += self.flock_separation(idx, alpha=s_rate, idx_group=flock_idx)
            store_velocity[idx, :] += self.flock_alignment(idx, alpha=a_rate, idx_group=flock_idx)
            store_velocity[idx, :] += self.flock_cohesion(idx, alpha=c_rate, idx_group=flock_idx)
            store_velocity[idx, :] += self.flock_cohesion(idx, alpha=0.005)
        
        self.velocities += store_velocity
        
        # velocities = norm(self.velocities, axis=1).reshape(-1, 1)
        # velocities_cor = 1 + np.tanh(velocities-1)
        
        self.velocities /= norm(self.velocities, axis=1).reshape(-1, 1)
        # self.velocities *= velocities_cor
        
        self.positions += store_position + 1*self.timestep * self.velocities
        
# This part is for checking computation speed
t = time.perf_counter()
if __name__ == "__main__":    
    flock = Boids(num_boids=60, n_dim=2, timestep=1)
    for i in range(400):
        flock.flock_update(radius=10, c_rate=0.01, a_rate=0.01, s_rate=0.01)
print(time.perf_counter() - t)