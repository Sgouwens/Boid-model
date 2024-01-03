import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

# It appears self.timestep is forgotten in some situations update rules

class Boids():
    """This class will be a group of Boids, consisting of both preditors and preys"""
    
    def __init__(self, num_boids=10, n_dim=2, timestep=1):
        self.positions = np.random.uniform(-50,50, n_dim*num_boids).reshape(num_boids, n_dim)
        self.velocities = np.random.normal(0, 5, n_dim*num_boids).reshape(num_boids, n_dim)
        self.velocities /= np.linalg.norm(self.velocities, axis=0, keepdims=True)
        
        self.num_boids = num_boids
        self.n_dim = n_dim
        self.timestep = timestep
                
    @staticmethod
    def flock_mean(arr):
        """Computes the center of the group as well as the group's direction"""
        mean = np.mean(arr, axis=0)
        return mean
    
    #@jit
    def get_local_flock_idx(self, boid_idx, radius):
        """Within a flock of birds, we distinguish local groups. These are the 
        boids that respond to each other in terms of movement. This function
        finds the boid indices of the group a specific boid responds to.
        By convention, a boid is not in own group preventing interacting with itself"""
        flock_idx = np.linalg.norm(self.positions - self.positions[boid_idx,:], axis=1) < radius        
        flock_idx[boid_idx] = False
        
        return np.where(flock_idx)[0]
            
    @staticmethod
    @jit
    def steer_away(vel_boid, vel_group, prop):
        
        vel_boid_dir = vel_boid / np.linalg.norm(vel_boid)
        vel_group_dir = vel_group / np.linalg.norm(vel_group).reshape(-1,1)
        
        displacement = (prop.reshape(-1, 1)**1) * (vel_boid_dir - vel_group_dir) # squared
        vel_boid_update = vel_boid + np.mean(displacement, axis=0)
        vel_boid_update /= np.linalg.norm(vel_boid_update)
        
        return vel_boid_update
    
    #@jit
    def flock_alignment(self, idx, alpha, idx_group):
        """Steer towards the average heading of local flockmates with rate alpha
        Input: boid, flock of boid, effect strength
        Output: Size of velocity nudge"""
        
        group_vel = self.velocities[idx_group, :]
        group_mean_dir = self.flock_mean(group_vel)
        group_mean_dir /= np.linalg.norm(group_mean_dir)
        
        boid_dir = self.velocities[idx, :] / np.linalg.norm(self.velocities[idx, :])
        
        boid_dir_updated = boid_dir + alpha * group_mean_dir   
        boid_dir_updated /= np.linalg.norm(boid_dir_updated)
        
        return boid_dir_updated - boid_dir
    
    #@jit
    def flock_separation(self, idx, alpha, idx_group):
        """Steer to avoid crowding local flockmates. If no flockmates are within the specified
        radius, the radius is increased"""
        # Here, we need to find a rule that steers away proportional to the closeness to flockmates
        
        group_pos = self.positions[idx_group, :]
        group_vel = self.velocities[idx_group, :]
        
        boid_pos = self.positions[idx, :]
        boid_vel = self.velocities[idx, :]
        
        # Simple Euler forward approximation of next position
        group_pos_update = group_pos + self.timestep * group_vel
        
        group_pos_distances = np.linalg.norm(boid_pos - group_pos_update, axis=1)
        group_prop_repulsion = 1 / group_pos_distances
        
        boid_vel = self.steer_away(boid_vel, group_vel, group_prop_repulsion) - boid_vel
        
        return alpha*boid_vel
    
    #@jit
    def flock_cohesion(self, idx, alpha, idx_group=[], to_center=False):
        """steer to move towards the average position (center of mass) of local flockmates
        currently, the center point is still. 
        We can choose to first move the center point one timestep forward"""
        
        if len(idx_group)>0:
            group_pos = np.mean(self.positions[idx_group, :], axis=0)
        else:
            group_pos = np.array([0, 0])
            
        boid_pos = self.positions[idx, :]
        boid_vel = self.velocities[idx, :]
        
        # First find the angle between the boids position and direction and the centerpoint
        boid_direction = boid_pos + boid_vel
        boid_to_center = group_pos - boid_pos
        
        dotproduct = np.dot(boid_direction, boid_to_center)
        denominator = np.linalg.norm(boid_direction) * np.linalg.norm(boid_to_center)
        angle_to_center = np.arccos(dotproduct / denominator)
        
        # Reduce the angle by alpha and apply to velocity vector
        angle_new = alpha*angle_to_center
        rotation_matrix = Rotation.from_euler('z', angle_new).as_matrix()[:2, :2]
        rotated_vector = np.dot(rotation_matrix, boid_vel) 
        
        # if idx==2:
        #     print("boid velocity", boid_vel)
        #     print("boid rotated_vector", rotated_vector)
        
        return rotated_vector - boid_vel
        
    def flock_update(self, radius, rate):
        
        store_position = np.full((self.num_boids, self.n_dim), 0, dtype=np.float64)
        store_velocity = np.full((self.num_boids, self.n_dim), 0, dtype=np.float64)
        
        # A for-loop to find the positional/speed update for each boid.
        for idx in range(self.num_boids):
            
            radius_local = radius
            flock_idx = self.get_local_flock_idx(idx, radius_local)
            
            while len(flock_idx)==0:
                radius_local *= 1.1
                flock_idx = self.get_local_flock_idx(idx, radius_local)
            
            #print(idx)
            
            store_velocity[idx, :] += self.flock_separation(idx, alpha=rate, idx_group=flock_idx)
            store_velocity[idx, :] += self.flock_alignment(idx, alpha=rate, idx_group=flock_idx)
            store_velocity[idx, :] += self.flock_cohesion(idx, alpha=rate, idx_group=flock_idx)
            store_velocity[idx, :] += self.flock_cohesion(idx, alpha=0.0001)
        
        # print("self.velocities", self.velocities[2,])
        # print("store_velocity", store_velocity[2,])
        self.velocities += store_velocity
        self.velocities /= np.linalg.norm(self.velocities, axis=1).reshape(-1, 1)
        # print("self.velocities updated", self.velocities[2,])
        # print("self.position", self.positions[2,])
        # old_pos = self.positions[2,]
        self.positions += store_position + 1*self.timestep * self.velocities
        # print("self.position updated", self.positions[2,])
        # print("diff in positions in step:", self.positions[2,]- old_pos)

flock = Boids(num_boids=60, n_dim=2, timestep=1)

for i in range(200):
    #time.sleep(0.1)
    flock.flock_update(radius=20, rate=0.01)
    # print(i)
    plt.quiver(flock.positions[:,0], flock.positions[:,1],
                flock.velocities[:,0], flock.velocities[:,1])
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.show()
    
# Add this stuff to a pygame for better visualisation


    