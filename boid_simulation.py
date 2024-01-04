from boid_rules import Boids
import numpy as np
import time
import sys
import pygame

# TODO: 
# 1) allow varying speeds among boids (contract towards 1 by updating |v| = np.tanh(|v|))
# 2) try to apply the rotation trick of cohesion to other update rules
# 3) include an euler forward step before calculating angle (done
# 4) [FOCUS] separation does not appear to work. find out why.
# 5) make cohesion proportional to distance. weak when close, strong when far.

def print_parameters(c, a, s, r):
    print(f"Cohesion: {c:.2f}, Alignment: {a:.2f}, Separation: {s:.2f}, Radius: {r}, ")

pygame.init()
width, height = 500, 500
screen = pygame.display.set_mode((width, height))

# Set up colors
blue = (0, 0, 255)
black = (0, 0, 0)
lightblue = (0, 0, 200)
white = (255,255,255)

arrow_length = 5

# Initialise the flock
flocksize = 100
radius = 50
flock = Boids(num_boids=flocksize, n_dim=2, timestep=1)

# Configurations
cohesion_rate = 0.01
alignment_rate = 0.01
separation_rate = 0.01

# Pygame loop
clock = pygame.time.Clock() 
running = True
while running:
    #time.sleep(1)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                flock = Boids(num_boids=flocksize, n_dim=2, timestep=1); print('Flock reset')
            
            if event.key == pygame.K_UP:
                cohesion_rate += 0.01
            if event.key == pygame.K_DOWN:
                cohesion_rate -= 0.01
         
            if event.key == pygame.K_RIGHT:
                alignment_rate += 0.01
            if event.key == pygame.K_LEFT:
                alignment_rate -= 0.01
        
            if event.key == pygame.K_w:
                separation_rate += 0.01
            if event.key == pygame.K_q:
                separation_rate -= 0.01
                
            if event.key == pygame.K_a:
                radius -= 1
            if event.key == pygame.K_s:
                radius += 1
            
            print_parameters(cohesion_rate, alignment_rate, separation_rate, radius)

    flock.flock_update(radius=10, c_rate=cohesion_rate, a_rate=alignment_rate, s_rate=separation_rate)

    screen.fill(blue)
    
    pygame.draw.circle(screen, black, [width/2, width/2], radius=radius/2)
    pygame.draw.circle(screen, blue, [width/2, width/2], radius=radius/2-1)

    for idx in range(flocksize):
        arrow_end = [flock.positions[idx, 0] + arrow_length * flock.velocities[idx, 0] + width/2,
                      flock.positions[idx, 1] + arrow_length * flock.velocities[idx, 1] + width/2]
        pygame.draw.line(screen, white, flock.positions[idx, :] + np.array([width/2, width/2]), arrow_end, 2)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
