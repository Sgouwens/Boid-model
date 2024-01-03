from boid_rules import Boids
import numpy as np

import pygame

def print_parameters(c, a, s, r):
    print(f"Cohesion: {c:.2f}, Alignment: {a:.2f}, Separation: {s:.2f}, Radius: {r}, ")

pygame.init()
width, height = 500, 500
screen = pygame.display.set_mode((width, height))

# Set up colors
white = (0, 0, 255)
blue = (255, 255, 255)
red = (255, 0, 0)

arrow_length = 5

# Initialise the flock
flocksize = 150
radius = 10
flock = Boids(num_boids=flocksize, n_dim=2, timestep=1)

# Configurations
cohesion_rate = 0.01
alignment_rate = 0.01
separation_rate = 0.01

# Pygame loop
clock = pygame.time.Clock() 
running = True
while running:
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

    screen.fill(white)

    for idx in range(flocksize):
        arrow_end = [flock.positions[idx, 0] + arrow_length * flock.velocities[idx, 0] + width/2,
                      flock.positions[idx, 1] + arrow_length * flock.velocities[idx, 1] + width/2]
        pygame.draw.line(screen, blue, flock.positions[idx, :] + np.array([width/2, width/2]), arrow_end, 2)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
