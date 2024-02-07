from boid_rules import Boids

import numpy as np
import time
import sys
import pygame
import seaborn as sns

def print_parameters(cohesion, alignment, separation, origin, radius):
    print(f"""
          Cohesion: {cohesion:.2f}, 
          Alignment: {alignment:.2f}, 
          Separation: {separation:.2f}, 
          Cohesion to dot: {origin:.2f}, 
          Radius: {radius}""")

pygame.init()
size = 800
screen = pygame.display.set_mode((size, size))

# Set up colors
blue = (0, 0, 255)
red = (255, 0, 0)
black = (0, 0, 0)
lightblue = (0, 0, 200)
white = (255,255,255)

mouse_x, mouse_y = size/2, size/2
new_center = np.array([0, 0])

arrow_length =  5

# Initial configurations
flocksize = 100
radius = 50

cohesion_rate = 0.01
alignment_rate = 0.01
separation_rate = 0.01
origin_rate = 0.02

flock = Boids(num_boids=flocksize, n_dim=2, timestep=1)

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
                new_center = np.array([0, 0])
            
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
        
            if event.key == pygame.K_p:
                origin_rate += 0.01
            if event.key == pygame.K_o:
                origin_rate -= 0.01
                    
            if event.key == pygame.K_a:
                radius -= 1
            if event.key == pygame.K_s:
                radius += 1
                
            print_parameters(cohesion_rate, alignment_rate, separation_rate, origin_rate, radius)
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_x, mouse_y = pygame.mouse.get_pos()
                new_center = np.array([mouse_x-size/2, mouse_y-size/2])
                print(f"Centerpoint updated to: ({mouse_x}, {mouse_y})")
                print_parameters(cohesion_rate, alignment_rate, separation_rate, origin_rate, radius)
            
        
            
    flock.flock_update(radius=radius, 
                       c_rate=cohesion_rate, 
                       a_rate=alignment_rate, 
                       s_rate=separation_rate,
                       o_rate=origin_rate,
                       center=new_center)
    
    
    screen.fill(blue)
    pygame.draw.circle(screen, black, [size/2, size/2], radius=radius/2)
    pygame.draw.circle(screen, blue, [size/2, size/2], radius=radius/2-1)
    pygame.draw.circle(screen, red, [mouse_x, mouse_y], radius=2)
    
    for idx in range(flocksize):
        
        arrows = flock.polar_to_vec(flock.velocities, flock.angles)
        arrow_end = [flock.positions[idx, 0] + arrow_length * arrows[idx, 0] + size/2,
                      flock.positions[idx, 1] + arrow_length * arrows[idx, 1] + size/2]
        pygame.draw.line(screen, white, flock.positions[idx, :] + np.array([size/2, size/2]), arrow_end, 2)
        

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
