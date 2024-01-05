# Boid simulation

In this project, bird-oid (boid) objects are simulated. Given three rules, complex group behaviour emerges. The rules that we base our model on is:

1) separation: steer to avoid crowding local flockmates,
2) alignment: steer towards the average heading of local flockmates,
3) cohesion: steer to move towards the average position (center of mass) of local flockmates.

We are free in how do model each interaction. From the description it is clear that every step refers only to local flock mates. Therefore we create a function that selects the flockmates that are within a radius w.r.t. Euclidean distance. Let $A$ be the set of indices that are within radius. Then for each boid we define functions that quantify the interaction effects. Let $x_1$, $v_1$, $x_2$, $v_2$ be the positions and velocities of boid 1 and boid 2.

The interactions within two separate boids are modelled according to the functions 
1) separation: $s(a, x_1, x_2, v_1, v_2)$
2) alignment: $a(a, x_1, x_2, v_1, v_2)$
3) cohesion: $c(a, x_1, x_2, v_1, v_2)$

such that the update in a single simulation step for a boid is given by
$$v_1 -> v_1 + \sum_{a\in A} s(a, x_1, x_2, v_1, v_2) + s(a, x_1, x_2, v_1, v_2) + s(a, x_1, x_2, v_1, v_2)$$

This models a change in velocity. The updated velocity is then used to update the position.

# Note

This project is unfinished. The current implementation of the separation step requires a better technical solution.

# Ideas

Once the flock behaves more naturally, the plan is to introduce predators which hunt the boids. Extra boid behaviour needs to be introduced, that is, moving away from the predator when it is detected. Rotating the boids velocity to the perpendicular velocity vector of the predator is the most natural solution. This is how groups of sardines respond to shark, for example.
