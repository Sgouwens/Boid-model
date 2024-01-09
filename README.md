# Boid simulation

In this project, bird-oid (boid) objects are simulated. Given three rules, complex group behaviour emerges. The rules that we base our model on is:

1) Separation: steer to avoid crowding local flockmates,
2) Alignment: steer towards the average heading of local flockmates,
3) Cohesion: steer to move towards the average position (center of mass) of local flockmates.

In this project, no further specifications were given. From the description it is clear that every step refers only to local flock mates. Therefore we create a function that selects the flockmates that are within a radius w.r.t. Euclidean distance. Let $J$ be the set of indices that are within radius. Then for each boid $j$, we define functions that quantify the interaction effects. Let $x_1$, $v_1$, $x_2$, $v_2$ be the positions and velocities of boid 1 and boid 2.

The interactions within two separate boids are modelled according to the functions 
1) Separation: $S(a, x_1, x_2, v_1, v_2)$
2) Alignment: $A(a, x_1, x_2, v_1, v_2)$
3) Cohesion: $C(a, x_1, x_2, v_1, v_2)$

such that the update in a single simulation step for a boid is given by
$$v_1^{(t+1)} = v_1^{(t)} + \sum_{j\in J} S(a, x_1, x_2, v_1, v_2) + A(a, x_1, x_2, v_1, v_2) + C(a, x_1, x_2, v_1, v_2)$$

This models a change in velocity. The updated velocity is then used to update the position.

# Note

This project is unfinished. The current implementation of the separation step requires a better technical solution.

# Ideas

Once the flock behaves more naturally, the plan is to introduce predators which hunt the boids. Extra boid behaviour needs to be introduced, that is, moving away from the predator when it is detected. Rotating the boids velocity to the perpendicular velocity vector of the predator is the most natural solution. This is how groups of sardines respond to shark, for example.
