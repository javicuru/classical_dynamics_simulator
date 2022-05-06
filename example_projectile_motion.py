"""
This is a sample script showing how to:
    1) Simulate the projectile motion of a classical point particle, using the ClassicalDynamicsSimulator class.
    2) Animate the simulation.
    3) Save the animation

Note that the animation displayed may be too slow, but it can be saved with high fps (frames per second) to play it
faster.
"""

from classical_dynamics_simulator import ClassicalDynamicsSimulator
from classical_dynamics_simulator import Animator

dt = 1e-1  # timestep [s]

m = 1  # mass [kg]

x0 = 0  # initial x[m]
y0 = 1  # initial y[m]

vx0 = 10  # initial vx [m/s]
vy0 = 10  # initial vy [m/s]

T = 8  # duration of the whole simulation [s]

n = int(T / dt) # number of samples

fx = [0] * n        # force in x direction, at every instant [N]
fy = [-9.8] * n     # force in y direction, at every instant [N]

simulator = ClassicalDynamicsSimulator(dt, x0, y0, vx0, vy0, fx, fy, m)

simulator.simulate()

animator = Animator(simulator, leave_trace=True)

animator.animate()

# Uncomment the following line to save the animation:

# animator.animation.save("projectile_motion.gif", fps=20) # This is using Matplotlib's Animation.save method which has

# many more options for saving the animation. For more details, check:

# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.Animation.html#matplotlib.animation.Animation.save
