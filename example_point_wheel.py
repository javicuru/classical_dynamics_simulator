"""
This is a sample script showing how to:
    1) Simulate the motion of a classical point particle in a wheel, using the ClassicalDynamicsSimulator class.
    2) Animate the simulation.
    3) Save the animation

Note that the animation displayed may be too slow, but it can be saved with high fps (frames per second) to play it
faster.
"""

from classical_dynamics_simulator import ClassicalDynamicsSimulator
from classical_dynamics_simulator import Animator
import numpy as np

dt = 1e-3  # timestep [s]

m = 1  # mass [kg]

x0 = 0  # initial x[m]
y0 = 0  # initial y[m]

vx0 = 0  # initial vx [m/s]
vy0 = 0  # initial vy [m/s]

w = 10  # angular acceleration [s⁻¹]

T = 2*np.pi/w  # duration of the whole simulation [s]

n = int(T / dt)  # number of samples

fx = m*w * np.sin([w * i * dt for i in range(n)])  # force in x direction, at every instant [N]
fy = m*w * np.cos([w * i * dt for i in range(n)])  # force in y direction, at every instant [N]

simulator = ClassicalDynamicsSimulator(dt, x0, y0, vx0, vy0, fx, fy, m)

simulator.simulate()

animator = Animator(simulator, leave_trace=True)

animator.animate()

# Uncomment the following line to save the animation:

# animator.animation.save("point_wheel.gif", fps=20) # This is using Matplotlib's Animation.save method which has

# many more options for saving the animation. For more details, check:

# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.Animation.html#matplotlib.animation.Animation.save
