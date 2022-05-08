from classical_dynamics_simulator import ClassicalDynamicsSimulator as CDS
from classical_dynamics_simulator import Animator

import numpy as np

dt = 1e-3  # timestep [s]

m = 1  # mass [kg]

x0 = 0  # initial x[m]
y0 = 0  # initial y[m]

w = 2 * np.pi / 1
A = 1.5

vx0 = w * A  # initial vx [m/s]
vy0 = 0  # initial vy [m/s]

T = .7  # duration of the whole simulation [s]

n = int(T / dt)

fx = -A * m * w**2 * np.sin([w * i * dt for i in range(n)])
fy = [0] * n

simulator = CDS(dt, x0, y0, vx0, vy0, fx, fy, m)

simulator.simulate()

animator = Animator(simulator)

animator.animate()

# Uncomment the following line to save the animation:

# animator.animation.save("projectile_motion.gif", fps=20) # This is using Matplotlib's Animation.save method which has

# many more options for saving the animation. For more details, check:

# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.Animation.html#matplotlib.animation.Animation.save
