"""
ClassicalDynamicSimulator and Animator class, for simulating and animating the motion of a classical point particle,
given its initial position and velocity, and the input force at every time.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class ClassicalDynamicsSimulator:
    """
    Class for numerical computation of a classical point particle trajectory.

    Parameters
    ----------
    timestep : float
        Time elapsed between steps, in seconds. The number of steps is len(fx)=len(fy). Hence the total time of
        the simulation is timestep * len(fx)

    x0 : float
        Initial x position, in metres.

    y0 : float
        Initial y position, in metres.

    vx0: float
        Initial x velocity, in m/s.

    vy0: float
        Initial y velocity, in m/s.

    fx: list or array
        Input x force at each instant, in newtons. len(fx) = len(fy) is the number of steps.

    fy: list or array
        Input y force at each instant, in newtons. len(fy) = len(fy) is the number of steps.

    mass: float
        Mass of the point particle, in kg.
    """

    def __init__(self, timestep, x0, y0, vx0, vy0, fx, fy, mass):
        self.params = {'timestep': timestep,
                       'init_x': x0, 'init_y': y0,
                       'init_vx': vx0, 'init_vy': vy0,
                       'force_x': fx, 'force_y': fy,
                       'mass': mass}

        self.x = None

        self.y = None

        self.vx = None

        self.vy = None

    def simulate(self):
        """
        Compute position and velocity at each instant.
        """
        if len(self.params.get('force_x')) != len(self.params.get('force_y')):
            raise ValueError("fx and fy must have the same length.")

        self.params['force_x'] = np.array(self.params.get('force_x'))
        self.params['force_y'] = np.array(self.params.get('force_y'))

        n_steps = len(self.params.get('force_x'))

        self.x = [self.params.get('init_x')]
        self.y = [self.params.get('init_y')]

        self.vx = [self.params.get('init_vx')]
        self.vy = [self.params.get('init_vy')]

        dt = self.params.get('timestep')

        for i in range(n_steps - 1):
            acx = self.params.get('force_x')[i] / self.params.get('mass')
            acy = self.params.get('force_y')[i] / self.params.get('mass')

            self.x.append(self.x[-1] + self.vx[-1] * dt + .5 * acx * dt ** 2)
            self.y.append(self.y[-1] + self.vy[-1] * dt + .5 * acy * dt ** 2)

            self.vx.append(self.vx[-1] + acx * dt)
            self.vy.append(self.vy[-1] + acy * dt)


class Animator:
    """
    Class for animation of classical dynamics simulation.

    Note that the animation displayed with the Animator.animate
    method may be too slow depending on the number of frames, but the animation may be saved with high fps (frames per
    second) using Matplotlib's Animation.save method, to play it faster.

    Parameters
    ----------
    simulator : ClassicalDynamicsSimulator

    scatter_size: int
        Size of the plotted particle point.

    leave_trace: bool
        If True, the particle leaves a trace as it moves.

    force_scaler: float
        Coefficient to modify the size of the force arrow.
        If force_scaler == 0, no arrow is plotted.

    vel_scaler: float
        Coefficient to modify the size of the plotted velocity arrow.
        If vel_scaler == 0, no arrow is plotted.

    force_width: float
        Width of the plotted force arrow.

    vel_width: float
        Width of the plotted velocity arrow.

    xlim: list
        Limits of the x axis.

    ylim: list
        Limits of the y axis.
    """
    def __init__(self, simulator, scatter_size=100, leave_trace=False, force_scaler=None, vel_scaler=None,
                 force_width=None, vel_width=None, xlim=None, ylim=None):
        self.simulator = simulator

        self.scatter_size = scatter_size

        self.leave_trace = leave_trace

        self.force_scaler = force_scaler

        self.vel_scaler = vel_scaler

        self.force_width = force_width

        self.vel_width = vel_width

        self.xlim = xlim

        self.ylim = ylim

        self.animation = None

    def animate(self):
        """
        Start the animation.
        """
        self.__set_defaults()

        fig, ax = plt.subplots()

        self.animation = FuncAnimation(fig,
                                       self.__animate_builder(ax),
                                       frames=len(self.simulator.params.get('force_x')), interval=5, repeat=True,
                                       blit=False)

        plt.show()

    def __animate_builder(self, ax):  # this is a wrapper function for passing ax to _animate
        def _animate(i):
            """
            Plot point particle, arrows, etc.
            """
            ax.clear()

            ax.scatter(self.simulator.x[i], self.simulator.y[i], self.scatter_size, 'black', zorder=2)

            if self.leave_trace:
                ax.plot(self.simulator.x[:i], self.simulator.y[:i], self.scatter_size, color='black',
                        linestyle='dashed', zorder=3)

            ax.arrow(self.simulator.x[i], self.simulator.y[i], self.vel_scaler * self.simulator.vx[i],
                     self.vel_scaler * self.simulator.vy[i], width=self.vel_width, color='green', label='Velocity',
                     alpha=.8, zorder=1, length_includes_head=False)

            ax.arrow(self.simulator.x[i], self.simulator.y[i],
                     self.force_scaler * self.simulator.params.get('force_x')[i],
                     self.force_scaler * self.simulator.params.get('force_y')[i], width=self.force_width, color='red',
                     label='Force',
                     zorder=0, length_includes_head=False)

            ax.set_aspect('equal', 'box')

            ax.set_xlim(self.xlim)

            ax.set_ylim(self.ylim)

            ax.legend(loc='upper right', framealpha=0.5)

            ax.set_xlabel('x')

            ax.set_ylabel('y')

        return _animate

    def __set_defaults(self):
        if self.force_scaler is None:
            self.force_scaler = self.__force_autoscaler()

        if self.vel_scaler is None:
            self.vel_scaler = self.__vel_autoscaler()

        if self.force_width is None:
            self.__force_autowidth()

        if self.vel_width is None:
            self.__vel_autowidth()

        if self.xlim is None:
            self.__axis_autolimits("x")

        if self.ylim is None:
            self.__axis_autolimits("y")

        self.__reset_lims_legend()

    def __axis_autolimits(self, axis):
        """
        Automatic adjustment of x and y axis.
        """
        if axis == 'x':
            self.xlim = [min(self.simulator.x +
                             list(np.array(self.simulator.x) +
                             self.force_scaler * self.simulator.params.get('force_x')) +
                            [self.simulator.x[i] +
                             self.vel_scaler * self.simulator.vx[i] for i in range(len(self.simulator.x))]),
                         max(self.simulator.x +
                             list(np.array(self.simulator.x) +
                             self.force_scaler * self.simulator.params.get('force_x')) +
                            [self.simulator.x[i] +
                             self.vel_scaler * self.simulator.vx[i] for i in range(len(self.simulator.x))])]

        if axis == 'y':
            self.ylim = [min(self.simulator.y +
                             list(np.array(self.simulator.y) +
                             self.force_scaler * self.simulator.params.get('force_y')) +
                            [self.simulator.y[i] +
                             self.vel_scaler * self.simulator.vy[i] for i in range(len(self.simulator.y))]),
                         max(self.simulator.y +
                             list(np.array(self.simulator.y) +
                             self.force_scaler * self.simulator.params.get('force_y')) +
                            [self.simulator.y[i] +
                             self.vel_scaler * self.simulator.vy[i] for i in range(len(self.simulator.y))])]

    def __reset_lims_legend(self):
        """
        Modify x and y limits to make space fot the legend.
        """
        p0 = .3

        dx = self.xlim[1] - self.xlim[0]

        dy = self.ylim[1] - self.ylim[0]

        p = dy / dx

        if p < p0:
            print("Resetting ylim to fit legend into box.")

            self.ylim[0] -= .5 * (p0 - p) * dx

            self.ylim[1] += .5 * (p0 - p) * dx

        elif p ** -1 < p0:
            print("Resetting xlim to fit legend into box.")

            self.xlim[0] -= .5 * (p0 - p ** -1) * dy

            self.xlim[1] += .5 * (p0 - p ** -1) * dy

    def __force_autoscaler(self):
        """
        Automatic setting of force arrow scaler.
        """
        k = .15

        if np.any(self.simulator.params.get('force_y')) and not np.any(self.simulator.params.get('force_x')):
            return k * (max(self.simulator.y) - min(self.simulator.y)) /\
                   np.mean(abs(self.simulator.params.get('force_y')))

        elif np.any(self.simulator.params.get('force_x')) and not np.any(self.simulator.params.get('force_y')):
            return k * (max(self.simulator.x) - min(self.simulator.x)) /\
                   np.mean(abs(self.simulator.params.get('force_x')))

        else:
            return k * min(max(self.simulator.x) - min(self.simulator.x),
                           max(self.simulator.y) - min(self.simulator.y)) / \
                   max(np.mean(abs(self.simulator.params.get('force_x'))),
                       np.mean(abs(self.simulator.params.get('force_y'))))

    def __vel_autoscaler(self):
        """
        Automatic setting of velocity arrow scaler.
        """
        k = .15

        if np.any(self.simulator.vy) and not np.any(self.simulator.vx):
            return k * (max(self.simulator.y) - min(self.simulator.y)) / np.mean(abs(np.array(self.simulator.vy)))
            # return k * (max(self.simulator.y) - min(self.simulator.y)) / max(abs(np.array(self.simulator.vy)))

        elif np.any(self.simulator.vx) and not np.any(self.simulator.vy):
            return k * (max(self.simulator.x) - min(self.simulator.x)) / np.mean(abs(np.array(self.simulator.vx)))
            # return k * (max(self.simulator.x) - min(self.simulator.x)) / max(abs(np.array(self.simulator.vx)))

        else:
            return k * min(max(self.simulator.x) - min(self.simulator.x),
                           max(self.simulator.y) - min(self.simulator.y)) / \
                   max(np.mean(abs(np.array(self.simulator.vx))), np.mean(abs(np.array(self.simulator.vy))))
            # max(max(abs(np.array(self.simulator.vx))), max(abs(np.array(self.simulator.vy))))

    def __force_autowidth(self):
        """
        Automatic setting of force arrow width.
        """
        k = .1

        if np.any(self.simulator.params.get('force_y')) and not np.any(self.simulator.params.get('force_x')):
            self.force_width = k * self.force_scaler * max(abs(self.simulator.params.get('force_y')))

        elif np.any(self.simulator.params.get('force_y')) and not np.any(self.simulator.params.get('force_x')):
            self.force_width = k * self.force_scaler * max(abs(self.simulator.params.get('force_x')))

        else:
            self.force_width = k * self.force_scaler * max(max(abs(self.simulator.params.get('force_y'))),
                                                           max(abs(self.simulator.params.get('force_x'))))

    def __vel_autowidth(self):
        """
        Automatic setting of velocity arrow width.
        """
        k = .1

        if np.any(self.simulator.vy) and not np.any(self.simulator.vx):
            self.vel_width = k * self.vel_scaler * max(abs(self.simulator.vy))

        elif np.any(self.simulator.vy) and not np.any(self.simulator.vx):
            self.vel_width = k * self.vel_scaler * max(abs(self.simulator.vx))

        else:
            self.vel_width = k * self.vel_scaler * max(max(abs(np.array(self.simulator.vy))),
                                                       max(abs(np.array(self.simulator.vx))))
