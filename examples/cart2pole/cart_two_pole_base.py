'''
The double pole balancing problem.

This is a modified version of the example in the book
"Hands-On Neuroevolution with Python" published by Packt.

Original code:
https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter4/cart_two_pole.py

MIT License
Copyright (c) 2019 Packt
https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/LICENSE
'''
#
# This is simulation of cart-poles apparatus with two poles based on the Newton laws
# which use Runge-Kutta fourth-order method for numerical approximation of system dynamics.
#
import math
import random

import matplotlib.pyplot as plt

#
# The constants defining physics of cart-2-poles apparatus
#
GRAVITY = -9.8  # m/s^2 - here negative as equations of motion for 2-pole system assume it to be negative
MASS_CART = 1.0  # kg
FORCE_MAG = 10.0  # N
# The first pole
MASS_POLE_1 = 1.0  # kg
LENGTH_1 = 1.0 / 2.0  # m - actually half the first pole's length (center of mass)
# The second pole
MASS_POLE_2 = 0.1  # kg
LENGTH_2 = 0.1 / 2.0  # m - actually half the second pole's length (center of mass)
# The coefficient of friction of pivot of the pole
MUP = 0.000002

# The maximum allowed angle of pole fall from the vertical
THIRTY_SIX_DEG_IN_RAD = (36 * math.pi) / 180.0  # rad


def calc_step(action, x, x_dot, theta1, theta1_dot, theta2, theta2_dot):
    """
    The function to perform calculations of system dynamics for one step
    of simulations.
    Arguments:
        action:     The binary action defining direction of
                    force to be applied.
        x:          The current cart X position
        x_dot:      The velocity of the cart
        theta1:      The current angle of the first pole from vertical
        theta1_dot:  The angular velocity of the first pole.
        theta2:      The current angle of the second pole from vertical
        theta2_dot:  The angular velocity of the second pole.
    Returns:
        The calculated values for cart acceleration along with angular
        accelerations of both poles.
    """
    # Find the input force direction
    # force = -FORCE_MAG if action == 0 else FORCE_MAG # action has binary values
    force = (action - 0.5) * FORCE_MAG * 2.0  # continuous action
    # Calculate projections of forces for the poles
    cos_theta_1 = math.cos(theta1)
    sin_theta_1 = math.sin(theta1)
    g_sin_theta_1 = GRAVITY * sin_theta_1
    cos_theta_2 = math.cos(theta2)
    sin_theta_2 = math.sin(theta2)
    g_sin_theta_2 = GRAVITY * sin_theta_2
    # Calculate intermediate values
    ml_1 = LENGTH_1 * MASS_POLE_1
    ml_2 = LENGTH_2 * MASS_POLE_2
    temp_1 = MUP * theta1_dot / ml_1
    temp_2 = MUP * theta2_dot / ml_2
    fi_1 = (ml_1 * theta1_dot * theta1_dot * sin_theta_1) + \
        (0.75 * MASS_POLE_1 * cos_theta_1 * (temp_1 + g_sin_theta_1))
    fi_2 = (ml_2 * theta2_dot * theta2_dot * sin_theta_2) + \
        (0.75 * MASS_POLE_2 * cos_theta_2 * (temp_2 + g_sin_theta_2))
    mi_1 = MASS_POLE_1 * (1 - (0.75 * cos_theta_1 * cos_theta_1))
    mi_2 = MASS_POLE_2 * (1 - (0.75 * cos_theta_2 * cos_theta_2))
    # Calculate the results: cart acceleration and poles angular accelerations
    x_ddot = (force + fi_1 + fi_2) / (mi_1 + mi_2 + MASS_CART)
    theta_1_ddot = -0.75 * (x_ddot * cos_theta_1 + g_sin_theta_1 + temp_1) / LENGTH_1
    theta_2_ddot = -0.75 * (x_ddot * cos_theta_2 + g_sin_theta_2 + temp_2) / LENGTH_2

    return x_ddot, theta_1_ddot, theta_2_ddot


def outside_bounds(x, theta1, theta2):
    """
    Function to test whether cart-2-pole system is outside of the
    constraints.
    Arguments:
        action:     The binary action defining direction of
                force to be applied.
        x:          The current cart X position.
        theta1:      The current angle of the first pole from vertical.
        theta2:      The current angle of the second pole from vertical.
    Returns:
        True if system violated constraints, False - otherwise.
    """
    res = x < -2.4 or x > 2.4 or \
        theta1 < -THIRTY_SIX_DEG_IN_RAD or theta1 > THIRTY_SIX_DEG_IN_RAD or \
        theta2 < -THIRTY_SIX_DEG_IN_RAD or theta2 > THIRTY_SIX_DEG_IN_RAD
    return res


def rk4(f, y, dydx, tau):
    """
    The Runge-Kutta fourth order method of numerical approximation of
    the double-pole-cart system dynamics. This function will update
    values in provided list with state variables (y).
    Arguments:
        f:      The current control action
        y:      The list with current system state variables
                (x, x_dot, theta1, theta1_dot, theta2, theta2_dot)
        dydx:   The list with derivatives of current state variables
        tau:    The simulation approximation time step size
    """
    hh = tau / 2.0
    yt = [None] * 6
    # update intermediate state
    for i in range(6):
        yt[i] = y[i] + hh * dydx[i]
    # do simulation step
    x_ddot, theta_1_ddot, theta_2_ddot = calc_step(action=f,
                                                   x=yt[0],
                                                   x_dot=yt[1],
                                                   theta1=yt[2],
                                                   theta1_dot=yt[3],
                                                   theta2=yt[4],
                                                   theta2_dot=yt[5])
    # store derivatives
    dyt = [yt[1], x_ddot, yt[3], theta_1_ddot, yt[5], theta_2_ddot]

    # update intermediate state
    for i in range(6):
        yt[i] = y[i] + hh * dyt[i]

    # do one simulation step
    x_ddot, theta_1_ddot, theta_2_ddot = calc_step(action=f,
                                                   x=yt[0],
                                                   x_dot=yt[1],
                                                   theta1=yt[2],
                                                   theta1_dot=yt[3],
                                                   theta2=yt[4],
                                                   theta2_dot=yt[5])
    # store derivatives
    dym = [yt[1], x_ddot, yt[3], theta_1_ddot, yt[5], theta_2_ddot]

    # update intermediate state
    for i in range(6):
        yt[i] = y[i] + tau * dym[i]
        dym[i] += dyt[i]

    # do one simulation step
    x_ddot, theta_1_ddot, theta_2_ddot = calc_step(action=f,
                                                   x=yt[0],
                                                   x_dot=yt[1],
                                                   theta1=yt[2],
                                                   theta1_dot=yt[3],
                                                   theta2=yt[4],
                                                   theta2_dot=yt[5])
    # store derivatives
    dyt = [yt[1], x_ddot, yt[3], theta_1_ddot, yt[5], theta_2_ddot]

    # find system state after approximation
    h6 = tau / 6.0
    for i in range(6):
        y[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i])


def apply_action(action, state, step_number):
    """
    Method to apply the control action to the cart-pole simulation.
    Arguments:
        action:      The binary action defining direction of
                        force to be applied.
        state:       The state variables (x, x_dot, theta1, theta1_dot, theta2, theta2_dot)
        step_number: The current simulation step number
    Returns:
        The updated state.
    """
    # The simulation time step size
    TAU = 0.01

    # The control inputs frequency is two times less than simulation
    # step frequency - hence do two simulation steps
    dydx = [None] * 6  # the state derivatives holder
    for _ in range(2):
        # copy the state derivatives
        dydx[0] = state[1]  # x_dot
        dydx[2] = state[3]  # theta1_dot
        dydx[4] = state[5]  # theta2_dot
        # do one simulation step and store derivatives
        x_ddot, theta_1_ddot, theta_2_ddot = calc_step(action=action,
                                                       x=state[0], x_dot=state[1],
                                                       theta1=state[2], theta1_dot=state[3],
                                                       theta2=state[4], theta2_dot=state[5])
        dydx[1] = x_ddot
        dydx[3] = theta_1_ddot
        dydx[5] = theta_2_ddot
        # do Runge-Kutta numerical approximation and update state
        rk4(f=action, y=state, dydx=dydx, tau=TAU)

    # return the updated state values (x, x_dot, theta1, theta1_dot, theta2, theta2_dot)
    return state


def run_simulation(net, max_bal_steps=100000, bool_discrete_action=False):
    """
    The function to run cart-two-pole apparatus simulation for a
    certain number of time steps as maximum.
    Arguments:
        net: The ANN of the phenotype to be evaluated.
        max_bal_steps: The maximum number of time steps to
            execute simulation.
        bool_discrete_action: Discrete action space?
        act_interval: Activates network every act_interval time steps. 
    Returns:
        the number of steps that the control ANN was able to
        maintain the single-pole balancer in stable state.
    """
    # Run simulation for specified number of steps while
    # cart-pole system stays within constraints
    # input = [None] * 6 # the inputs
    input = [None] * 3  # the inputs
    state = reset_state([None] * 6)
    for steps in range(max_bal_steps):
        # # scale inputs
        # input[0] = (state[0] + 2.4) / 4.8
        # input[1] = (state[1] + 1.5) / 3.0
        # input[2] = (state[2] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        # input[3] = (state[3] + 2.0) / 4.0
        # input[4] = (state[4] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        # input[5] = (state[5] + 2.0) / 4.0

        # scale inputs
        input[0] = (state[0] + 2.4) / 4.8
        input[1] = (state[2] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        input[2] = (state[4] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)

        # Activate the NET
        output = net.calculateNetwork(input)
        action = output[0]

        if bool_discrete_action:
            # Make action values discrete
            action = 0 if output[0] < 0.5 else 1

        # Apply action to the simulated cart-two-pole
        state = apply_action(action=action, state=state, step_number=steps)

        # check if simulation still within bounds
        if outside_bounds(x=state[0], theta1=state[2], theta2=state[4]):
            return steps

    return max_bal_steps


def run_simulation_gruau(net, max_bal_steps=1000, bool_discrete_action=False):
    """
    The function to run cart-two-pole apparatus simulation for a
    certain number of time steps as maximum.
    Arguments:
        net: The ANN of the phenotype to be evaluated.
        max_bal_steps: The maximum number of time steps to
            execute simulation.
        bool_discrete_action: Discrete action space?
        act_interval: Activates network every act_interval time steps. 
    Returns:
        the number of steps that the control ANN was able to
        maintain the single-pole balancer in stable state.
    """
    # Run simulation for specified number of steps while
    # cart-pole system stays within constraints
    # input = [None] * 6 # the inputs
    input = [None] * 3  # the inputs
    state = reset_state([None] * 6)
    f_1, f_2 = 0.0, 0.0
    tmp_f_2 = []
    for steps in range(max_bal_steps):
        # # scale inputs
        # input[0] = (state[0] + 2.4) / 4.8
        # input[1] = (state[1] + 1.5) / 3.0
        # input[2] = (state[2] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        # input[3] = (state[3] + 2.0) / 4.0
        # input[4] = (state[4] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        # input[5] = (state[5] + 2.0) / 4.0

        # scale inputs
        input[0] = (state[0] + 2.4) / 4.8
        input[1] = (state[2] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        input[2] = (state[4] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)

        # Activate the NET
        output = net.calculateNetwork(input)
        action = output[0]

        if bool_discrete_action:
            # Make action values discrete
            action = 0 if output[0] < 0.5 else 1

        # Apply action to the simulated cart-two-pole
        state = apply_action(action=action, state=state, step_number=steps)

        f_1 = steps / 1000.0
        tmp_f_2 += [math.fabs(state[0]) + math.fabs(state[1]) + math.fabs(state[2]) + math.fabs(state[3])]

        if steps >= 100:
            tmp_f_2.pop(0)
            f_2 = 0.75 / sum(tmp_f_2)

        # check if simulation still within bounds
        if outside_bounds(x=state[0], theta1=state[2], theta2=state[4]):
            return 0.1 * f_1 + 0.9 * f_2

    return 0.1 * f_1 + 0.9 * f_2


class CartAnimation:
    def __init__(self, ax, net, min_x, max_x,
                 pole_length_1=LENGTH_1, pole_length_2=LENGTH_2, max_bal_steps=1000):
        self.ax = ax
        self.ax.set_aspect('equal')  # same scaling for x and y.
        self.net = net
        self.min_x = min_x
        self.max_x = max_x
        self.pole_length_1 = pole_length_1 * 2.0  # actual pole length
        self.pole_length_2 = pole_length_2 * 2.0  # actual pole length
        self.max_bal_steps = max_bal_steps

        # self.input = [None] * 6 # the inputs
        self.input = [None] * 3  # the inputs
        self.state = reset_state([None] * 6)

        self.failed_flag = False

    def run_simulation_anim(self, frame, bool_discrete_action=False, act_interval=1):
        """
        The function to run cart-two-pole apparatus simulation for a
        certain number of time steps as maximum.
        Arguments:
            net: The ANN of the phenotype to be evaluated.
            max_bal_steps: The maximum number of time steps to
                execute simulation.
            bool_discrete_action: Discrete action space?
            act_interval: Activates network every act_interval time steps. 
        """
        # Run simulation for specified number of steps while
        # cart-pole system stays within constraints

        plt.cla()  # clear the previous figure

        # scale inputs
        # self.input[0] = (self.state[0] + 2.4) / 4.8
        # self.input[1] = (self.state[1] + 1.5) / 3.0
        # self.input[2] = (self.state[2] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        # self.input[3] = (self.state[3] + 2.0) / 4.0
        # self.input[4] = (self.state[4] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        # self.input[5] = (self.state[5] + 2.0) / 4.0

        # scale inputs
        self.input[0] = (self.state[0] + 2.4) / 4.8
        self.input[1] = (self.state[2] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)
        self.input[2] = (self.state[4] + THIRTY_SIX_DEG_IN_RAD) / (THIRTY_SIX_DEG_IN_RAD * 2.0)

        if frame % act_interval == 0:
            # Activate the NET
            output = self.net.calculateNetwork(self.input)
            action = output[0]

            if bool_discrete_action:
                # Make action values discrete
                action = 0 if output[0] < 0.5 else 1
        else:
            # Zero force
            action = 0.5

        # Apply action to the simulated cart-two-pole
        self.state = apply_action(action=action, state=self.state, step_number=frame)

        # check if simulation still within bounds
        if outside_bounds(x=self.state[0], theta1=self.state[2], theta2=self.state[4]):
            # plt.clf()
            # plt.close()
            if self.failed_flag == False:
                print("Failed at {} time steps. ".format(frame))
                self.failed_flag = True

        # data to be plotted
        cart_x = self.state[0]
        theta_1 = self.state[2]
        theta_2 = self.state[4]

        # draw timestep
        t_text = self.ax.text(self.min_x + 0.02 * (self.max_x-self.min_x),
                              self.pole_length_1 + 0.05 * (self.max_x-self.min_x),
                              "timestep: {}".format(frame))

        # draw poles
        pole_end_1 = (cart_x + math.sin(theta_1) * self.pole_length_1, math.cos(theta_1) * self.pole_length_1)
        pole_end_2 = (cart_x + math.sin(theta_2) * self.pole_length_2, math.cos(theta_2) * self.pole_length_2)
        pole_1 = plt.plot([cart_x, pole_end_1[0]], [0.0, pole_end_1[1]], c='tab:cyan', linewidth=2.5)
        pole_2 = plt.plot([cart_x, pole_end_2[0]], [0.0, pole_end_2[1]], c='tab:orange', linewidth=2.5)

        # draw cart
        rect = plt.Rectangle((cart_x - 0.05 * (self.max_x-self.min_x), -0.04 * (self.max_x-self.min_x)),
                             0.1 * (self.max_x-self.min_x), 0.04 * (self.max_x-self.min_x),
                             fc='tab:gray', zorder=0)
        self.ax.add_patch(rect)
        joint = plt.scatter([cart_x], [0.0], c='black', s=3, zorder=10)
        tire = plt.scatter([cart_x - 0.03 * (self.max_x-self.min_x),
                           cart_x + 0.03 * (self.max_x-self.min_x)],
                           [-0.04 * (self.max_x-self.min_x), -0.04 * (self.max_x-self.min_x)],
                           c='black', s=50)

        self.ax.set_xlim([self.min_x, self.max_x])
        self.ax.set_ylim([-0.054 * (self.max_x-self.min_x), self.pole_length_1 + 0.03 * (self.max_x-self.min_x)])
        self.ax.set_ylim(bottom=-0.054 * (self.max_x-self.min_x))

        # Hide labels and ticks for Y axis.
        self.ax.yaxis.set_tick_params(labelleft=False)
        self.ax.set_yticks([])

        # if frame >= self.max_bal_steps:
        #     plt.clf()
        #     plt.close()


def reset_state(state):
    """
    The function to reset state array to initial values.
    Arguments:
        state: the array with state variables.
    Returns:
        The state array with values set to initial ones.
    """
    state[0], state[1], state[3], state[4], state[5] = 0, 0, 0, 0, 0
    state[2] = math.pi / 180.0  # the one_degree
    return state
