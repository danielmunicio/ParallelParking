#!/usr/bin/env python
"""
Starter code for EECS C106B Spring 2022 Project 2.
Author: Valmik Prabhu, Amay Saxena

Implements the self.optimization-based path planner.
"""

import scipy as sp
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from configuration_space import BicycleConfigurationSpace, Plan, expanded_obstacles
#from self.optimization_planner_casadi import plan_to_pose

class OptimizationPlanner(object):
    def __init__(self, config_space):
        self.config_space = config_space

        self.input_low_lims = self.config_space.input_low_lims
        self.input_high_lims = self.config_space.input_high_lims
        self.robot_length = self.config_space.robot_length

        self.opti = ca.Opti()
        self.q = ca.SX.sym('x', 4)
        self.u = ca.SX.sym('u', 2)
        self.t = ca.SX.sym('t')
        self.x_low_lims = self.config_space.low_lims
        self.x_high_lims = self.config_space.high_lims
        self.dynamics = self.q + self.t * ca.vertcat(
            ca.cos(self.q[2]) * self.u[0], 
            ca.sin(self.q[2]) * self.u[0],
            (1/self.robot_length) * ca.tan(self.q[3]) * self.u[0],
            self.u[1]
        )
    
        self.thefunkymonkey = ca.Function('thefunkymonkey', [self.q, self.u, self.t], [self.dynamics])

    def reid_big_function(self, start, goal, N, dt):
        fix_angle = ca.Function('fix_angle', [self.q], [ca.vertcat(self.q[0], self.q[1], ca.sin(self.q[2]/2), self.q[3])])

        X = []
        U = []
        X0 = self.opti.parameter(4)
        Xf = self.opti.parameter(4)
        self.opti.set_value(X0, start.flatten().tolist())
        self.opti.set_value(Xf, goal.flatten().tolist())

        for i in range(N):
            X.append(self.opti.variable(4))
            U.append(self.opti.variable(2))
        X.append(self.opti.variable(4))
        # cost = ca.sumsqr(fix_angle(X[-1] - Xf))
        # cost = 0
        for k in range(N):
            self.opti.subject_to(X[k+1] == self.thefunkymonkey(X[k], U[k], dt))

            if k == 0:
                self.opti.subject_to(fix_angle(X[0]-X0)==ca.DM([0.0]*4))
            if k == N-1:
                self.opti.subject_to(fix_angle(X[N-1]-Xf) == ca.DM([0.0]*4))

            self.opti.subject_to(U[k] >= ca.DM(self.input_low_lims))
            self.opti.subject_to(U[k] <= ca.DM(self.input_high_lims))
            self.opti.subject_to(X[k] >= ca.DM(self.x_low_lims))
            self.opti.subject_to(X[k] <= ca.DM(self.x_high_lims))
            
            for obs in self.config_space.obstacles:
                self.opti.subject_to(ca.sumsqr(X[k][0:2] - obs[0:2]) >= obs[2] ** 2)

        X = ca.hcat(X)

        self.options = {}
        self.options["structure_detection"] = "auto"
        self.options["debug"] = False

        q = self.opti.variable(4, N+1)
        u = self.opti.variable(2, N)

        Q = np.diag([1, 1, 2, 0.1])
        R = 2 * np.diag([1, 0.5])
        P = N * Q

        obj = self.objective_func(q, u, goal, Q, R, P)
        self.opti.minimize(obj)

        # self.opti.minimize(cost)
        self.opti.solver("fatrop", self.options)
        try:
            sol = self.opti.solve()
        except:
            sol = self.opti.debug
            print("failed!!")
        # print(sol.value(X).T)
        poses = np.array(sol.value(X))
        poses_xy = poses[:2]
        self.positions = poses_xy.T
        # print(poses)

    def objective_func(self, q, u, q_goal, Q, R, P):
        n = q.shape[1] - 1
        obj = 0
        for i in range(n):
            qi = q[:, i]
            ui = u[:, i]

            term  = ((qi - q_goal).T @ Q @ (qi - q_goal) + ui.T @ R @ ui)
            obj += term

        q_last = q[:, n]
        term_last = (q_last - q_goal).T @ P @ (q_last - q_goal)
        obj += term_last

        return obj
    def plan_to_pose(self, start, goal, dt=0.01, N=1000):
        """
            Uses your self.optimization based path planning algorithm to plan from the 
            start configuration to the goal configuration.

            Args:
                start: starting configuration of the robot.
                goal: goal configuration of the robot.
                dt: Discretization time step. How much time we would like between
                    subsequent time-stamps.
                N: How many waypoints would we like to have in our path from start
                   to goal
        """

        print("======= Planning with self.optimizationPlanner =======")

        # Expand obstacles to account for the radius of the robot.
        with expanded_obstacles(self.config_space.obstacles, self.config_space.robot_radius + 0.05):

            self.plan = None

            q_opt, u_opt = plan_to_pose(np.array(start), np.array(goal), 
                self.config_space.low_lims, self.config_space.high_lims, 
                self.input_low_lims, self.input_high_lims, self.config_space.obstacles, 
                L=self.config_space.robot_length, n=N, dt=dt)

            times = []
            target_positions = []
            open_loop_inputs = []
            t = 0

            for i in range(0, N):
                qi = np.array([q_opt[0][i], q_opt[1][i], q_opt[2][i], q_opt[3][i]])
                ui = np.array([u_opt[0][i], u_opt[1][i]])
                times.append(t)
                target_positions.append(qi)
                open_loop_inputs.append(ui)
                t = t + dt

            # We add one extra step since q_opt has one more state that u_opt
            qi = np.array([q_opt[0][N], q_opt[1][N], q_opt[2][N], q_opt[3][N]])
            ui = np.array([0.0, 0.0])
            times.append(t)
            target_positions.append(qi)
            open_loop_inputs.append(ui)

            self.plan = Plan(np.array(times), np.array(target_positions), np.array(open_loop_inputs), dt)
        return self.plan

    def plot_execution(self):
        """
        Creates a plot of the planned path in the environment. Assumes that the 
        environment of the robot is in the x-y plane, and that the first two
        components in the state space are x and y position. Also assumes 
        plan_to_pose has been called on this instance already.
        """
        ax = plt.subplot(1, 1, 1)
        ax.set_aspect(1)
        ax.set_xlim(self.config_space.low_lims[0], self.config_space.high_lims[0])
        ax.set_ylim(self.config_space.low_lims[1], self.config_space.high_lims[1])

        for obs in self.config_space.obstacles:
            xc, yc, r = obs
            circle = plt.Circle((xc, yc), r, color='black')
            ax.add_artist(circle)

        if True or self.plan:
            plan_x = self.positions[:, 0]
            plan_y = self.positions[:, 1]
            ax.plot(plan_x, plan_y, color='green')

        plt.show()

def main():
    """Use this function if you'd like to test without ROS.
    """
    start = np.array([1, 1, 0, 0]) 
    goal = np.array([9, 9, 0, 0])
    xy_low = [0, 0]
    xy_high = [10, 10]
    phi_max = 0.6
    u1_max = 2
    u2_max = 3
    obstacles = [[6, 3.5, 1.5], [3.5, 6.5, 1]]
    #obstacles = [[4.5, 4.5, 2]]
    config = BicycleConfigurationSpace( xy_low + [-np.inf, -phi_max],
                                        xy_high + [np.inf, phi_max],
                                        [-u1_max, -u2_max],
                                        [u1_max, u2_max],
                                        obstacles,
                                        0.15)

    planner = OptimizationPlanner(config)
    planner.reid_big_function(start, goal, N=50, dt = 0.1)
    planner.plot_execution()
    #plan = planner.plan_to_pose(start, goal)
    #planner.plot_execution()

if __name__ == '__main__':
    main()
