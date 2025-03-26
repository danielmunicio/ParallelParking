#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Amay Saxena
"""
import numpy as np
import sys

import tf2_ros
import tf
from std_srvs.srv import Empty as EmptySrv
import rospy
from proj2_pkg.msg import BicycleCommandMsg, BicycleStateMsg
from proj2.planners import SinusoidPlanner, RRTPlanner, BicycleConfigurationSpace
from proj2.controller.mpc_controller import MPCPathFollower

class BicycleModelController(object):
    def __init__(self):
        """
        Executes a plan made by the planner
        """
        self.pub = rospy.Publisher('/bicycle/cmd_vel', BicycleCommandMsg, queue_size=10)
        self.sub = rospy.Subscriber('/bicycle/state', BicycleStateMsg, self.subscribe)
        self.state = BicycleStateMsg()
        if not rospy.has_param("/bicycle_converter/converter/length"):
            raise ValueError("Converter length not found on parameter server")    
        L = rospy.get_param("/bicycle_converter/converter/length")
        if not rospy.has_param("/bicycle_converter/converter/max_steering_rate"):
            raise ValueError("Max Steering Rate not found on parameter server")
        DF_MAX = rospy.get_param("/bicycle_converter/converter/max_steering_rate")
        if not rospy.has_param("/bicycle_converter/converter/max_linear_velocity"):
            raise ValueError("Max Linear Velocity not found on parameter server")
        V_MAX = rospy.get_param("/bicycle_converter/converter/max_linear_velocity")
        if not rospy.has_param("/bicycle_converter/converter/max_steering_angle"):
            raise ValueError("No robot information loaded on parameter server. Did you run init_env.launch?")
        PHI_MAX = rospy.get_param("/bicycle_converter/converter/max_steering_angle")
        self.mpc = MPCPathFollower(
            N=20, 
            DT=0.05, 
            L_F=L/2,
            L_R=L/2,
            V_MIN=-PHI_MAX, # min steering angle
            V_MAX=PHI_MAX, # max steering angle
            A_MIN=-V_MAX, # min fwd vel
            A_MAX=V_MAX, # max fwd vel
            A_DOT_MIN=-np.inf, # min fwd accel
            A_DOT_MAX=np.inf, # max fwd acc
            DF_MIN=-DF_MAX, # min steering vel
            DF_MAX=DF_MAX, # max steering vel
            DF_DOT_MIN=-np.inf, # min steering acceleration
            DF_DOT_MAX=np.inf, # max steering acceleration
            Q=[10, 10, 0, 0], # weight cost matrix
            R=[0, 0], # input cost matrix
            RUNTIME_FREQUENCY=20, # runtime frequency (hz)
            ivpsolver=dict(n=3, method='rk4'),
            nlpsolver=dict(opts={'structure_detection': 'auto', 'expand': False, 'debug': False, 'fatrop.print_level': -1, 'print_time': False}, name='fatrop')
        )
        rospy.on_shutdown(self.shutdown)
<<<<<<< HEAD
=======
        self.last_state = None
        self.last_steering_angle = None
        self.lookahead = 0.3
>>>>>>> 3d04efa (Stuff isn't working >:()

    def execute_plan(self, plan):
        """
        Executes a plan made by the planner

        Parameters
        ----------
        plan : :obj: Plan. See configuration_space.Plan
        """
        if len(plan) == 0:
            return
        rate = rospy.Rate(int(1 / plan.dt))
        start_t = rospy.Time.now()
        self.start_t = start_t
        self.plan = plan
        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_t).to_sec()
            if t > plan.times[-1]:
                break
            state, cmd = plan.get(t)
            self.step_control(state, cmd)
            rate.sleep()
        self.cmd([0, 0])

    def step_control(self, target_position, open_loop_input):
        """Specify a control law. For the grad/EC portion, you may want
        to edit this part to write your own closed loop controller.
        Note that this class constantly subscribes to the state of the robot,
        so the current configuratin of the robot is always stored in the 
        variable self.state. You can use this as your state measurement
        when writing your closed loop controller.

        Parameters
        ----------
            target_position : target position at the current step in
                              [x, y, theta, phi] configuration space.
            open_loop_input : the prescribed open loop input at the current
                              step, as a [u1, u2] pair.
        Returns:
            None. It simply sends the computed command to the robot.
        """
        t0 = rospy.Time.now().to_sec() - self.start_t.to_sec()
<<<<<<< HEAD
        targets = list(zip(*[self.plan.get(t0 + i*self.mpc.DT) for i in range(self.mpc.N)]))
        x_traj = np.array(targets[0])
        u_traj = np.array(targets[1])
        print(x_traj, u_traj)
        # print(x_traj.shape)
        # print(u_traj.shape)
        # x_traj = self.path[xidxs, 0:4]
        # u_traj = self.path[uidxs, 4:6]
        trajectory = np.hstack([x_traj, u_traj]).T
        if 'prev_soln' not in self.__dict__: self.prev_soln = np.array([0.0, 0.0])
        self.prev_soln = self.mpc.solve(self.state, self.prev_soln, trajectory).flatten()
        self.cmd(self.prev_soln)
        print(self.prev_soln)
        print(self.state)
        print(target_position, open_loop_input)
        print()
=======
        dt = t0 - self.last_t
        error = target_position - self.state
        error_dist = np.linalg.norm(error[:2])
        
        p = 0.2 * error_dist
        
        dx, dy = target_position[0] - self.state[0], target_position[1] - self.state[1]
        L = np.hypot(dx, dy)
        alpha = np.arctan2(dy, dx) - self.state[2]
        
        steering_angle = np.arctan2(2 * L * np.sin(alpha), self.lookahead)
        
        if self.last_steering_angle is not None:
            steering_velocity = (steering_angle - self.last_steering_angle) / dt
        else:
            steering_velocity = 0
        
        print(f"Target Position: {target_position}")
        print(f"Current State: {self.state}")
        print(f"Distance to Target (L): {L}")
        print(f"Angle Error (alpha): {alpha}")
        print(f"Steering Angle: {steering_angle}")
        print(f"Steering Velocity: {steering_velocity}")
        print(f"Velocity (p): {p}")
        
        control_out = [open_loop_input[0], open_loop_input[1]]
        
        # Send command
        self.cmd(control_out)
        
        # Update last state and time
        self.last_t = t0
        self.last_state = self.state.copy()
        self.last_steering_angle = steering_angle
>>>>>>> 3d04efa (Stuff isn't working >:()

    def cmd(self, msg):
        """
        Sends a command to the turtlebot / turtlesim

        Parameters
        ----------
        msg : numpy.ndarray
        """
        self.pub.publish(BicycleCommandMsg(*msg))

    def subscribe(self, msg):
        """
        callback fn for state listener.  Don't call me...
        
        Parameters
        ----------
        msg : :obj:`BicycleStateMsg`
        """
        self.state = np.array([msg.x, msg.y, msg.theta, msg.phi])

    def shutdown(self):
        rospy.loginfo("Shutting Down")
        self.cmd((0, 0))
