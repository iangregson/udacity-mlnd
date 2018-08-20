import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.init_pos = init_pose if init_pose is not None else np.array([0., 0., 10.])
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

class Takeoff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.timeout = runtime
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.init_pos = init_pose if init_pose is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        done = False
        reward = 0.
        x_reward = 0.
        y_reward = 0.
        z_reward = 0.
        
        x, y, z, phi, theta, psi = self.sim.pose
        init_x, init_y, init_z, init_phi, init_theta, init_psi = self.init_pos
        x_target, y_target, z_target = self.target_pos

        x_delta = abs(x_target - x)
        y_delta = abs(y_target - y)
        z_delta = abs(z_target - z)
        
        x_init_delta = x_target - init_x
        y_init_delta = y_target - init_y
        z_init_delta = z_target - init_z
        
        delta_sum = np.sum([x_delta, y_delta, z_delta])
        
        # if the position is closer to the target than when we started, apply a reward. If not, apply a penalty
        if x_delta < x_init_delta:
            x_reward += (x_target - x_delta)**2
        else:
            x_reward -= (x_target + x_delta)
        if y_delta < y_init_delta:
            y_reward += (y_target - y_delta)**2
        else:
            y_reward -= (y_target + y_delta)
        if z_delta < z_init_delta:
            z_reward += (z_target - z_delta)**2
        else:
            z_reward -= (z_target + z_delta)
        
        reward += np.sum([x_reward, y_reward, z_reward])

        # if we hit the target apply a large bonus and terminate
        if int(round(delta_sum)) == 0:
            reward += 100.
            done = True
    
        # CODE REVIEW: subtract angular velocity to encourage copter to fly straight up
        angular_v = np.sum(self.sim.angular_v)
        reward -= angular_v
        
        # CODE REVIEW: add z velocity to reward to encourage copter to move toward target
        xv, yv, zv = self.sim.v
        reward += zv
        
        # CODE REVIEW: clip reward to between 1 and -1
        reward = np.clip([reward], -1., 1.)[0]
        
        return reward, done
    

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0.
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            r, d = self.get_reward()
            reward += r
            if d == True:
                done = True
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state