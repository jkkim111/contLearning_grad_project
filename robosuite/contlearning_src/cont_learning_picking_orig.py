import os,sys
# from behavior_cloning import * #SimpleCNN
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../IL'))
sys.path.append(os.path.join(FILE_PATH, '../scripts'))
sys.path.append(os.path.join(FILE_PATH, '../wrappers'))

import argparse
import numpy as np
import time
import robosuite
from robosuite.wrappers import IKWrapper
import matplotlib.pyplot as plt
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.utils.transform_utils import quat2euler
from new_motion_planner import move_to_6Dpos, get_camera_pos, force_gripper, move_to_pos, get_target_pos, get_arm_rotation, object_pass, stop_force_gripper
from gym import spaces
from mujoco_py.generated import const
import shutil

INIT_ARM_POS = [0.40933302, -1.24377906, 0.68787495, 2.03907987, -0.27229507, 0.8635629,
                0.46484251, 0.12655639, -0.74606415, -0.15337326, 2.04313409, 0.39049096,
                0.30120114, 0.43309788]


class BaxterTestingEnv():
    def __init__(self, env, task='pick', continuous=False,
                 render=True, using_feature=False,
                 random_spawn=True, rgbd=False, print_on=False, action_type='2D', \
                 viewpoint1 = 'wholeview',viewpoint2 = 'rlview2'):
        #put env, task, cont, rgbd values
        self.env = env
        self.task = task # 'reach', 'push', 'pick' or 'place'
        self.is_continuous = continuous
        self.rgbd = rgbd
        self.print_on = print_on
        self.action_type = action_type #'2D' # or '3D'

        # define action dimensions depending on continuous and task
        if self.is_continuous:
            if task=='reach':
                action_dim = 3
            elif task=='push':
                action_dim = 6
            elif task=='pick' or task=='place':
                action_dim = 6
            self.action_space = spaces.Box(-1, 1, [action_dim])
            # action: [x, y, z, cos_th, sin_th, gripper]
            self.action_dim = action_dim
        else:
            if task=='reach':
                if self.action_type=='2D':
                    action_size = 8
                elif self.action_type=='3D':
                    action_size = 10 #8
            elif task=='push':
                if self.action_type=='2D':
                    action_size = 8
                elif self.action_type=='3D':
                    action_size = 10 #12
            elif task=='pick':
                action_size = 12
            elif task=='place':
                action_size = 10
            self.action_space = spaces.Discrete(action_size)
            self.action_size = action_size

        # define mov_dist, state, grasp, init_obj_pos, obj_pos, target_pos,
        # random_spawn, render, using_feature global_done etc
        self.mov_dist = 0.02 if self.task=='pick' else 0.04
        self.state = None
        self.grasp = None
        self.init_obj_pos = None
        self.obj_pos = None
        self.target_pos = None

        self.random_spawn = random_spawn
        self.render = render
        self.using_feature = using_feature
        self.max_step = 50

        self.viewpoint1 = viewpoint1
        self.viewpoint2 = viewpoint2


        self.min_reach_dist = None  # for 'reach' task

        self.global_done = False

    def reset(self):
        # resetting the BaxterEnv
        self.step_count = 0
        arena_pos = self.env.env.mujoco_arena.bin_abs

        # state: former 3 are the left arm(from robot's perspective)
        self.state = np.array([0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.4, -0.9, 1.0, 0.0, 0.0, 0.0]) #np.array([0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.4, -0.6, 1.0, 0.0, 0.0, 0.0])
        self.grasp = 0.0

        #reset grasp, spawn_range, threshold
        if self.task=='reach' or self.task=='push':
            self.grasp = 1.0
        if self.task=='reach':
            spawn_range = 1
            threshold = 0.25
        else:
            spawn_range = 1 #0.15
            threshold = 0.25  #0.20


        self.env.reset() # start MujocoEnv.reset()")

        # resetting init_pos and goal randomly
        # spawn new goal point repetitively until out of threshold
        spawn_count = 1
        while True:  # <0.15
            spawn_count += 1
            randomness = np.array([spawn_range, spawn_range, 0.0]) * np.random.uniform(low=-0.27, high=0.27, size=3)
            init_pos = arena_pos + randomness + np.array([0.0, 0.0, 0.05])
            print("pos: ", str(arena_pos), " + " , str(randomness) )
            # self.goal = arena_pos + np.array([spawn_range, spawn_range, 0.0]) * \
            #             np.random.uniform(low=-1.0, high=1.0, size=3) + np.array([0.0, 0.0, 0.05])  # 0.025
            self.goal = np.array([-init_pos[0],-init_pos[1], init_pos[2]])
            # change init_pos for every 10 spawns
            if spawn_count % 10 == 0:
                init_pos = arena_pos + np.array([spawn_range, spawn_range, 0.0]) \
                           * np.random.uniform(low=-1.0, high=1.0, size=3) \
                           + np.array([0.0, 0.0, 0.05])
            # if np.linalg.norm(self.goal[0:2] - init_pos[0:2]) < threshold:
            if (init_pos[0]-arena_pos[0])**2 + (init_pos[1]-arena_pos[1])**2 > 0.01:
                break

        #placing blocks
        main_block = self.env.model.worldbody.find("./body[@name='CustomObject_0']")
        main_block.set("pos", array_to_string(init_pos))
        target_point = self.env.model.worldbody.find("./body[@name='target']")
        #if env.num_object is 2, make the target block at the goal position,
        #if 1, just the target_point
        if self.env.num_objects==2:
            target_block = self.env.model.worldbody.find("./body[@name='CustomObject_1']")
            target_block.set("pos", array_to_string(self.goal))
            target_point.find("./geom[@name='target']").set("rgba", "0 0 0 0")
            target = target_block
        elif self.env.num_objects==1:
            self.goal -= np.array([0.0, 0.0, 0.03])
            target_point.set("pos", array_to_string(self.goal))
            if self.task=='reach' or self.task=='pick':
                target_point.find("./geom[@name='target']").set("rgba", "0 0 0 0")
            target = target_point

        # if pick and place, align main_block's quat, if push and reach, just place random quat
        if self.task=='pick' or self.task=='place':
            main_block.set("quat", array_to_string(np.array([1.0, 0.0, 0.0, 1.0])))
            target.set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
        else:
            main_block.set("quat", array_to_string(random_quat()))
            target.set("quat", array_to_string(random_quat()))
        #if num_obj is 1, only set the target block
        if self.env.num_objects==1:
            target.set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))

        # print("IN cont_learning_picking.py: BaxterPush.reset_sims() -> lasting window")

        self.env.reset_sims()

        # print("IN cont_learning_picking.py: got back from BaxterPush.reset_sims()")

        if self.render == 1:
            self.env.viewer.viewer.cam.type = const.CAMERA_FIXED
            self.env.viewer.viewer.cam.fixedcamid = 0

        self.obj_id = self.env.obj_body_id['CustomObject_0']
        self.init_obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        self.obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        self.max_height = self.init_obj_pos[2]

        # self.target_id = self.env.obj_body_id['CustomObject_1']
        # self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
        if self.env.num_objects==2:
            self.target_id = self.env.obj_body_id['CustomObject_1']
            self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
        elif self.env.num_objects==1:
            self.target_pos = self.goal
        self.pre_vec = self.target_pos - self.obj_pos

        # print("start the task") ########
        ## set robot init state ##
        if self.task=='reach':
            if self.action_type=='2D':
                self.state[8] = arena_pos[2] + 0.08 * np.random.uniform(low=0.47, high=1.0)
                if self.random_spawn:
                    self.state[6:8] = arena_pos[:2] + spawn_range * np.random.uniform(low=-1.0, high=1.0, size=2)
                else:
                    self.state[6:8] = arena_pos[:2]

            elif self.action_type=='3D':
                if self.random_spawn:
                    self.state[6:8] = arena_pos[:2] + spawn_range * np.random.uniform(low=-1.0, high=1.0, size=2)
                    self.state[8] = arena_pos[2] + 0.14 + 0.06 * np.random.uniform(low=-1.0, high=1.0)
                else:
                    self.state[6:8] = arena_pos[:2]
                    self.state[8] = arena_pos[2] + 0.14  # 0.16
                # if self.random_spawn:
                #     self.state[6:8] = arena_pos[:2] + spawn_range * np.random.uniform(low=-1.0, high=1.0, size=2)

        elif self.task=='push':
            #####change######
            #the initial angle of the gripper aligns vertically to the pre_vec
            align_direction = self.pre_vec[:2] / np.linalg.norm(self.pre_vec[:2])
            self.state[6:8] = self.obj_pos[:2] - 0.08 * align_direction
            align_angle = np.arctan2(align_direction[1],align_direction[0])
            align_angle_gripper=[]
            for a in range(8):
                align_angle_gripper.append(abs(a * np.pi / 4.0-align_angle))
            self.state[11] = -align_angle#-np.argmin(align_angle_gripper)*np.pi/4;

            #initial position of (x,y,z) = colinear with object and target.

            if self.action_type == '2D':
                if self.random_spawn:
                    self.state[8] = arena_pos[2] + np.random.uniform(low=0.47, high=1.0) * 0.075
                else:
                    self.state[8] = arena_pos[2] + 0.055

            elif self.action_type == '3D':
                if self.random_spawn:
                    self.state[8] = arena_pos[2] + 0.14 + 0.04 * np.random.uniform(low=-1.0, high=1.0)
                else:
                    self.state[8] = arena_pos[2] + 0.14

        #Init Position for Pick and Place
        elif self.task=='pick':
            if self.random_spawn:
                self.state[6:8] = arena_pos[:2] + spawn_range * np.random.uniform(low=-1.0, high=1.0, size=2)
                self.state[8] = arena_pos[2] + 0.14 #+ 0.04 * np.random.uniform(low=-1.0, high=1.0)
            else:
                self.state[6:8] = arena_pos[:2]
                self.state[8] = arena_pos[2] + 0.14

        elif self.task=='place':
            self.state[6:9] = self.obj_pos + np.array([0.0, 0.0, 0.10])

        elif self.task=='pickNplace':
            if self.random_spawn:
                self.state[6:8] = self.obj_pos[:2] + 2 * self.mov_dist * np.random.uniform(low=-1.0, high=1.0, size=2)
                self.state[8] = arena_pos[2] + 0.14 + 0.04 * np.random.uniform(low=-1.0, high=1.0)
            else:
                self.state[6:8] = self.obj_pos[:2] + self.mov_dist * np.random.uniform(low=-1.0, high=1.0, size=2)
                self.state[8] = arena_pos[2] + 0.14

        ## move robot arm to init pos ##
        _ = move_to_pos(self.env, [0.4, 0.6, 1.0], [0.4, -0.6, 1.0], arm='both', level=1.0, render=self.render)
        if self.task == 'push':
            _ = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9] + np.array([0., 0., 0.1]), self.state[9:12],
                                    arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)
        _ = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12],
                                arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)
        if self.task=='place':
            _ = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.obj_pos, self.state[9:12],
                              arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)
            self.grasp = 1.0
            _ = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.obj_pos, self.state[9:12],
                              arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)
            _ = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12],
                                arm='both', left_grasp=0.0, right_grasp=self.grasp, level=1.0, render=self.render)

        self.pre_obj_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        if self.env.num_objects==2:
            self.pre_target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
        elif self.env.num_objects==1:
            self.pre_target_pos = self.goal

        self.arm_pos = self.env._r_eef_xpos
        self.pre_arm_pos = self.arm_pos.copy()
        self.global_done = False

        if self.task == 'reach':
            self.min_reach_dist = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])

        if self.using_feature:
            return np.concatenate([self.state[6:9], self.obj_pos, self.target_pos], axis=0)
        else:
            im_1, im_2 = self.get_camera_obs()
            ## visualizing observations ##
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(im_1)
            # ax[1].imshow(im_2)
            # plt.show()
            return [im_1, im_2]


    def step(self, action):
        self.step_count += 1
        if np.squeeze(action)==-1:
            im_1, im_2 = self.get_camera_obs()
            state = [im_1, im_2]
            reward = 0.0
            done = True
            return state, reward, done, {}

        if self.is_continuous:
            # [dx, dy, dz] + [cos(theta), sin(theta), grasp]
            action = np.squeeze(action)
            position_change = action[:3] / 20.0
            self.arm_pos = self.arm_pos + position_change

            if self.task == 'push' or self.task == 'pick':
                cos_theta = action[3]
                sin_theta = action[4]
                grasp = (action[5] + 1.)/2.
                theta = np.arctan2(sin_theta, cos_theta)
                self.state[11] = theta
                self.grasp = grasp
        else:
            # 8 directions
            # up / down
            # gripper close / open
            action = np.squeeze(action) #action[0][0]
            # assert action < self.action_size
            mov_dist = self.mov_dist

            self.pre_arm_pos = self.arm_pos.copy()
            if action < 8:
                mov_degree = action * np.pi / 4.0
                self.arm_pos = self.arm_pos + np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
            elif action == 8:
                self.arm_pos = self.arm_pos + np.array([0.0, 0.0, mov_dist])
            elif action == 9:
                self.arm_pos = self.arm_pos + np.array([0.0, 0.0, -mov_dist])
            elif action == 10:
                self.grasp = 1.0
            elif action == 11:
                self.grasp = 0.0

        ## check the arm pos is in the working area ##
        if self.arm_pos[0] < -1.5 or self.arm_pos[0] > 2.5:
            print('stuck-type1')
            stucked = -1
        elif self.arm_pos[1] < -0.62 or self.arm_pos[1] > 0.06:
            print('stuck-type2')
            stucked = -1
        else:
            stucked = move_to_6Dpos(self.env, None, None, self.arm_pos, self.state[9:12], arm='right', left_grasp=0.0,
                                right_grasp=self.grasp, level=1.0, render=self.render)
        self.state[6:9] = self.env._r_eef_xpos
        self.arm_pos = self.state[6:9]
        self.obj_pos = self.env.sim.data.body_xpos[self.obj_id]

        self.pre_target_pos = self.target_pos.copy()
        if self.env.num_objects==2:
            self.target_pos = self.env.sim.data.body_xpos[self.target_id]
        elif self.env.num_objects==1:
            self.target_pos = self.goal
        vec = self.target_pos - self.obj_pos

        done = False
        reward = 0.0
        arm_euler = quat2euler(self.env.env._right_hand_quat)
        if self.task == 'reach':
            # if stucked == -1 or #1 - np.abs(self.env.env._right_hand_quat[1]) > 0.01:
            if stucked == -1 or check_stucked(arm_euler):
                reward = 0.0 #np.exp(-1.0 * np.min([np.linalg.norm(self.state[6:9]-self.obj_pos), np.linalg.norm(self.state[6:9]-self.target_pos)]))
                done = True
                print('episode done. [STUCKED]')
            else:
                d1 = np.linalg.norm(self.arm_pos - self.obj_pos)
                # d1 = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])

                # if d1 < self.mov_dist / 2:  # or d2 < 0.025:
                if np.linalg.norm(self.obj_pos - self.pre_obj_pos) > 0.005:
                    reward = 100
                    done = True
                    print('episode done. [SUCCESS]')

                ## sparse reward ##
                # elif d1 < self.min_reach_dist - 0.001:
                #     self.min_reach_dist = d1
                #     reward = 1.0
                # elif self.arm_pos[2] > self.env.env.mujoco_arena.bin_abs[2] + 0.18:
                #     reward = -0.2
                else:
                    pass # reward = -0.1

        elif self.task == 'push':
            goal_threshold = 0.10 if self.env.num_objects == 2 else 0.05
            if self.action_type=='2D':
                def get_cos(vec1, vec2):
                    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                vec_target_obj = self.target_pos - self.obj_pos
                vec_obj_arm = self.obj_pos - self.arm_pos
                # if stucked == -1 or #1 - np.abs(self.env.env._right_hand_quat[1]) > 0.01:
                if stucked == -1 or check_stucked(arm_euler):
                    if stucked ==-1 :
                        print("stuckec==-1")
                    else :
                        print("check_Stucked")
                    reward = 0.0  # -10
                    done = True
                    print('episode done. [STUCKED]')
                else:
                    x = np.linalg.norm(vec)
                    x_old = np.linalg.norm(self.pre_vec)
                    d1 = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])
                    # d1_old = np.linalg.norm(self.pre_arm_pos[:2] - self.pre_obj_pos[:2])

                    if np.linalg.norm(vec) < goal_threshold: #0.05
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                    elif get_cos(vec_target_obj[:2], vec_obj_arm[:2]) < 0 and np.linalg.norm(vec_obj_arm[:2]) > 0.02:
                        reward = 0.0
                        done = True
                    # get away #
                    elif d1 > 0.4:
                        done = True
                        pass # reward = -5
                    elif d1 > 2 * self.mov_dist:
                        pass # reward = -0.5
                    # moving distance reward #
                    elif x_old - x > 0.01:
                        pass # reward = 2.0 # 100 * (x_old - x)
                    # touching reward #
                    elif np.linalg.norm(self.obj_pos - self.pre_obj_pos) > 0.01:
                        pass # reward = 1.0
                    # step penalty #
                    else:
                        pass # reward = 0.0

                    self.pre_vec = vec

            elif self.action_type=='3D':
                # if stucked == -1 or #1 - np.abs(self.env.env._right_hand_quat[1]) > 0.01:
                if stucked == -1 or check_stucked(arm_euler):
                    reward = 0.0 #-10
                    done = True
                    print('episode done. [STUCKED]')
                else:
                    x = np.linalg.norm(vec)
                    x_old = np.linalg.norm(self.pre_vec)
                    d1 = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])
                    d1_old = np.linalg.norm(self.pre_arm_pos[:2] - self.pre_obj_pos[:2])

                    if np.linalg.norm(vec) < goal_threshold: #0.05
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                    # get away #
                    elif d1 > 0.4:
                        done = True
                        pass # reward = -5
                    elif d1 > 2 * self.mov_dist:
                        pass # reward = -0.5
                    # moving distance reward #
                    elif x_old - x > 0.01:
                        pass # reward = 2.0 # 100 * (x_old - x)
                    # touching reward #
                    elif np.linalg.norm(self.obj_pos - self.pre_obj_pos) > 0.01:
                        pass # reward = 1.0
                    # step penalty #
                    else:
                        pass # reward = 0.0

                    self.pre_vec = vec

    ####PICK RL
        elif self.task == 'pick':
            #obj_movement = self.obj_pos - self.pre_obj_pos
            if action ==9:
                arm_movement = self.arm_pos - self.pre_arm_pos
                if self.arm_pos[2] < 0.02 and abs(arm_movement[2]) <0.0105 :
                    print("gripper could not go down")
                    stucked = -1

            if stucked == -1 : #or modify_stucked(arm_euler) :
                reward = 0.0
                done = True
                print('episode done. [STUCKED]')

            else:
                # check for grasping success #
                if action==10:
                    if self.check_grasp():
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                    else:
                        reward = 0.0
                        done = True
                        print('episode done. [Failed]')
                # check for picking up the block #
                # if self.obj_pos[2] - self.init_obj_pos[2] > self.mov_dist / 2:
                #     reward = 100
                #     done = True
                #     print('episode done. [SUCCESS]')

        elif self.task == 'place':
            goal_threshold = 0.10 if self.env.num_objects == 2 else 0.04
            if stucked == -1 or check_stucked(arm_euler):
                reward = 0.0
                done = True
            else:
                # check for placing the block #
                if self.env.num_objects==1:
                    if np.linalg.norm(self.obj_pos[:2] - self.target_pos[:2]) < self.mov_dist/2 and self.obj_pos[2] - self.target_pos[2] < goal_threshold:
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                elif self.env.num_objects == 2:
                    if self.check_contact():
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')

        if not done and self.step_count >= self.max_step:
            done = True
            print('Episode stopped. (max step)')

        self.global_done = done
        if self.using_feature:
            state = np.concatenate([self.state[6:9], self.obj_pos, self.target_pos], axis=0)
        else:
            im_1, im_2 = self.get_camera_obs()
            state = [im_1, im_2]

        ### testing ###
        # for contact in self.env.env.sim.data.contact[0 : self.env.env.sim.data.ncon]:
        #     name1 = self.env.env.sim.model.geom_id2name(contact.geom1)
        #     name2 = self.env.env.sim.model.geom_id2name(contact.geom2)
        #     print(contact, name1, name2)
        ### ###

        if self.print_on:
            print('action:', action, '\t/  reward:', reward)
        return state, reward, done, {}

    def check_grasp(self):
        contact_tipr, contact_tipl = False, False
        for contact in self.env.env.sim.data.contact[0 : self.env.env.sim.data.ncon]:
            name1 = self.env.env.sim.model.geom_id2name(contact.geom1)
            name2 = self.env.env.sim.model.geom_id2name(contact.geom2)
            if 'CustomObject_0' in (name1, name2):
                if 'r_fingertip_g0' in (name1, name2):
                    contact_tipr = True
                if 'l_fingertip_g0' in (name1, name2):
                    contact_tipl = True
        if contact_tipr and contact_tipl:
            return True

    def check_contact(self):
        for contact in self.env.env.sim.data.contact[0 : self.env.env.sim.data.ncon]:
            name1 = self.env.env.sim.model.geom_id2name(contact.geom1)
            name2 = self.env.env.sim.model.geom_id2name(contact.geom2)
            if 'CustomObject_0' in (name1, name2) and 'CustomObject_1' in (name1, name2):
                return True
        return False

    def get_camera_obs(self):
        # GET CAMERA IMAGE
        # prepare for rendering #
        _ = self.env.sim.render(
            camera_name="birdview",
            width=10, #self.env.camera_width,
            height=10, #self.env.camera_height,
            depth=False, #self.env.camera_depth
            # mode="window",
            # device_id = 2
        )

        # rl view 1
        camera_obs = self.env.sim.render(
            camera_name=self.viewpoint1,  # "birdview", #"rlview1",
            width=self.env.camera_width,
            height=self.env.camera_height,
            depth=True, #self.env.camera_depth,
            # mode="window",
            # device_id = 2
        )
        rgb, ddd = camera_obs
        # rgb = camera_obs

        extent = self.env.mjpy_model.stat.extent
        near = self.env.mjpy_model.vis.map.znear * extent
        far = self.env.mjpy_model.vis.map.zfar * extent

        im_depth = near / (1 - ddd * (1 - near / far))
        im_rgb = rgb / 255.0
        if self.rgbd:
            im_1 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)
            im_1 = np.flip(im_1, axis=0)
        else:
            im_1 = np.flip(im_rgb, axis=0)

        # rl view 2
        camera_obs = self.env.sim.render(
            camera_name=self.viewpoint2, #"eye_on_left_wrist",
            width=self.env.camera_width,
            height=self.env.camera_height,
            depth=self.env.camera_depth
        )
        rgb, ddd = camera_obs

        extent = self.env.mjpy_model.stat.extent
        near = self.env.mjpy_model.vis.map.znear * extent
        far = self.env.mjpy_model.vis.map.zfar * extent

        im_depth = near / (1 - ddd * (1 - near / far))
        im_rgb = rgb / 255.0
        if self.rgbd:
            im_2 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)
            im_2 = np.flip(im_2, axis=0)
        else:
            im_2 = np.flip(im_rgb, axis=0)

        crop = self.env.crop
        if crop is not None:
            im_1 = im_1[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                  (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]
            im_2 = im_2[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                   (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]

        return [im_1, im_2]


def check_stucked(arm_euler):
    # print(np.array(arm_euler) / np.pi)
    check1 = arm_euler[0] % np.pi < 0.02 or np.pi - arm_euler[0] % np.pi < 0.02
    # check2 = arm_euler[1] % np.pi < 0.02 or np.pi - arm_euler[1] % np.pi < 0.02
    return not check1 #or not check2

def modify_stucked(arm_euler):
    # print(np.array(arm_euler) / np.pi)
    check1 = arm_euler[0] % np.pi < 0.01 or np.pi - arm_euler[0] % np.pi < 0.01
    check2 = arm_euler[1] % np.pi < 0.01 or np.pi - arm_euler[1] % np.pi < 0.01
    #check_stucked  = false when check1 and check2 is both true
    #check1 =true , check2 = true -> check_stucked -> false
    #otherwise check_stucked -> true
    #success = check1 and check2 is true.
    return not (check1 and check2)

def random_quat():
    yaw, pitch, roll = 2 * np.pi * np.array([np.random.rand(), 0.0, 0.0])
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cr = np.cos(roll/2)
    sr = np.sin(roll/2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    q = np.array([qw, qx, qy, qz])
    return q
    '''
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array((np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2), dtype=np.float32)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=0)
    parser.add_argument(
        '--num-objects', type=int, default=2)
    parser.add_argument(
        '--num-episodes', type=int, default=10000)
    parser.add_argument(
        '--num-steps', type=int, default=1)
    parser.add_argument(
        '--render', type=bool, default=False)
    parser.add_argument(
        '--bin-type', type=str, default="table")  # table, bin, two
    parser.add_argument(
        '--object-type', type=str,
        default="cube")  # T, Tlarge, L, 3DNet, stick, round_T_large
    parser.add_argument(
        '--test', type=bool, default=False)
    parser.add_argument(
        '--config-file', type=str, default="config_example.yaml")
    parser.add_argument(
        '--task', type=str, default="place")

    args = parser.parse_args()
    # print(args.render)
    np.random.seed(args.seed)

    screen_width = 192
    screen_height = 192
    crop = 128
    env = robosuite.make(
        "BaxterPush",
        bin_type='table',
        object_type='cube',
        ignore_done=True,
        has_renderer=bool(args.render),  # True,
        has_offscreen_renderer=not bool(args.render),  # added this new line
        camera_name="eye_on_right_wrist",
        gripper_visualization=False,
        use_camera_obs=False,
        use_object_obs=False,
        camera_depth=True,
        num_objects=2,
        control_freq=100,
        camera_width=screen_width,
        camera_height=screen_height,
        crop=crop
    )
    env = IKWrapper(env)

    render = args.render
    print(args)
    rl_env = BaxterTestingEnv1(env, task=args.task, render=render, using_feature=False, action_type='3D', random_spawn=True)

    print('befor resetting env')
    rl_env.reset()
    print('after resetting env')
    for i in range(5):
        rl_env.step(np.random.randint(rl_env.action_size))
        print(str(i)+ 'th iteration steps')
    # exit()
    ## test for object center position ##
    for _idx in range(5):
        rl_env.reset()
        print(str(i) + 'th iteration tests')
        for _idx2 in range(2):
            move_to_6Dpos(env, rl_env.state[0:3], rl_env.state[3:6], rl_env.obj_pos + np.array([0., 0., 0.1]),
                          rl_env.state[9:12], arm='both', left_grasp=0.0, right_grasp=rl_env.grasp, level=1.0,
                          render=render)
            move_to_6Dpos(env, rl_env.state[0:3], rl_env.state[3:6], rl_env.obj_pos + np.array([0., 0., 0.15]),
                          rl_env.state[9:12], arm='both', left_grasp=0.0, right_grasp=rl_env.grasp, level=1.0,
                          render=render)
            move_to_6Dpos(env, rl_env.state[0:3], rl_env.state[3:6], rl_env.goal + np.array([0., 0., 0.1]),
                          rl_env.state[9:12], arm='both', left_grasp=0.0, right_grasp=rl_env.grasp, level=1.0,
                          render=render)
            move_to_6Dpos(env, rl_env.state[0:3], rl_env.state[3:6], rl_env.goal + np.array([0., 0., 0.15]),
                          rl_env.state[9:12], arm='both', left_grasp=0.0, right_grasp=rl_env.grasp, level=1.0,
                          render=render)
