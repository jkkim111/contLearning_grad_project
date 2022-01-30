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

class BaxterTestingEnv1():
    def __init__(self, env, task='pick', continuous=False,
                 render=True, using_feature=False,
                 random_spawn=True, rgbd=False, print_on=False, action_type='2D', \
                 viewpoint1 = 'wholeview',viewpoint2 = 'rlview2',viewpoint3 = "birdview", grasping_env = 1,
                 down_level = 2, is_test = True, arm_near_block=False,
                 only_above_block = False, down_grasp_combined = False):
        #put env, task, cont, rgbd values
        self.env = env
        self.task = task # 'reach', 'push', 'pick' or 'place'
        self.is_continuous = continuous
        self.rgbd = rgbd
        self.print_on = print_on
        self.action_type = action_type #'2D' # or '3D'
        self.down_grasp_combined = down_grasp_combined

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
                action_size = 10+ grasping_env + (down_level-1)
            elif task=='place':
                action_size = 10
            self.action_space = spaces.Discrete(action_size)
            self.action_size = action_size

        # define mov_dist, state, grasp, curr_main_pos, obj_pos, target_pos,
        # random_spawn, render, using_feature global_done etc
        self.state = None
        self.grasp = None
        self.curr_main_pos  = None
        self.obj_pos = None
        self.target_pos = None
        self.viewpoint1 = viewpoint1
        self.viewpoint2 = viewpoint2
        if viewpoint3 == "None":
            self.viewpoint3 = None
        else:
            self.viewpoint3 = viewpoint3
        self.main_quat = None
        self.target_quat = None
        self.spawn_range = .15
        self.threshold = .12
        self.only_near_block = True
        self.only_above_block = only_above_block

        self.random_spawn = random_spawn
        self.render = render
        self.using_feature = using_feature

        self.min_reach_dist = None  # for 'reach' task
        self.grasping_env = grasping_env
        self.down_level= down_level
        self.global_done = False
        self.is_test = is_test
        if self.only_near_block: #main block nearby
            self.block_rim_prob = 0.0
        else:
            self.block_rim_prob = 0.2

        if not self.is_test and arm_near_block:
            self.arm_rim_prob = 1.0 #change this to collect data of init_pos around the block near
        else:
            self.arm_rim_prob = 0.0

        self.mov_dist = 0.02 if self.task == 'pick' else 0.04
        self.max_step = 80 # int(1.60/self.mov_dist)


    def reset(self):
        # resetting the BaxterEnv and state - former 3 are the left arm(from robot's perspective)
        self.arena_pos = self.env.env.mujoco_arena.bin_abs
        self.state = np.array([0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.4, -0.9, 1.0, 0.0, 0.0, 0.0]) #np.array([0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.4, -0.6, 1.0, 0.0, 0.0, 0.0])
        self.grasp = 0.0
        self.env.reset() # start MujocoEnv.reset()")
        self.step_count = 0

        # SET MAIN_POS and goal randomly
        spawn_range = self.spawn_range
        threshold = self.threshold  #0.20
        spawn_count = 1

        # randomly spawn until goal and main_pos satisfy conditions
        faraway = np.random.uniform(0.0, 1.0)
        while True:
            spawn_count += 1
            randomness = np.array([spawn_range, spawn_range, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3)
            self.main_pos = self.arena_pos + np.array([0.0, 0.0, 0.05])+ randomness
            self.goal = np.array([-self.main_pos[0],-self.main_pos[1], self.main_pos[2]])

            # change init_pos for every 10 spawns
            if spawn_count % 10 == 0:
                randomness = np.array([spawn_range, spawn_range, 0.0]) * np.random.uniform(low=-1.0, high=1.0, size=3)
                self.main_pos = self.arena_pos + np.array([0.0, 0.0, 0.05]) + randomness

            #break acording to self.block_rim_prob
            if faraway > self.block_rim_prob:
                if np.linalg.norm(self.main_pos[:2] - self.arena_pos[:2]) < threshold and np.linalg.norm(self.main_pos[:2] - self.arena_pos[:2]) > 0.3 * threshold:
                    break
            else:
                if np.linalg.norm(self.main_pos[:2] - self.arena_pos[:2]) > threshold:
                    break

        #CREATE BLOCK
        main_block = self.env.model.worldbody.find("./body[@name='CustomObject_0']")
        main_block.set("pos", array_to_string(self.main_pos))
        target_point = self.env.model.worldbody.find("./body[@name='target']")
        if self.env.num_objects==2:
            target_block = self.env.model.worldbody.find("./body[@name='CustomObject_1']")
            target_block.set("pos", array_to_string(self.goal))
            target_point.find("./geom[@name='target']").set("rgba", "1 0 0 0")
            target = target_block
        elif self.env.num_objects==1:
            self.goal -= np.array([0.0, 0.0, 0.03])
            target_point.set("pos", array_to_string(self.goal))
            if self.task=='reach' or self.task=='pick':
                target_point.find("./geom[@name='target']").set("rgba", "1 0 0 0")
            target = target_point

        # BLOCK QUAT : if pick and place, align main_block's quat, if push and reach, just place random quat
        if self.task=='place':
            angle_tmp = np.random.uniform(low=0, high=np.pi/2, size=1)
            # main_block.set("quat", array_to_string(random_quat()))
            main_block.set("quat", array_to_string(np.array([0.0,0.0, 0.0, 1.0])))

            # target.set("quat", array_to_string(random_quat()))
            target.set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
        elif self.task=='pick':
            if self.grasping_env == 0:
                if  self.env.object_type == "lemon_1":
                    main_block.set("quat", array_to_string(np.array([0.0, 0.0, 0.0, 1.0])))
                else:
                    main_block.set("quat", array_to_string(np.array([1.0, 0.0, 0.0, 1.0])))

            else:
                while True:
                    self.main_quat = random_quat()
                    angle_diff = (self.state[11]-quat2euler(self.main_quat)[0]) % (np.pi)
                    if angle_diff < 3.0*np.pi / 8.0: #not (angle_diff > 3.0*np.pi / 8.0 and angle_diff < 5.0 * np.pi/8.0):
                        break
                self.target_quat = random_quat()
                main_block.set("quat", array_to_string(self.main_quat))
                target.set("quat", array_to_string(self.target_quat))

        # RENDER SETTINGS
        self.env.reset_sims()
        if self.render == 1:
            self.env.viewer.viewer.cam.type = const.CAMERA_FIXED
            self.env.viewer.viewer.cam.fixedcamid = 0

        # GET INIT_POS = init_obj_pos
        self.obj_id = self.env.obj_body_id['CustomObject_0']
        self.curr_main_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])
        self.max_height = self.curr_main_pos[2]

        if self.env.num_objects==2:
            self.target_id = self.env.obj_body_id['CustomObject_1']
            self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
        elif self.env.num_objects==1:
            self.target_pos = self.goal
        self.pre_vec = self.target_pos - self.main_pos

        ## ROBOT ARM INIT POS SETTING ##
        #3. Pick
        elif self.task=='pick':
            if self.random_spawn: #randomly spawn arm position
                while True:
                    randomness = (spawn_range + 0.02) * np.random.uniform(low=-1.0, high=1.0, size=2)
                    self.state[6:8] = self.arena_pos[:2] + randomness
                    if np.linalg.norm(self.state[6:8] - self.arena_pos[:2]) > 0.1 \
                            and np.linalg.norm(self.state[6:8] - self.main_pos[:2], ord=1) < 0.25:
                        break
                if self.down_level == 1:
                    self.state[8] = self.arena_pos[2] + np.random.uniform(low=0.12, high=0.16)
                elif self.down_level == 2:
                    self.state[8] = self.arena_pos[2] + np.random.uniform(low=0.16, high=0.20)

            #arm only above the block
            elif self.only_above_block:
                self.state[6:8] = self.main_pos[:2]
                self.state[8] = self.arena_pos[2] + 0.22
            #arm near the rim with self.arm_rim_prob and near the center with 1-prob
            else:
                centerorrim = np.random.uniform(0.0,1.0, size=1)
                if centerorrim < self.arm_rim_prob:
                    while True:
                        randomness = (spawn_range + 0.02) * np.random.uniform(low=-1.0, high=1.0, size=2)
                        self.state[6:8] = self.arena_pos[:2] + randomness
                        if np.linalg.norm(self.state[6:8] - self.arena_pos[:2])+0.02 < np.linalg.norm(self.arena_pos[:2] - self.main_pos[:2]) \
                                and np.linalg.norm(self.state[6:8] - self.main_pos[:2]) < 0.04:
                            break
                else:
                    self.state[6:8] = self.arena_pos[:2]
                self.state[8] = self.arena_pos[2] + 0.22

        ## move adjust robot arm according to the set init pos ##
        _ = move_to_pos(self.env, [0.4, 0.6, 1.0],self.state[6:9], arm = 'both', level=1.0, render=self.render)
        if self.task == 'push':
            _ = move_to_6Dpos(self.env, self.state[0:3], self.state[3:6], self.state[6:9] + np.array([0., 0., 0.1]), self.state[9:12],
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

        #for reach, add min_reach_dist var
        if self.task == 'reach':
            self.min_reach_dist = np.linalg.norm(self.arm_pos[:2] - self.obj_pos[:2])

        if self.using_feature:
            return np.concatenate([self.state[6:9], self.obj_pos, self.target_pos], axis=0)
        elif self.viewpoint3 is not None:
            im_1, im_2, im_3 = self.get_camera_obs()
            return [im_1, im_2, im_3]
        else:
            im_1, im_2 = self.get_camera_obs()
            return [im_1, im_2]


    def step(self, action):
        self.step_count += 1
        if np.squeeze(action)==-1:
            im_1, im_2 = self.get_camera_obs()
            state = [im_1, im_2]
            reward = 0.0
            done = True
            return state, reward, done, {}

        self.curr_main_pos = np.copy(self.env.sim.data.body_xpos[self.obj_id])

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

            elif action == 8 :
                if not self.down_grasp_combined:
                    self.grasp = 1.0
                    # self.arm_pos = self.arm_pos + np.array([0.0, 0.0, (self.curr_main_pos[2]-self.arm_pos[2])]) #-mov_dist])
                    # self.grasp = 1.0
            elif action == 9:
                if self.down_grasp_combined:
                    self.arm_pos = self.arm_pos + np.array([0.0, 0.0, (self.curr_main_pos[2]-self.arm_pos[2])]) #-mov_dist])
                    self.grasp = 1.0
                else:
                    self.arm_pos = self.arm_pos + np.array([0.0, 0.0, -mov_dist ]) #-mov_dist])
                #    self.arm_pos = self.arm_pos + np.array([0.0, 0.0, -mov_dist*2])
            elif action == 10:
                self.state[11] -= np.pi / 8.0
            elif action == 11:
                self.arm_pos = self.arm_pos + np.array([0.0, 0.0, -mov_dist * 2])
            elif action==12:
                self.grasp = 0.0
            elif action ==13:
                self.arm_pos = self.arm_pos + np.array([0.0, 0.0, mov_dist])
            elif action ==14:
                self.arm_pos = self.arm_pos + np.array([0.0, 0.0, mov_dist*2])
            elif action ==15:
                self.state[11] += np.pi / 8.0

        ## check the arm pos is in the working area ##

        if self.arm_pos[0] < self.arena_pos[0]-self.spawn_range-0.03  or self.arm_pos[0] > self.arena_pos[0]+self.spawn_range+0.03 :
            print('x-axis out of bound')
            stucked = -2
        elif self.arm_pos[1] < self.arena_pos[1]-self.spawn_range-0.03 or self.arm_pos[1] > self.arena_pos[1]+self.spawn_range+.03:
            print('y-axis out of bound')
            stucked = -2
        elif self.arm_pos[2] < self.arena_pos[2] - .2 or self.arm_pos[2] > self.arena_pos[2] + .4:
            print('z-axis out of bound')
            stucked = -2
        elif np.linalg.norm(self.curr_main_pos-self.main_pos) > 0.2 :
            stucked = 4
        else: #############moves the environment according to the actions
            stucked = move_to_6Dpos(self.env, None, None, self.arm_pos, self.state[9:12], arm='right', left_grasp=0.0,
                                right_grasp=0.0, level=1.0, render=self.render) # down and grasp
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
        # print("arm_euler : " + str(arm_euler))
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
                        print("stucked==-1")
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

    ####PICK RL -> When to decide the end of Episode
        elif self.task == 'pick':
            #obj_movement = self.obj_pos - self.pre_obj_pos
            if action == 9:
                # arm_movement = self.arm_pos - self.pre_arm_pos
                # if self.arm_pos[2] < 0.57 and abs(arm_movement[2]) < 0.0105:
                #     print("gripper could not go down")
                #     stucked = 2
                if self.down_grasp_combined:
                    if self.check_grasp():
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                        stucked = 0
                    else:
                        reward = 0.0
                        done = True
                        print('episode done. [Grasp at WRONG MOMENT]')
                        stucked = 1
                else:
                    pass
            if stucked == -1 : # or modify_stucked(arm_euler) :
                reward = 0.0
                done = True
                print('episode done. [STUCKED]')
            elif stucked == 2 :
                reward = 0.0
                done = True
                print('episode done. [Not Going Down]')
            elif stucked == -2:
                reward = 0.0
                done = True
                print('episode done. [Gripper OUT OF BOUNDS -  X/Y/Z]')

            elif stucked == 4:
                reward = 0.0
                done = True
                print('Main Block Not in initial place')
            else:
                # check for grasping success #
                if action== 8:
                    if self.check_grasp():
                        reward = 100
                        done = True
                        print('episode done. [SUCCESS]')
                        stucked = 0
                    else:
                        reward = 0.0
                        done = True
                        print('episode done. [Grasp at WRONG MOMENT]')
                        stucked = 1
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
            print('Episode stopped. [MAX STEP]')
            stucked = 4

        self.global_done = done
        if self.using_feature:
            state = np.concatenate([self.state[6:9], self.obj_pos, self.target_pos], axis=0)
        elif self.viewpoint3 is not None:
            im_1, im_2, im_3 = self.get_camera_obs()
            state = [im_1, im_2, im_3]
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
        return state, reward, done, stucked ,{}

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
            camera_name= 'frontview',
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
            depth=self.env.camera_depth,
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

        # rl view 3
        if self.viewpoint3 is not None:
            camera_obs = self.env.sim.render(
                camera_name=self.viewpoint3,  # "eye_on_left_wrist",
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
                im_3 = np.concatenate((im_rgb, im_depth[..., np.newaxis]), axis=2)
                im_3 = np.flip(im_3, axis=0)
            else:
                im_3 = np.flip(im_rgb, axis=0)

        crop = self.env.crop
        if crop is not None:
            im_1 = im_1[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                  (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]
            im_2 = im_2[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                   (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]
            if self.viewpoint3 is not None:
                im_3 = im_3[(self.env.camera_width - crop) // 2:(self.env.camera_width + crop) // 2, \
                       (self.env.camera_height - crop) // 2:(self.env.camera_height + crop) // 2, :]

        return [im_1, im_2] if self.viewpoint3 is None else [im_1, im_2, im_3]

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
