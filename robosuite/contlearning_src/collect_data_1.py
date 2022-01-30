import os,sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=int, default=1)
parser.add_argument('--task', type=str, default="pick")
parser.add_argument('--save_data', type=int, default=1)
parser.add_argument('--save_view', type=int, default=1)
parser.add_argument('--max_buff', type=int, default=256)  # 512
parser.add_argument('--viewpoint1', type=str, default="rlview1")
parser.add_argument('--viewpoint2', type=str, default="rlview2")
parser.add_argument('--viewpoint3', type=str, default=None)
parser.add_argument('--data_loc', type=str, default="data")
parser.add_argument('--reset_loc', type=int, default=0)
parser.add_argument('--grasp_env', type=int, default=1)
parser.add_argument('--down_level', type=int, default=2)
parser.add_argument('--arm_near_block', type=int, default=0)
parser.add_argument('--down_grasp_combined', type=int, default=0)
parser.add_argument('--only_above_block', type=int, default=0)

parser.add_argument('--obj', type=str, default="smallcube")
parser.add_argument('--num_episodes', type=int, default=1500)
parser.add_argument('--use_feature', type=int, default=0)
parser.add_argument('--action_type', type=str, default="2D")
parser.add_argument('--random_spawn', type=int, default=0)
parser.add_argument('--block_random', type=int, default=1)
# parser.add_argument('--small_cube', type=int, default=1)
parser.add_argument('--num_blocks', type=int, default=1)
parser.add_argument('--num_data_target', type=int, default=4000)
args = parser.parse_args()

#mjviewer_rendering_related_environ
os.environ["CYMJ_RENDER"] = "0" if not args.render else "1"

import datetime
# from behavior_cloning import * #SimpleCNN
import pickle, csv
# import tensorflow as tf
import robosuite
from ik_wrapper import IKWrapper
from cont_learning_picking import BaxterTestingEnv1
from robosuite.utils.transform_utils import quat2euler
import numpy as np
from PIL import Image
import shutil
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def main():
    # camera resolution -> image resolution
    screen_width = 192  # 64
    screen_height = 192  # 64
    crop = 128  # None

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU is not efficient here

    using_feature = bool(args.use_feature)
    if using_feature:
        print('This agent will use feature-based states..!!')
    else:
        print('This agent will use image-based states..!!')
    render = bool(args.render)
    save_data = bool(args.save_data)
    print_on = False
    task = args.task
    action_type = args.action_type
    random_spawn = bool(args.random_spawn)
    block_random = bool(args.block_random)

    num_blocks = args.num_blocks
    arm_near_block = bool(args.arm_near_block)

    save_view = bool(args.save_view)
    viewpoint1 = args.viewpoint1
    viewpoint2 = args.viewpoint2
    viewpoint3 = args.viewpoint3
    data_loc = args.data_loc
    reset_loc = bool(args.reset_loc)
    object_type = args.obj
    grasping_env = args.grasp_env
    down_level = args.down_level
    num_data_target = args.num_data_target
    down_grasp_combined = bool(args.down_grasp_combined)
    only_above_block = bool(args.only_above_block)

    # target action distribution
    if not grasping_env :
        target_action_dist = [8.0,10.0 ,8.0,10.0,8.0,10.0,8.0,10.0,0.0,0.0]
    else:
        target_action_dist = [0.0,0.0 ,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.0, 10.0] # [8.0,10.0 ,8.0,10.0,8.0,10.0,8.0,10.0,0.0,6.0, 6.0]

    target_action_dist = target_action_dist / np.sum(target_action_dist)

    #build env
    env = robosuite.make(
        "BaxterPush",
        bin_type='table',
        object_type= object_type,
        ignore_done=True,
        has_renderer= bool(render), #True,
        has_offscreen_renderer= not bool(render), # added this new line
        camera_name="eye_on_right_wrist",
        gripper_visualization=False,
        use_camera_obs=False,
        use_object_obs=False,
        camera_depth=True,
        num_objects=num_blocks,
        control_freq=100,
        camera_width=screen_width,
        camera_height=screen_height,
        crop=crop
    )
    env = IKWrapper(env)
    env = BaxterTestingEnv1(env, task=task, render=render,
                    using_feature=using_feature,
                    random_spawn=random_spawn, block_random = block_random ,
                    rgbd=True, action_type=action_type, viewpoint1 = viewpoint1,
                    viewpoint2=viewpoint2, viewpoint3= viewpoint3,grasping_env=grasping_env, down_level=down_level,
                    is_test=False, arm_near_block = arm_near_block,
                    only_above_block=only_above_block, down_grasp_combined=down_grasp_combined)
    agent = GreedyAgent(env, down_grasp_combined)

    #create directory for view images
    save_name = os.path.join(FILE_PATH, data_loc)
    if save_data:
        if not os.path.exists(save_name):
            os.makedirs(save_name)
            os.makedirs(os.path.join(save_name, "pkls"))
            if save_view and not os.path.exists(os.path.join(save_name,"views")):
                os.makedirs(os.path.join(save_name,"views"))
        elif reset_loc:
            shutil.rmtree(save_name)
            os.makedirs(save_name)
            os.makedirs(os.path.join(save_name,"pkls"))
            if save_view:
                os.makedirs(os.path.join(save_name,"views"))

    #create or load action_dist file
    action_dist_his = np.zeros((env.action_size), dtype=int)
    if not os.path.exists(os.path.join(save_name, 'action_dist.csv')):
        with open(os.path.join(save_name, 'action_dist.csv'), 'w') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(action_dist_his)
    else:
        with open(os.path.join(save_name, 'action_dist.csv'), 'r') as f:
            reader = list(csv.reader(f))
            current = np.zeros((env.action_size), dtype=int)
            for k in range(len(reader[0])):
                current[k] = int(reader[0][k])
            # print(np.array(reader[0]))
            action_dist_his = current
    if np.sum(action_dist_his) == 0:
        action_dist_his_prob = action_dist_his
    else:
        action_dist_his_prob = action_dist_his / np.sum(action_dist_his)

    #start collecting
    total_steps = 0
    success_log = []
    buff_states = []
    buff_actions = []
    buff_dones = []
    for n in range(args.num_episodes):
        if print_on:
            print('[Episode %d]'%n)
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        step_count = 0
        ep_buff_states = []
        ep_buff_actions = []
        ep_buff_dones = []
        action_dist = np.zeros((env.action_size), dtype=int)
        while not done:
            step_count += 1
            action = agent.get_action(obs)
            new_obs, reward, done, stucked, _ = env.step(action)
            if print_on:
                print('action: %d / reward: %.2f'%(action, reward))
            cumulative_reward += reward

            # print(action_dist_his_prob[action])
            if action_dist[action] <= 12 and not (only_above_block and action != 9 ) :#\
                            #and not (not only_above_block and action_dist_his_prob[action] > target_action_dist[action]):
                obs = new_obs
                new_obs = np.concatenate([k for k in new_obs], axis=-1)
                ep_buff_states.append(new_obs)
                ep_buff_actions.append(action)
                ep_buff_dones.append(int(done))
                action_dist[action] +=1

            #save first image and last image if failed
            if (step_count==1 or (done == True and reward ==0)) and save_view:
                for i, k in enumerate(obs) :
                    im = Image.fromarray((255.0*k[:,:,:3]).astype(np.uint8))
                    now = datetime.datetime.now()
                    im.save(os.path.join(save_name, 'views'+'/'+'%d_%s-view%d.png'%(n,now.strftime("%m%d_%H%M%S"),i)))

        success = bool(cumulative_reward >= 90)
        success_log.append(int(success))
        # current buff action distribution

        for ind in range(env.action_size):
            action_dist[ind] =  ep_buff_actions.count(ind)

        print('action distribution (this episode) :')
        print(list(action_dist))
        print('Episode %d Ends with %3d steps and %3d Ep reward ' % (n + 1, step_count, cumulative_reward), )
        print('( Total steps:', total_steps, ')')
        print('success:', sum(success_log), ' ,   failure:', len(success_log) - sum(success_log))

        # recording the trajectories as pkl files
        if success and save_data:
            buff_states += ep_buff_states
            buff_actions += ep_buff_actions
            buff_dones += ep_buff_dones
            total_steps += len(ep_buff_states)

            if len(buff_states) >= args.max_buff:
                f_list = os.listdir(os.path.join(save_name,'pkls'))
                num_pickles = len([f for f in f_list if task in f])
                save_num = num_pickles // 3

                now = datetime.datetime.now()

                f = open(os.path.join(save_name, 'pkls/'+task + '_s_%d_%s.pkl'%(save_num,now.strftime("%m%d_%H%M%S"))), 'wb')
                pickle.dump(np.array(buff_states)[:args.max_buff], f)
                print(np.shape(np.asarray(buff_states)[:args.max_buff]))
                f.close()
                f = open(os.path.join(save_name, 'pkls/'+task + '_a_%d_%s.pkl'%(save_num,now.strftime("%m%d_%H%M%S"))), 'wb')
                pickle.dump(np.asarray(buff_actions)[:args.max_buff], f)
                f.close()
                f = open(os.path.join(save_name, 'pkls/'+task + '_d_%d_%s.pkl'%(save_num,now.strftime("%m%d_%H%M%S"))), 'wb')
                pickle.dump(np.asarray(buff_dones)[:args.max_buff], f)
                f.close()

                #find the action distribution of the saved buffer
                newly_added = np.zeros((env.action_size), dtype=int)
                for ind in range(env.action_size):
                    newly_added[ind] = buff_actions[:args.max_buff].count(ind)

                with open(os.path.join(save_name, 'action_dist.csv'),'r') as f:
                    reader = list(csv.reader(f))
                    current = np.zeros((env.action_size), dtype=int)
                    for k in range(len(reader[0])):
                        current[k] = int(reader[0][k])
                    # print(np.array(reader[0]))
                    action_dist_his = current + newly_added

                with open(os.path.join(save_name, 'action_dist.csv'),'w') as f:
                    writer = csv.writer(f, dialect='excel')
                    writer.writerow(action_dist_his)
                    action_dist_his_prob = action_dist_his / np.sum(action_dist_his)

                print()
                print('---' * 10)
                print(save_num, '-th file in this data folder')
                print('current action distribution (in this data folder):')
                print(list(action_dist_his))
                print('current success rate:', np.mean(success_log))
                print('---' * 10)

                buff_states = buff_states[args.max_buff:]
                buff_actions = buff_actions[args.max_buff:]
                buff_dones = buff_dones[args.max_buff:]
                # buff_states, buff_actions = [], []
        print()
        if np.sum(action_dist_his) > num_data_target:
            print("data sufficiently collected!")
            break



class GreedyAgent():
    def __init__(self, env, down_grasp_combined = False):
        self.env = env
        self.task = self.env.task
        # self.using_feature = self.env.using_feature
        self.mov_dist = self.env.mov_dist
        self.action_size = self.env.action_size
        self.action_type = self.env.action_type
        self.down_grasp_combined = down_grasp_combined

    def get_action(self, obs):
        mov_dist = self.mov_dist
        action = None
        if self.task == 'reach':
            predicted_distance_list = []
            for action in range(self.action_size):
                if action < 8:
                    mov_degree = action * np.pi / 4.0
                    arm_pos = self.env.arm_pos + np.array([mov_dist* np.cos(mov_degree),
                                                           mov_dist * np.sin(mov_degree),
                                                           0.0])
                elif action == 8:
                    arm_pos = self.env.arm_pos + np.array([0.0, 0.0, mov_dist])
                elif action == 9:
                    arm_pos = self.env.arm_pos + np.array([0.0, 0.0, -mov_dist])

                if arm_pos[2] < 0.57:
                    predicted_distance_list.append(np.inf)
                else:
                    dist = np.linalg.norm(arm_pos - self.env.obj_pos)
                    predicted_distance_list.append(dist)

            action = np.argmin(predicted_distance_list)

        elif self.task == 'push':
            if self.action_type=='2D':
                vec_target_obj = self.env.target_pos - self.env.obj_pos
                vec_obj_arm = self.env.obj_pos - self.env.arm_pos
                mov_vec_list = []
                for a in range(8):
                    mov_degree = a * np.pi / 4.0
                    mov_vec_list.append(np.array([mov_dist * np.cos(mov_degree),
                                                  mov_dist * np.sin(mov_degree)]))
                mov_cos_list = [self.get_cos(v, vec_obj_arm[:2]) for v in mov_vec_list]

                pred_vec_target_obj = vec_target_obj \
                                      - self.mov_dist * vec_obj_arm \
                                      / np.linalg.norm(vec_obj_arm)

                theta_threshold = np.pi/5 \
                    if np.linalg.norm(vec_target_obj[:2])>0.10 \
                    else np.pi/3
                if self.get_cos(pred_vec_target_obj[:2],
                                vec_obj_arm[:2]) > np.cos(theta_threshold):  # > 0
                    action = np.argmax(mov_cos_list)
                else:
                    mov_cos_list = [self.get_cos(v, vec_target_obj[:2]) for v in mov_vec_list]
                    action = np.argmax(mov_cos_list)
                    '''
                    next_obj_arm = [vec_obj_arm[:2] - v for v in mov_vec_list]
                    next_cos_list = [self.get_cos(vec_target_obj[:2], w) for w in next_obj_arm]
                    action = np.argmax(next_cos_list)
                    ## to avoid repetition ##
                    if (action+4)%8 == np.argmax(mov_cos_list):
                        next_cos_list[action] = -1.0
                        action = np.argmax(next_cos_list)
                    '''

            elif self.action_type=='3D':
                vec_target_obj = self.env.target_pos - self.env.obj_pos
                vec_obj_arm = self.env.obj_pos - self.env.arm_pos
                mov_vec_list = []
                for a in range(8):
                    mov_degree = a * np.pi / 4.0
                    mov_vec_list.append(np.array([mov_dist * np.cos(mov_degree),
                                                  mov_dist * np.sin(mov_degree)]))
                    # elif a == 8:
                    #     mov_vec_list.append(np.array([0.0, 0.0, mov_dist]))
                    # elif a == 9:
                    #     mov_vec_list.append(np.array([0.0, 0.0, -mov_dist]))
                mov_cos_list = [self.get_cos(v, vec_obj_arm[:2]) for v in mov_vec_list]

                if self.get_cos(vec_target_obj[:2], vec_obj_arm[:2]) > 0:
                    if self.env.arm_pos[2] < 0.65:
                        action = np.argmax(mov_cos_list)
                    else:
                        if np.linalg.norm(vec_obj_arm) > 2.0 * mov_dist:
                            action = 9
                        else:
                            next_obj_arm = [vec_obj_arm[:2] - v for v in mov_vec_list]
                            next_cos_list = [self.get_cos(vec_target_obj[:2], w)
                                             for w in next_obj_arm]
                            action = np.argmax(next_cos_list)
                            '''
                            best_a = np.argmax(mov_cos_list)
                            mov_cos_list[best_a] = np.min(mov_cos_list)
                            next_best_a = np.argmax(mov_cos_list)
                            if self.get_cos(mov_vec_list[best_a][:2], vec_target_obj[:2]) > 0:
                                action = best_a
                            else:
                                action = next_best_a
                            action = (action + 4) % 8
                            '''
                else:
                    if self.env.arm_pos[2] < 0.65:
                        action = 8
                    else:
                        next_obj_arm = [vec_obj_arm[:2] - v for v in mov_vec_list]
                        next_cos_list = [self.get_cos(vec_target_obj[:2], w)
                                         for w in next_obj_arm]
                        action = np.argmax(next_cos_list)

        # greedy agent for pick
        elif self.task=='pick':
            self.env.obj_id = self.env.env.obj_body_id['CustomObject_0']
            self.env.init_obj_pos = np.copy(self.env.env.sim.data.body_xpos[self.env.obj_id])
            self.env.obj_pos = np.copy(self.env.env.sim.data.body_xpos[self.env.obj_id])
            self.env.max_height = self.env.init_obj_pos[2]

            # self.target_id = self.env.obj_body_id['CustomObject_1']
            # self.target_pos = np.copy(self.env.sim.data.body_xpos[self.target_id])
            if self.env.env.num_objects == 2:
                # self.target_id = self.env.obj_body_id['CustomObject_1']
                self.env.target_pos = np.copy(self.env.env.sim.data.body_xpos[self.env.target_id])
            elif self.env.env.num_objects == 1:
                self.env.target_pos = self.env.goal

            ## move to Pick ## -> if far away, try all moves and pick the best one

            curr_arm_pos = self.env.arm_pos
            dist = np.linalg.norm(curr_arm_pos[:2] - self.env.obj_pos[:2])

            self.env.pre_vec = self.env.target_pos - self.env.obj_pos
            align_direction = self.env.pre_vec[:2] / np.linalg.norm(self.env.pre_vec[:2])  ##

            # self.env.state[11]+=np.pi/2.0 + 1.0

            ##Move to where the block is
            if dist > mov_dist/ 2:
                predicted_distance_list = []
                for action in range(8):
                    mov_degree = action * np.pi / 4.0
                    arm_pos = curr_arm_pos + np.array(
                        [mov_dist * np.cos(mov_degree),
                         mov_dist * np.sin(mov_degree),
                         0.0])
                    predicted_dist = np.linalg.norm(arm_pos - self.env.obj_pos)
                    predicted_distance_list.append(predicted_dist)
                action = np.argmin(predicted_distance_list)

            ## Do gripper twist if needed
            if action is None and self.env.grasping_env == 1:  # if grasping is included in the action
                main_eul = quat2euler(self.env.main_quat)
                # main_angle = np.arctan2(main_eul[1], main_eul[0])
                # print("state[11] : " + str(self.env.state[11]) + "  self.env.target_quat : " + str(main_eul))
                angle_diff = (self.env.state[11] - main_eul[0]) % (np.pi)
                # angle_diff = (align_angle) % (np.pi)

                if (self.env.env.object_type == 'stick' or 'smallcube') and not (
                        angle_diff > 3.0 * np.pi / 8.0 and angle_diff < 5.0 * np.pi / 8.0):
                    if angle_diff < 4.0 * np.pi / 8.0:
                        # print("angle_diff 1 is : "+ str(angle_diff))
                        action = 10
                    elif angle_diff >= 4.0 * np.pi / 8.0:
                        # print("angle_diff 2 is : " + str(angle_diff))
                        action = 10

                elif not (self.env.env.object_type == 'stick' or 'smallcube') \
                        and (angle_diff > np.pi / 8.0 and angle_diff < 7.0 * np.pi / 8.0):
                    if angle_diff < 4.0 * np.pi / 8.0:
                        action = 10
                    elif angle_diff >= 4.0 * np.pi / 8.0:
                        action = 10

            ## move down ##
            if action is None: #and self.env.grasp == 0.0: #if gripper open, move down
                if self.env.down_level==2 and curr_arm_pos[2] - self.env.obj_pos[2] > 0.04:  # move down
                    action = 11

                ## move down ##
                elif curr_arm_pos[2] - self.env.obj_pos[2] > 0:
                    action = 9
                ## And Grasp
                else:                   #if gripper open and almose near the object, grab the object
                    if self.down_grasp_combined:
                        action = 9
                    else:
                        action = 8
            ## move up ##
            # elif self.env.grasp == 1.0: #if already grabbed the object, move upward
            #     action = 12

        elif self.task=='place':
            if np.linalg.norm(self.env.obj_pos[:2]
                              - self.env.target_pos[:2]) > mov_dist / 2:
                curr_arm_pos = self.env.arm_pos
                predicted_distance_list = []
                for action in range(8):
                    mov_degree = action * np.pi / 4.0
                    arm_pos = curr_arm_pos + np.array(
                        [mov_dist * np.cos(mov_degree),
                         mov_dist * np.sin(mov_degree), 0.0])
                    predicted_dist = np.linalg.norm(arm_pos - self.env.target_pos)
                    predicted_distance_list.append(predicted_dist)
                action = np.argmin(predicted_distance_list)
            else:
                action = 9
        return action

    def get_cos(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__=='__main__':
    main()
