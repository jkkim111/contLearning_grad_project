import os
from behavior_cloning import * #SimpleCNN

import pickle
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('render', 0, 'render the screens')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes')
flags.DEFINE_integer('use_feature', 0, 'using feature-base states or image-base states.')
flags.DEFINE_string('task', 'reach', 'name of task: [ reach / push / pick / place ]')
flags.DEFINE_string('action_type', '2D', '[ 2D / 3D ]')
flags.DEFINE_integer('random_spawn', 0, 'randomly init robot arm pos and block pos')
flags.DEFINE_integer('small_cube', 1, 'use small block')
flags.DEFINE_integer('num_blocks', 1, 'number of blocks')

flags.DEFINE_integer('save_data', 1, 'save data or not')
flags.DEFINE_integer('max_buff', 512, 'number of steps saved in one data file.')

FLAGS = flags.FLAGS
using_feature = bool(FLAGS.use_feature)
if using_feature:
    print('This agent will use feature-based states..!!')
else:
    print('This agent will use image-based states..!!')

render = bool(FLAGS.render)
save_data = bool(FLAGS.save_data)
print_on = False
task = FLAGS.task
action_type = FLAGS.action_type
random_spawn = bool(FLAGS.random_spawn)
use_small_cube = bool(FLAGS.small_cube)
num_blocks = FLAGS.num_blocks


# camera resolution
screen_width = 192 #64
screen_height = 192 #64
crop = 128 #None

# Path which data will be saved in.
# save_name = os.path.join(FILE_PATH, 'data')
save_name = '/home/gun/ssd/disk/BPdata/one_block/place_2block'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU is not efficient here

def main():
    object_type = 'smallcube' if use_small_cube else 'cube' #if task=='pick' or task=='place' else 'cube'
    env = robosuite.make(
        "BaxterPush",
        bin_type='table',
        object_type=object_type,
        ignore_done=True,
        has_renderer=True,
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
    env = BaxterEnv(env, task=task, render=render, using_feature=using_feature, random_spawn=random_spawn, rgbd=True, action_type=action_type)

    agent = GreedyAgent(env)

    if not os.path.exists(save_name) and save_data:
        os.makedirs(save_name)

    total_steps = 0
    success_log = []
    buff_states = []
    buff_actions = []
    buff_dones = []
    for n in range(FLAGS.num_episodes):
        if print_on:
            print('[Episode %d]'%n)
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        step_count = 0
        ep_buff_states = []
        ep_buff_actions = []
        ep_buff_dones = []

        while not done:
            step_count += 1
            action = agent.get_action(obs)
            new_obs, reward, done, _ = env.step(action)
            if print_on:
                print('action: %d / reward: %.2f'%(action, reward))
            # print(step_count, 'steps \t action: ', action, '\t reward: ', reward)
            cumulative_reward += reward

            ep_buff_states.append(obs)
            ep_buff_actions.append(action)
            ep_buff_dones.append(int(done))
            obs = new_obs

        success = bool(cumulative_reward >= 90)
        success_log.append(int(success))

        # recording the trajectories
        if success and save_data:
            buff_states += ep_buff_states
            buff_actions += ep_buff_actions
            buff_dones += ep_buff_dones
            total_steps += len(ep_buff_states)

            if not os.path.isdir(save_name):
                os.makedirs(save_name)
            if len(buff_states) >= FLAGS.max_buff:
                f_list = os.listdir(save_name)
                num_pickles = len([f for f in f_list if task in f])
                save_num = num_pickles // 3
                with open(os.path.join(save_name, task + '_s_%d.pkl'%save_num), 'wb') as f:
                    pickle.dump(np.array(buff_states)[:FLAGS.max_buff], f)
                with open(os.path.join(save_name, task + '_a_%d.pkl'%save_num), 'wb') as f:
                    pickle.dump(np.array(buff_actions)[:FLAGS.max_buff], f)
                with open(os.path.join(save_name, task + '_d_%d.pkl'%save_num), 'wb') as f:
                    pickle.dump(np.array(buff_dones)[:FLAGS.max_buff], f)

                print('---' * 10)
                print(save_num, '-th file saved.')
                print('action distribution:')
                for a in range(max(env.action_size, max(buff_actions)+1)):
                    print('%d: %.2f'%(a, list(buff_actions).count(a)/len(buff_actions)))
                print('---' * 10)
                print('current success rate:', np.mean(success_log))

                buff_states = buff_states[FLAGS.max_buff:]
                buff_actions = buff_actions[FLAGS.max_buff:]
                buff_dones = buff_dones[FLAGS.max_buff:]
                # buff_states, buff_actions = [], []

        # print('success rate?:', np.mean(success_log))
        print('Episode %d ends.'%(n+1), '( Total steps:', total_steps, ')')
        print('Ep len:', step_count, 'steps.   Ep reward:', cumulative_reward)
        print('success:', sum(success_log), ' ,   failure:', len(success_log)-sum(success_log))
        print()


class GreedyAgent():
    def __init__(self, env):
        self.env = env
        self.task = self.env.task
        # self.using_feature = self.env.using_feature
        self.mov_dist = self.env.mov_dist
        self.action_size = self.env.action_size
        self.action_type = self.env.action_type

    def get_action(self, obs):
        mov_dist = self.mov_dist
        if self.task == 'reach':
            predicted_distance_list = []
            for action in range(self.action_size):
                if action < 8:
                    mov_degree = action * np.pi / 4.0
                    arm_pos = self.env.arm_pos + np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
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
                    mov_vec_list.append(np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree)]))
                mov_cos_list = [self.get_cos(v, vec_obj_arm[:2]) for v in mov_vec_list]

                pred_vec_target_obj = vec_target_obj - self.mov_dist * vec_obj_arm / np.linalg.norm(vec_obj_arm)

                theta_threshold = np.pi/5 if np.linalg.norm(vec_target_obj[:2])>0.10 else np.pi/3
                if self.get_cos(pred_vec_target_obj[:2], vec_obj_arm[:2]) > np.cos(theta_threshold):  # > 0
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
                    mov_vec_list.append(np.array([mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree)]))
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
                            next_cos_list = [self.get_cos(vec_target_obj[:2], w) for w in next_obj_arm]
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
                        next_cos_list = [self.get_cos(vec_target_obj[:2], w) for w in next_obj_arm]
                        action = np.argmax(next_cos_list)

        elif self.task=='pick':
            curr_arm_pos = self.env.arm_pos
            dist = np.linalg.norm(curr_arm_pos[:2] - self.env.obj_pos[:2])

            ## move to Pick ##
            if dist > mov_dist / 2:
                predicted_distance_list = []
                for action in range(8):
                    mov_degree = action * np.pi / 4.0
                    arm_pos = curr_arm_pos + np.array(
                        [mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
                    predicted_dist = np.linalg.norm(arm_pos - self.env.obj_pos)
                    predicted_distance_list.append(predicted_dist)
                action = np.argmin(predicted_distance_list)
            ## move down ##
            elif self.env.grasp == 0.0:
                if curr_arm_pos[2] - self.env.obj_pos[2] > 0:  # mov_dist:
                    action = 9
                    # if self.env.fix_stuck and bool(1 - self.env.grasp):
                    #     print("move up to fix the gripper")
                    #     # self.env.pre_obj_pos = self.env.obj_pos
                    #     self.env.fix_stuck = False
                    #     action = 8
                    # else :
                    #     action = 9
                ## Pick ##
                else:
                    action = 10
            ## move up ##
            elif self.env.grasp == 1.0:
                action = 8

        elif self.task=='place':
            if np.linalg.norm(self.env.obj_pos[:2] - self.env.target_pos[:2]) > mov_dist / 2:
                curr_arm_pos = self.env.arm_pos
                predicted_distance_list = []
                for action in range(8):
                    mov_degree = action * np.pi / 4.0
                    arm_pos = curr_arm_pos + np.array(
                        [mov_dist * np.cos(mov_degree), mov_dist * np.sin(mov_degree), 0.0])
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


