import os, sys
import argparse

## argument parsing ##
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=int, default=1)
parser.add_argument('--task', type=str, default="pick")
parser.add_argument('--save_view', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--viewpoint1', type=str, default="rlview1")
parser.add_argument('--viewpoint2', type=str, default="rlview2")
parser.add_argument('--data_loc', type=str, default="data")
parser.add_argument('--grasp_env', type=int, default=1)
parser.add_argument('--down_level', type=int, default=2)
parser.add_argument('--max_pool', type=int, default=1)

parser.add_argument('--obj', type=str, default="smallcube")
parser.add_argument('--num_episodes', type=int, default=50)
parser.add_argument('--action_type', type=str, default="2D")
parser.add_argument('--random_spawn', type=int, default=0)
parser.add_argument('--num-blocks', type=int, default=1)
# parser.add_argument('--small_cube', type=int, default=1)
args = parser.parse_args()

render = bool(args.render)
task = args.task
save_view = args.save_view
batch_size = args.batch_size
viewpoint1 = args.viewpoint1
viewpoint2 = args.viewpoint2
data_loc = args.data_loc
grasp_env = args.grasp_env
down_level = args.down_level
max_pool = args.max_pool
obj = args.obj
num_episodes = args.num_episodes
action_type = args.action_type
random_spawn = bool(args.random_spawn)
# use_small_cube = bool(args.small_cube)
num_blocks = args.num_blocks
os.environ["CYMJ_RENDER"] = "0" if not args.render else "1"

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..'))
sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
from cont_learning_picking import *
from ops import *

import datetime
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
import pickle

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

class SimpleCNN():
    def __init__(self, task, action_size, cnn_format='NHWC', model_name=None, max_pool=True, batch_size=50):
        self.task = task
        self.action_size = action_size

        self.cnn_format = cnn_format
        self.screen_height = 128
        self.screen_width = 128
        self.screen_channel = 4

        self.data_path = None
        self.num_epochs = num_episodes
        self.lr = 1e-4
        self.loss_type = 'ce' #'l2'
        self.test_freq = 1
        self.eval_freq = 5
        self.eval_from = 10
        self.num_test_ep = 1
        self.env = None
        self.batch_size = batch_size

        self.dueling = False
        self.now = datetime.datetime.now()
        if model_name is None:
            self.model_name = self.task + '_' + self.now.strftime("%m%d_%H%M%S")
        else:
            self.model_name = model_name
        self.checkpoint_dir = os.path.join(FILE_PATH, 'bc_train_log/', 'checkpoint/', self.model_name)

        self.max_pool = max_pool
        self.build_net()
        self.saver = tf.compat.v1.train.Saver(max_to_keep=20) #self.saver = tf.train.Saver(max_to_keep=20)
        self.max_accur = 0.0

    def set_datapath(self, data_path, data_type='pkl'): # data_type='npy' / 'pkl'
        #set path to data and load the collected data to self.a_list and self.s_list
        self.data_type = data_type
        file_list = os.listdir(os.path.join(os.getcwd(),data_path+'/pkls'))
        data_list = [f for f in file_list if data_type in f and self.task in f]
        if len(data_list) == 0:
            print('No data files exist. Wrong data path!!')
            return
        self.data_list = sorted([os.path.join(data_path, p) for p in data_list])
        self.a_list = sorted([p for p in data_list if self.task + '_a_' in p])
        self.s_list = sorted([p for p in data_list if self.task + '_s_' in p])
        # print(self.a_list)
        # print(self.s_list)
        assert len(self.a_list) == len(self.s_list)
        self.data_path = data_path+'/pkls'

    def build_net(self):
        self.w = {}
        self.t_w = {}

        # Set Initializer
        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.initializers.truncated_normal(0, 0.1) #0.1)
        activation_fn = tf.nn.relu

        # training network
        self.a_true = tf.placeholder('int64', [None], name='a_t')
        with tf.variable_scope('prediction'):
            if self.cnn_format == 'NHWC':
                self.s_t = tf.placeholder('float32', [None, 2, self.screen_height, self.screen_width, self.screen_channel], name='s_t')
            else:
                self.s_t = tf.placeholder('float32', [None, 2, self.screen_channel, self.screen_height, self.screen_width], name='s_t')

            # todo: concatenate the input states [b,2,h,w,d] -> [b,h,w,2d]
            first_ = self.s_t[:,0,:,:,:]
            second_ = self.s_t[:,1,:,:,:]
            L0 = tf.concat([first_,second_], axis=-1)
            # print(L0)
            # todo: build network using conv2d function (3x3 convolution, 2x2 strides, [32, 64, 128, 256] depths)
            # conv channel 32relu 32relu maxpooling dropout(0.25) 64 relu 64 relu  maxpool dropout

            w = {}
            L1, w['w_1'],w['b_1'] = conv2d(L0, 32, [3, 3], [2, 2], initializer=initializer, activation_fn=activation_fn, data_format=self.cnn_format, padding= 'SAME', name="l1")
            if self.max_pool:
                L1 = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L2, w['w_2'],w['b_2'] = conv2d(L1, 64, [3, 3], [2, 2], initializer=initializer, activation_fn=activation_fn, data_format=self.cnn_format, padding= 'SAME', name="l2")
            if self.max_pool:
                L2 = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L3, w['w_3'],w['b_3'] = conv2d(L2, 128, [3, 3], [2, 2], initializer=initializer, activation_fn=activation_fn, data_format=self.cnn_format, padding= 'SAME', name="l3")
            if self.max_pool:
                L3 = tf.nn.max_pool2d(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L4, w['w_4'],w['b_4'] = conv2d(L3, 256, [3, 3], [2, 2], initializer=initializer, activation_fn=activation_fn, data_format=self.cnn_format, padding= 'SAME', name="l4")

            # todo: global pooling & linear layer using linear function ([128, 128, action dim] depths) & argmax
            L4 = tf.reduce_mean(L4,[1, 2])
            L5, w['w_5'],w['b_5'] = linear(L4, 128, activation_fn=activation_fn, name="l5")
            L6, w['w_6'],w['b_6'] = linear(L5, 128, activation_fn=activation_fn, name="l6")
            self.res, w['w_7'],w['b_7'] = linear(L6, self.action_size, activation_fn=activation_fn, name="l7")
            # print(model)
            self.q_action = tf.argmax(self.res, axis = -1)
            print(self.q_action)

            # todo: define the loss type ('ce', 'l2') & optimizer & measurement (accuracy)
            if self.loss_type=='ce':
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.res, labels=tf.one_hot(self.a_true, depth=self.action_size)))
            elif self.loss_type=='l2':
                cross_entropy = tf.nn.l2_loss(self.res)
            self.cost = cross_entropy
            self.optimizer = tf.train.AdamOptimizer(1e-2).minimize(self.cost) #0.0001 batch_size 늘리는게 lr내리는 효과 (4-0.0001)

            self.correct_prediction = tf.equal(self.q_action, self.a_true)
            self.acc =  tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)) #tf.equal

    def set_env(self, env):
        self.env = env

    def load_pkl(self, pkl_file):
        with open(os.path.join(self.data_path, pkl_file), 'rb') as f:
            return pickle.load(f)

    def load_npy(self, npy_file):
        return np.load(os.path.join(self.data_path, npy_file))

    def load_model(self, sess):
        print(" [*] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.checkpoint_dir)
            return False

    def test_agent(self, sess):
        if self.env is None:
            return
        success_log = []
        for n in range(self.num_test_ep):
            obs = self.env.reset()
            np.shape(obs) ########3
            done = False
            cumulative_reward = 0.0
            step_count = 0
            action_his = []
            while not done:
                step_count += 1
                clipped_obs = np.clip(obs, 0.0, 5.0)
                s_0 = clipped_obs[0, :, :, :].copy()
                s_1 = clipped_obs[1, :, :, :].copy()
                state = np.concatenate([s_0, s_1], axis=-1)
                action = self.model.predict(state[np.newaxis, :])

                # action = sess.run(self.q_action, feed_dict={self.s_t: [clipped_obs]})
                action_his.append(action)
                # print("[TEST EPISODE %d] q_action : "%n + str(action))
                obs, reward, done, _ = self.env.step(action)
                cumulative_reward += reward

            if cumulative_reward >= 90:
                success_log.append(1.0)
            else:
                success_log.append(0.0)

            print("Episode : "+ str(n) + " / Step count : "+ str(step_count) + " / Cum Rew : " + str(cumulative_reward))
            action_dist = np.zeros((self.env.action_size), dtype=int)
            for ind in range(env.action_size):
                action_dist[ind] = action_his.count(ind)
            print('action distribution (this episode) :')
            print(list(action_dist))

            print('SUCCESS RATE on picking?:', np.mean(success_log))

        return np.mean(success_log)

    def train(self, sess):
        if self.data_type=='pkl':
            load_data = self.load_pkl
        elif self.data_type == 'npy':
            load_data = self.load_npy

        writer = SummaryWriter(os.path.join(FILE_PATH, 'bc_train_log/', 'tensorboard', self.task + '_' + self.now.strftime("%m%d_%H%M%S")+'_'+str(12+grasp_env+down_level)+'dim_'+str(obj)))
        sess.run(tf.global_variables_initializer())

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print('Training starts..')
        bs = self.batch_size
        test_bs = 8 #self.batch_size
        data_size = len(load_data(self.a_list[0]))
        if bs <= data_size:
            for epoch in range(self.num_epochs):
                epoch_cost = []
                epoch_accur = []
                for p_idx in np.random.permutation(len(self.a_list) - 1):  # -1

                    pkl_action = self.a_list[p_idx]
                    pkl_state = self.s_list[p_idx]
                    # print("pkl_action : " + pkl_action)
                    # print("pkl_state : " + pkl_state)
                    assert pkl_action[-5:] == pkl_state[-5:]
                    buff_actions = load_data(pkl_action)
                    buff_states = load_data(pkl_state)
                    assert len(buff_actions) == len(buff_states)

                    shuffler = np.random.permutation(len(buff_actions))
                    buff_actions = buff_actions[shuffler]
                    buff_states = buff_states[shuffler]
                    buff_states = np.clip(buff_states, 0.0, 5.0)

                    for i in range(len(buff_actions) // bs):
                        batch_actions = buff_actions[bs * i:bs * (i + 1)]
                        batch_states = buff_states[bs * i:bs * (i + 1)]
                        # print(batch_states)
                        # todo: train the network (additional output: cost, accuracy)
                        _, accuracy, cost = sess.run([self.optimizer, self.acc, self.cost],
                                                     feed_dict={self.s_t: batch_states, self.a_true: batch_actions})
                        epoch_cost.append(cost)
                        epoch_accur.append(accuracy)
                        # print(epoch_cost)

        elif bs > data_size :
            files_per_batch = bs//data_size
            for epoch in range(self.num_epochs):
                epoch_cost = []
                epoch_accur = []
                shuffler1 = np.random.permutation(len(self.a_list)-1)
                # print(shuffler1)
                self.a_list = list(np.array(self.a_list)[shuffler1])
                self.s_list = list(np.array(self.s_list)[shuffler1])
                iter_num = len(self.a_list)//files_per_batch
                for iter in range(iter_num) : # for every batch_size
                    buff_actions = np.array([])
                    buff_states = np.array([])
                    for file in range(files_per_batch):
                        np.append(buff_actions,load_data(self.a_list[iter*files_per_batch+file]))
                        np.append(buff_states,load_data(self.s_list[iter*files_per_batch+file]))

                    assert len(buff_actions) == len(buff_states)
                    shuffler = np.random.permutation(len(buff_actions)-1)
                    buff_actions = buff_actions[shuffler.astype(int)]
                    buff_states = buff_states[shuffler.astype(int)]
                    buff_states = np.clip(buff_states, 0.0, 5.0)

                    for i in range(len(buff_actions)//bs):
                        batch_actions = buff_actions[bs * i:bs * (i + 1)]
                        batch_states = buff_states[bs * i:bs * (i + 1)]
                        # print(batch_states)
                        # todo: train the network (additional output: cost, accuracy)
                        _, accuracy, cost = sess.run([self.optimizer,self.acc, self.cost], feed_dict={self.s_t: batch_states, self.a_true: batch_actions})
                        epoch_cost.append(cost)
                        epoch_accur.append(accuracy)
                        # print(epoch_cost)

            if (epoch+1) % self.test_freq == 0:
                random_test = np.random.randint(0,len(self.a_list)-1)
                pkl_action = self.a_list[random_test]
                pkl_state = self.s_list[random_test]
                assert pkl_action[-5:] == pkl_state[-5:]
                buff_actions = load_data(pkl_action)
                buff_states = load_data(pkl_state)
                assert len(buff_actions) == len(buff_states)
                buff_states = np.clip(buff_states, 0.0, 5.0)

                accur_list = []
                for i in range(len(buff_actions) // test_bs):
                    batch_actions = buff_actions[test_bs * i:test_bs * (i + 1)]
                    batch_states = buff_states[test_bs * i:test_bs * (i + 1)]
                    # todo: test the network (output: accuracy)
                    test_accur, test_equal = sess.run([self.acc, self.correct_prediction], feed_dict={self.s_t: batch_states, self.a_true: batch_actions})
                    # accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                    # print("test_equal : " + str(test_equal))
                    # print("test_accur : "+ str(test_accur)+ " / True is "+ str(list(test_equal).count(True)) +" out of " + str(test_bs))
                    accur_list.append(test_accur)
                test_accuracy = np.mean(accur_list)
                # test_accuracy_ran = np.mean(accur_list)

            writer.add_scalar('train-%s/test_accuracy' % self.task, test_accuracy, epoch + 1)
            writer.add_scalar('train-%s/train_cost'%self.task, np.mean(epoch_cost), epoch+1)
            writer.add_scalar('train-%s/train_accuracy'%self.task, np.mean(epoch_accur), epoch+1)
            print()
            print('[Epoch %d]' %epoch)
            print('cost: %.3f\t/   train accur: %.3f\t/   test accur:[%.3f]' \
                  % (np.mean(epoch_cost), np.mean(epoch_accur), test_accuracy))

            # save the model parameters
            if np.mean(test_accuracy) > self.max_accur:
                self.saver.save(sess, os.path.join(self.checkpoint_dir, 'model'), global_step=epoch)
                self.max_accur = test_accuracy
            # performance evaluation
            if (epoch+1 > self.eval_from) and (epoch+1) % self.eval_freq == 0:
                success_rate = self.test_agent(sess)
                writer.add_scalar('train-%s/success_rate' % self.task, success_rate, epoch + 1)

        print('Training done!')
        return


def main():
    eval = True
    screen_width = 192 #264
    screen_height = 192 #64
    crop = 128
    rgbd = True
    env = None

    if eval:
        object_type = obj #if task == 'pick' or task == 'place' else 'cube'
        print("Black screen ckpt 0")
        env = robosuite.make(
            "BaxterPush",
            bin_type='table',
            object_type=object_type,
            ignore_done=True,
            has_renderer=render,  # True,
            has_offscreen_renderer=not render,  # added this new line
            camera_name="frontview",
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
        # print("Black screen ckpt 1")
        env = IKWrapper(env)

        env = BaxterTestingEnv1(env, task=task, render=render, using_feature=False, random_spawn=random_spawn, \
              rgbd=rgbd, action_type=action_type, viewpoint1 = viewpoint1,\
              viewpoint2=viewpoint2, grasping_env=grasp_env, down_level=down_level)

        # print("Black screen ckpt 1")
        action_size = env.action_size

    if env is None:
        action_size = 8 if action_type=='2D' else 10

    data_path = data_loc #'data/processed_data'
    model = SimpleCNN(task=task, action_size=action_size, max_pool=max_pool, batch_size=batch_size)
    model.set_datapath(data_path, data_type='pkl')
    model.set_env(env)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model.train(sess)


if __name__=='__main__':
    main()
