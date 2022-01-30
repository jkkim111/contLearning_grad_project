import os, sys
import argparse

## argument parsing ##
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=int, default=1)
parser.add_argument('--task', type=str, default="pick")
parser.add_argument('--save_view', type=int, default=1)
parser.add_argument('--tags', type=str, default="pick_ResNet_10dim")

#data_path config
parser.add_argument('--data_path', type=str, default="data")
parser.add_argument('--data_path2', type=str, default="data")
parser.add_argument('--data_path3', type=str, default="data")

#task config
parser.add_argument('--task_num_old', type=int, default=0)
parser.add_argument('--task_num_new', type=int, default=1)
parser.add_argument('--task_num_newer', type=int, default=2)

# train-test config
parser.add_argument('--only_test', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=30)
parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--num_test_ep', type=int, default=10)

#baxter_env config
parser.add_argument('--mov_dist', type=float, default=0.02)
parser.add_argument('--random_spawn', type=int, default=1)
parser.add_argument('--block_random', type=int, default=0)
parser.add_argument('--obj', type=str, default="smallcube")
parser.add_argument('--num-blocks', type=int, default=1)

#network config
parser.add_argument('--network', type=str, default='ResNet')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--reg', type=str, default=None)
parser.add_argument('--reg_coeff', type=float, default=None)
parser.add_argument('--lrdecay', type=float, default=0.02)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch_dsize', type=int, default=1280)
parser.add_argument('--lambda_alpha', type=float, default=1.0)
parser.add_argument('--lambda_beta', type=float, default=1.0)

args = parser.parse_args()
render = bool(args.render)


task = args.task
save_view = args.save_view
tags = args.tags

batch_size = args.batch_size

data_path = args.data_path
data_path2 = args.data_path2
data_path3 = args.data_path3
only_test = args.only_test

lrdecay = args.lrdecay
network = args.network
mov_dist = args.mov_dist

random_spawn = bool(args.random_spawn)
block_random = bool(args.block_random)
obj = args.obj
num_episodes = args.num_episodes
eval_freq = args.eval_freq
num_test_ep = args.num_test_ep
num_blocks = args.num_blocks

lr= args.lr
reg = args.reg
reg_coeff = args.reg_coeff
epoch_dsize = args.epoch_dsize
lambda_alpha = args.lambda_alpha
lambda_beta = args.lambda_beta

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
from PIL import Image
from sklearn.utils import shuffle
import h5py

from keras.models import Sequential, Model, load_model as LoadModel
from keras.layers import Dense, Dropout, Activation, Flatten, Add, Lambda, Softmax
from keras.layers import Conv2D, MaxPooling2D, Input, Multiply, Reshape, BatchNormalization, AveragePooling2D
from keras.losses import categorical_crossentropy
import keras.activations as activations
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import random

reset_optimizer = False
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

if reg == "l2":
    reg = l2(reg_coeff)

else:
    reg = None


class ResNet_LwF:
    def __init__(self, task, action_size, model_name=None, batch_size=64, repetition = [3,4,6,3], task_num = [1,2,3]):
        self.action_size = action_size

        self.task = task
        self.action_size = action_size

        self.screen_height = 128
        self.screen_width = 128
        self.screen_channel = 4

        self.data_path = None
        self.num_epochs = num_episodes
        self.lr = lr
        self.loss_type = 'ce'  # 'l2'
        self.test_freq = 10

        self.eval_freq = eval_freq
        self.eval_from = 0
        self.num_test_ep = num_test_ep

        self.env = None
        self.batch_size = batch_size
        self.repetition = repetition
        self.s_test = None
        self.a_test = None
        self.only_test = only_test

        self.test_env_task_num = None

        self.dropout_rate = 0.1
        self.lrdecay = lrdecay
        if network == 'ResNet':
            self.network = self.ResNetwork
        elif network == 'simpleCNN':
            self.network = self.simpleCNN

        self.dueling = False
        self.now = datetime.datetime.now()
        if model_name is None:
            self.model_name = self.task + '_' + self.now.strftime("%m%d_%H%M%S")
        else:
            self.model_name = model_name
        self.checkpoint_dir = os.path.join(FILE_PATH, 'lwf_train_log/', 'keras_saved_model/')
        self.model = None
        self.data_lists = {}

        # self.saver = tf.compat.v1.train.Saver(max_to_keep=20)  # self.saver = tf.train.Saver(max_to_keep=20)
        self.max_accur = 0.0

    def get_shortcut(self, inputs, residual):
        input_shape = K.int_shape(inputs)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        shortcut = inputs
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[3],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="glorot_normal",
                              kernel_regularizer=reg)(inputs)  # 0.0001
        return shortcut

    def block_lv(self, name, inputs, num_filters, init_strides, is_first_block=False):
        if is_first_block:
            x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer="glorot_normal",name=name + 'conv_1', kernel_regularizer= reg)(inputs)
        else:
            x = BatchNormalization(axis=-1)(inputs)
            x = Activation('relu')(x)
            x = Conv2D(num_filters, (3, 3), strides=init_strides, padding='same', kernel_initializer="glorot_normal",kernel_regularizer= reg,name=name + 'conv_1')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3, 3), padding='same', kernel_initializer="glorot_normal",kernel_regularizer= reg,name=name + 'conv_2')(x)
        shortcut = self.get_shortcut(inputs, x)
        return Add()([shortcut, x])

    def res_block(self, inputs, num_filters, repetition, group_idx, is_first_layer=False):
        x = inputs
        for i in range(repetition):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            x = self.block_lv('group_%d_block_%d_' % (group_idx, i), x, num_filters, init_strides,
                         (i == 0) and is_first_layer)
        return x

    def ResNetwork(self, _input_shape, output_dim, repetition=[3,3,4]):
        inputs = Input(shape=_input_shape)

        x = Conv2D(32, (7, 7), strides=(2, 2), padding='same', kernel_initializer="glorot_normal",kernel_regularizer= reg,name='first_conv')(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        filters = 32
        for i, r in enumerate(repetition):
            x = self.res_block(x, filters, r, i, is_first_layer=(i == 0))
            filters *= 2

        # Last activation
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        # Classifier block
        x_shape = K.int_shape(x)
        pool2 = AveragePooling2D(pool_size=(x_shape[1], x_shape[2]))(x)
        x_last = Flatten()(x)
        # logits_first = Dense(units=output_dim,
        #                 activation='softmax', name='first_task')(x_last)
        # logits_second = Dense(units=output_dim,
        #                 activation='softmax', name='second_task')(x_last)
        # logits_third = Dense(units=output_dim,
        #                 activation='softmax', name='third_task')(x_last)

        logits_first = Dense(units=output_dim,
                        activation=None, name='logit_first')(x_last)
        logits_second = Dense(units=output_dim,
                        activation=None, name='logit_second')(x_last)
        logits_third = Dense(units=output_dim,
                        activation=None, name='logit_third')(x_last)
        output_first = Softmax(name='first_softmax')(logits_first)
        output_second = Softmax(name='second_softmax')(logits_second)
        output_third = Softmax(name='third_softmax')(logits_third)

        # model = Model(inputs=inputs, outputs=[output_first, output_second, output_third])
        # model = Model(inputs=inputs, outputs=[logits_first, logits_second, logits_third])
        model = Model(inputs=inputs, outputs=[logits_first, logits_second, logits_third,
                                                output_first, output_second, output_third])
        return model

    def set_datapath(self, task_num_old, task_num_new, task_num_newer,
                    data_path_old, data_path_new, data_path_newer, data_type='pkl'): # data_type='npy' / 'pkl'
        #set path to data and load the collected data to self.a_list and self.s_list
        self.data_path_old = data_path_old + '/pkls'
        self.data_path_new = data_path_new + '/pkls'
        self.data_path_newer = data_path_newer + '/pkls'

        self.file_list_old = os.listdir(os.path.join(os.getcwd(),self.data_path_old))
        self.file_list_new = os.listdir(os.path.join(os.getcwd(),self.data_path_new))
        self.file_list_newer = os.listdir(os.path.join(os.getcwd(),self.data_path_newer))

        self.task_num_old = task_num_old
        self.task_num_new = task_num_new
        self.task_num_newer = task_num_newer

        self.data_lists['a_list_task'+str(task_num_old)] = sorted([p for p in self.file_list_old if self.task + '_a_' in p])
        self.data_lists['s_list_task'+str(task_num_old)] = sorted([p for p in self.file_list_old if self.task + '_s_' in p])
        self.data_lists['a_list_task'+str(task_num_new)] = sorted([p for p in self.file_list_new if self.task + '_a_' in p])
        self.data_lists['s_list_task'+str(task_num_new)] = sorted([p for p in self.file_list_new if self.task + '_s_' in p])
        self.data_lists['a_list_task'+str(task_num_newer)] = sorted([p for p in self.file_list_newer if self.task + '_a_' in p])
        self.data_lists['s_list_task'+str(task_num_newer)] = sorted([p for p in self.file_list_newer if self.task + '_s_' in p])
        self.folder_name = 'tasks{}+{}+{}_{}_{}'.format(task_num_old, task_num_new, task_num_newer,self.now.strftime("%m%d_%H%M%S"),tags)
        if not os.path.exists(os.path.join(FILE_PATH,'lwf_train_log/failed',self.folder_name)):
            os.makedirs(os.path.join(FILE_PATH,'lwf_train_log/failed',self.folder_name))

    def set_env(self, env, task_num, init=0):
        if init:
            self.env = BaxterTestingEnv1(env, task=task, render=render,
                                using_feature=False,
                                random_spawn=random_spawn, block_random = block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = mov_dist, object_type = "smallcube")
        else:
            self.test_env_task_num = task_num
            if task_num == 0:
                self.env = BaxterTestingEnv1(env.env, task=task, render=render,
                                using_feature=False,
                                random_spawn=random_spawn, block_random = block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = mov_dist, object_type = "smallcube")
            elif task_num == 1:
                self.env = BaxterTestingEnv1(env.env, task=task, render=render,
                                using_feature=False,
                                random_spawn=random_spawn, block_random = block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = mov_dist, object_type = "lemon_1")

            elif task_num == 2:
                self.env = BaxterTestingEnv1(env.env, task=task, render=render,
                                using_feature=False,
                                random_spawn=random_spawn, block_random = block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = mov_dist, object_type = "stick")
            elif task_num == 3:
                self.env = BaxterTestingEnv1(env.env, task=task, render=render,
                                using_feature=False,
                                random_spawn=random_spawn, block_random = block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1_1",
                                viewpoint2="rlview2_1", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = mov_dist, object_type = "smallcube")
            elif task_num == 4:
                self.env = BaxterTestingEnv1(env.env, task=task, render=render,
                                using_feature=False,
                                random_spawn=random_spawn, block_random = block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1_2",
                                viewpoint2="rlview2_2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = mov_dist, object_type = "smallcube")


    def save_task_logits(self, which_head, task_num_old, task_num_new):
        #call set_datapath -> need first task dir, second task dir, and save_dir
        # self.model = self.ResNetwork(_input_shape=[128, 128, 8], output_dim=self.env.action_size, repetition=self.repetition)

        print("saving logits of head-%d_data-%d"%(task_num_old, task_num_new))
        self.logit_data_dir= os.path.join(FILE_PATH,'lwf_logits_saved',self.folder_name)
        if not os.path.exists(self.logit_data_dir):
            os.makedirs(self.logit_data_dir)

        # upload data by batch and forward and get logit data
        for i, batch in enumerate(self.data_lists['s_list_task'+str(task_num_new)]): #for all s_data
            buff_states = self.load_pkl(batch)
            logit_data = self.model.predict(buff_states)[which_head]

            now = datetime.datetime.now()
            f = open(os.path.join(self.logit_data_dir, task + '_l_head-%d_data-%d_%s'%(task_num_old, task_num_new, batch)), 'wb')
            pickle.dump(logit_data, f)
            # print(np.shape(np.asarray(logit_data)))
            f.close()

    def test_agent(self, num_rep = 10, task_num = 0, head = 0):
        if self.test_env_task_num != task_num:
            self.set_env(self.env, task_num)

        print("EVAL for TASK  %d (HEAD %d) STARTED"%(task_num, head))

        success_log = []
        failed_case_dist = np.zeros((8,))
        for n in range(num_rep): #self.num_test_ep):
            obs = self.env.reset()
            done = False
            cumulative_reward = 0.0
            step_count = 0
            action_his = []
            while not done:
                step_count += 1
                # clipped_obs = np.clip(obs, 0.0, 5.0)
                obs_re = np.concatenate([k for i,k in enumerate(obs)], axis=-1)
                action = np.argmax(self.model.predict(np.expand_dims(obs_re, axis=0))[head+3])
                action_his.append(action)
                # print("[TEST EPISODE %d] q_action : "%n + str(action))
                obs, reward, done, stucked, _ = self.env.step(action)
                cumulative_reward += reward

            if cumulative_reward >= 90:
                success_log.append(1.0)
                failed_case_dist[0] += 1
            else:
                success_log.append(0.0)
                failed_case_dist[stucked] +=1
                obs1 = np.array(obs[0])
                obs2 = np.array(obs[1])
                rescaled1 = (255.0*obs1[:,:,:3]).astype(np.uint8)
                rescaled2 = (255.0 * obs2[:,:,:3]).astype(np.uint8)
                im1 = Image.fromarray(rescaled1)
                im2 = Image.fromarray(rescaled2)
                im1.save(os.path.join(FILE_PATH, 'lwf_train_log', 'failed/'+self.folder_name+'/test_'
                                      + datetime.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(stucked)+'-view1.png'))
                im2.save(os.path.join(FILE_PATH, 'lwf_train_log', 'failed/'+self.folder_name+'/test_'
                                      + datetime.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(stucked)+'-view2.png'))


            # print("Episode : "+ str(n) + " / Step count : "+ str(step_count) + " / Cum Rew : " + str(cumulative_reward))
            action_dist = np.zeros((self.env.action_size), dtype=int)
            for ind in range(self.env.action_size):
                action_dist[ind] = action_his.count(ind)
            print('dist:',list(action_dist))
        print('SUCCESS RATE on picking?:', np.mean(success_log))
        print('test outcome dist (succ,wrong grasp,not going down, ,max step, ,out-bound,stuck):')
        print(list(failed_case_dist))
        print()

        return np.mean(success_log), failed_case_dist

    def train(self, task_order=0):
        print('Training starts..')
        print("task order: ", task_order)
        folder_name = self.folder_name+'_{}_{}'.format(task_order, datetime.datetime.now().strftime("%m%d_%H%M%S"))

        #load or make a keras model
        optimizer = Adam(lr=self.lr)
        self.model = self.ResNetwork(_input_shape=[128, 128, 8], output_dim=self.env.action_size, repetition=self.repetition)

        if task_order ==0:
            self.a_list =self.data_lists['a_list_task'+str(self.task_num_old)]
            self.s_list =self.data_lists['s_list_task'+str(self.task_num_old)]
            self.model.compile(loss ={'first_softmax':self.custom_loss1,
                                    'logit_second':self.custom_loss0,
                                    'logit_third':self.custom_loss0},
                                    optimizer=optimizer, metrics= ['accuracy'],
                                    loss_weights=[0, 0, 0, 1.0, 0, 0])
            self.task_num_curr = self.task_num_old

        elif task_order ==1:
            self.a_list =self.data_lists['a_list_task'+str(self.task_num_new)]
            self.s_list =self.data_lists['s_list_task'+str(self.task_num_new)]
            self.model.compile(loss ={'logit_first':self.custom_loss2,
                                    'second_softmax':self.custom_loss1,
                                    'logit_third': self.custom_loss0},
                                    optimizer=optimizer, metrics= ['accuracy'],
                                    loss_weights=[lambda_alpha, 0, 0, 0, 1.0, 0])
            self.task_num_curr = self.task_num_new

        elif task_order ==2:
            self.a_list =self.data_lists['a_list_task'+str(self.task_num_newer)]
            self.s_list =self.data_lists['s_list_task'+str(self.task_num_newer)]
            self.model.compile(loss ={'logit_first':self.custom_loss2,
                                    'logit_second':self.custom_loss2,
                                    'third_softmax':self.custom_loss1},
                                    optimizer=optimizer, metrics= ['accuracy'],
                                    loss_weights=[lambda_alpha, lambda_beta, 0, 0, 0, 1.0])
            self.task_num_curr = self.task_num_newer

        print("finished compiling")
        print("metrics_names is ", self.model.metrics_names)

        #tensorboard and mkdir
        writer = SummaryWriter(os.path.join(FILE_PATH, 'lwf_train_log/', 'tensorboard', self.folder_name))
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(os.path.join(FILE_PATH,'lwf_train_log','failed')):
            os.makedirs(os.path.join(FILE_PATH, 'lwf_train_log', 'failed'))

        #test only
        if self.only_test:
            if task_order == 0:
                success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_old, 0)
                writer.add_scalar('test_only-%s/order%d-task%d' % (self.task, 0, self.task_num_old), success_rate, epoch + 1)

                success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_new, 1)
                writer.add_scalar('test_only-%s/order%d-task%d' % (self.task, 1, self.task_num_new), success_rate, epoch + 1)

                success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_newer, 2)
                writer.add_scalar('test_only-%s/order%d-task%d' % (self.task, 2, self.task_num_newer), success_rate, epoch + 1)

                print("only test finished!")
            return

        #load data

        load_data = self.load_pkl
        data_size = len(load_data(self.a_list[0]))

        self.a_list, self.s_list = shuffle(self.a_list, self.s_list)

        hist= None
        if self.a_test is None:
            s_test = np.empty([0,128,128,4*(2)], float)
            a_test = np.empty([0,], int)
            test_a_file = self.a_list[-10:]
            test_s_file = self.s_list[-10:]
            self.a_list = self.a_list[:-10]
            self.s_list = self.s_list[:-10]
            for p_idx in range(len(test_a_file)):  # -1

                buff_actions = load_data(test_a_file[p_idx])
                buff_states = load_data(test_s_file[p_idx])

                if np.shape(buff_states)[-1] == 4:
                    buff_states = np.concatenate([buff_states[:, k, :, :, :] for k in range(np.shape(buff_states)[1])], axis=-1)

                s_test = np.append(s_test, buff_states, axis=0)
                a_test = np.append(a_test, buff_actions, axis=0)

            a_test = [int(k) for k in a_test]
            if np.shape(s_test)[-1] == 4:
                self.s_test = np.concatenate([s_test[:, k, :, :, :] for k in range(np.shape(s_test)[1])],
                                              axis=-1)
            else:
                self.s_test = s_test
            self.a_test = np.asarray([np.eye(self.env.action_size)[k] for k in a_test])

        for epoch in range(self.num_epochs+1):
            print()
            print('[Epoch %d]' % epoch)

            #shuffle the dataset before each epoch and iterate over the division of the dataset
            self.a_list, self.s_list = shuffle(self.a_list, self.s_list)
            num_rounds =  data_size*len(self.a_list) // epoch_dsize
            ep_train_acc = []
            ep_loss = []

            for round in range(num_rounds):
                train_s_data = np.empty([0,128,128,4*(2)], float)
                train_a_data = np.empty([0,], int)
                train_logit_data1 = np.empty([0,self.env.action_size], float)
                train_logit_data2 = np.empty([0,self.env.action_size], float)

                for p_idx in range(int(epoch_dsize/data_size)):  # -1
                    pkl_action = self.a_list[int(round*epoch_dsize/data_size+p_idx)]
                    pkl_state = self.s_list[int(round*epoch_dsize/data_size+p_idx)]
                    assert pkl_action[-5:] == pkl_state[-5:]
                    buff_actions = load_data(pkl_action)
                    buff_states = load_data(pkl_state)
                    assert len(buff_actions) == len(buff_states)
                    if np.shape(buff_states)[-1] == 4:
                        buff_states = np.concatenate([buff_states[:, k, :, :, :] for k in range(np.shape(buff_states)[1])],axis=-1)
                    # shuffler = np.random.permutation(len(buff_actions))
                    # buff_actions = buff_actions[shuffler]
                    # buff_states = buff_states[shuffler]
                    # buff_states = np.clip(buff_states, 0.0, 5.0)

                    train_s_data = np.append(train_s_data,buff_states,axis=0)
                    train_a_data = np.append(train_a_data,buff_actions,axis=0)
                    if task_order == 1:
                        #find the same name / load the pkl /  add them to the logit_data1
                        buff_logit = self.load_pkl_logit(pkl_state, self.task_num_old, self.task_num_new)
                        # print(np.shape(buff_logit))
                        # print(np.shape(train_logit_data1))
                        train_logit_data1= np.append(train_logit_data1, buff_logit, axis=0)
                    elif task_order == 2:
                        #find the same name / load the pkl /  add them to the logit_data1
                        buff_logit = self.load_pkl_logit(pkl_state, self.task_num_old, self.task_num_newer)
                        train_logit_data1= np.append(train_logit_data1, buff_logit, axis=0)
                        buff_logit = self.load_pkl_logit(pkl_state, self.task_num_new, self.task_num_newer)
                        train_logit_data2= np.append(train_logit_data2, buff_logit, axis=0)

                # fit
                if np.shape(train_s_data)[-1] ==4:
                    train_s_data = np.concatenate([train_s_data[:,k,:,:,:] for k in range(np.shape(train_s_data)[1])], axis=-1)

                train_a_data = np.asarray([np.eye(self.env.action_size)[int(k)] for k in train_a_data], dtype=int)
                if np.shape(train_logit_data1)[0] == 0:
                    train_logit_data1 = np.zeros((epoch_dsize, 10))
                if np.shape(train_logit_data2)[0] == 0:
                    train_logit_data2 = np.zeros((epoch_dsize, 10))

                # print("train-a_data:", np.shape(train_a_data))
                K.set_value(self.model.optimizer.lr, self.schedule_lr(epoch, round, num_rounds, self.lrdecay))
                if task_order ==0:
                    y= {'first_softmax':train_a_data, 'logit_second': train_logit_data1, 'logit_third':train_logit_data2}
                    met = ["first_softmax_loss", "first_softmax_acc", "loss"]
                    test_layer = ['first_softmax', 'logit_second', 'logit_third']
                elif task_order ==1:
                    y= {'second_softmax':train_a_data, 'logit_first': train_logit_data1, 'logit_third':train_logit_data2}
                    met = ["second_softmax_loss", "second_softmax_acc","loss"]
                    test_layer = ['second_softmax', 'logit_first','logit_third']
                elif task_order ==2:
                    y= {'third_softmax':train_a_data, 'logit_first': train_logit_data1, 'logit_second': train_logit_data2}
                    met = ["third_softmax_loss", "third_softmax_acc","loss"]
                    test_layer = ['third_softmax', 'logit_first','logit_second']

                hist = self.model.fit(train_s_data, y,
                                      epochs=1, verbose=0,validation_split=0.0, steps_per_epoch = int(epoch_dsize/self.batch_size))
                ep_train_acc += hist.history[met[1]]
                ep_loss += hist.history[met[0]]
                print("EP %d round %d : train_acc - %.3f / loss %.3f / task_loss %.3f"%(epoch, round, hist.history[met[1]][0], hist.history[met[-1]][0], hist.history[met[0]][0]))

            #evaluate and append accuracy
            ep_train_acc = np.sum(ep_train_acc)/num_rounds
            ep_loss = np.sum(ep_loss)/num_rounds
            test_accuracy1 = self.model.evaluate(self.s_test,
                                    {test_layer[0] : self.a_test, test_layer[1]:np.zeros((data_size*10, 10)),test_layer[2]:np.zeros((data_size*10, 10))})
            test_accuracy = test_accuracy1[-1]
            # print("test accuracy for EP {} is :".format(epoch), test_accuracy)

            #append
            writer.add_scalar('train-%s/test_accuracy_order%d-task%d' % (self.task, task_order, self.task_num_curr), test_accuracy, epoch + 1)
            writer.add_scalar('train-%s/train_cost_order%d-task%d' % (self.task, task_order, self.task_num_curr), ep_loss, epoch+1)
            writer.add_scalar('train-%s/train_accuracy_order%d-task%d' % (self.task, task_order, self.task_num_curr), ep_train_acc, epoch+1)


            print('cost: %.3f\t/   train accur: %.3f\t/   test accur:[%.3f]' \
                  % (ep_loss,
                     ep_train_acc,
                     test_accuracy))

            # save the model parameters
            if ep_train_acc > self.max_accur:
                self.model.save(os.path.join(FILE_PATH, 'lwf_train_log/', 'keras_saved_model', folder_name+".h5"))
                self.max_accur = ep_train_acc

            # performance evaluation
            if  (epoch+1) % self.eval_freq == 0:
                print("-----------EVAL FOR ORDER %d--------------"%(task_order))
                if task_order==0:
                 #(epoch+1 > self.eval_from) and
                    if (epoch+1) == self.num_epochs:
                        success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_old, 0)
                    else:
                        success_rate, _ = self.test_agent(self.num_test_ep, self.task_num_old, 0)
                    writer.add_scalar('train-%s/success_rate-order%d-task%d' % (self.task, task_order, self.task_num_old), success_rate, epoch + 1)
                elif task_order ==1:
                    if (epoch+1) == self.num_epochs:
                        success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_old, 0)
                    else:
                        success_rate, _ = self.test_agent(self.num_test_ep, self.task_num_old, 0)
                    writer.add_scalar('train-%s/success_rate-order%d-task%d' % (self.task, task_order, self.task_num_old), success_rate, epoch + 1)

                    if (epoch+1) == self.num_epochs:
                        success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_new, 1)
                    else:
                        success_rate, _ = self.test_agent(self.num_test_ep, self.task_num_new, 1)
                    writer.add_scalar('train-%s/success_rate-order%d-task%d' % (self.task, task_order, self.task_num_new), success_rate, epoch + 1)

                elif task_order ==2:
                    if (epoch+1) == self.num_epochs:
                        success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_old, 0)
                    else:
                        success_rate, _ = self.test_agent(self.num_test_ep, self.task_num_old, 0)
                    writer.add_scalar('train-%s/success_rate-order%d-task%d' % (self.task, task_order, self.task_num_old), success_rate, epoch + 1)

                    if (epoch+1) == self.num_epochs:
                        success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_new, 1)
                    else:
                        success_rate, _ = self.test_agent(self.num_test_ep, self.task_num_new, 1)
                    writer.add_scalar('train-%s/success_rate-order%d-task%d' % (self.task, task_order, self.task_num_new), success_rate, epoch + 1)

                    if (epoch+1) == self.num_epochs:
                        success_rate, _ = self.test_agent(5*self.num_test_ep, self.task_num_newer, 2)
                    else:
                        success_rate, _ = self.test_agent(self.num_test_ep, self.task_num_newer, 2)
                    writer.add_scalar('train-%s/success_rate-order%d-task%d' % (self.task, task_order, self.task_num_newer), success_rate, epoch + 1)

        print('Training done!')
        return

    def schedule_lr(self, epoch, round, num_rounds, decay_rate = 0.1):
        initial_lrate = self.lr
        k = decay_rate
        lrate = initial_lrate * np.exp(-k*(epoch+float(round)/num_rounds))
        # print("learning rate for this round is : ", lrate)
        return lrate


    def load_pkl(self, pkl_file):
        if pkl_file in self.file_list_old :
            with open(os.path.join(self.data_path_old, pkl_file), 'rb') as f:
                return pickle.load(f)
        elif pkl_file in self.file_list_new :
            with open(os.path.join(self.data_path_new, pkl_file), 'rb') as f:
                return pickle.load(f)
        elif pkl_file in self.file_list_newer :
            with open(os.path.join(self.data_path_newer, pkl_file), 'rb') as f:
                return pickle.load(f)

    def load_pkl_logit(self, pkl_file, task_num_old, task_num_new):
        self.file_list_logit = os.listdir(self.logit_data_dir)
        # for i in [s for s in self.file_list_logit if 'pick_l_head-%d_data-%d_%s'%(task_num_old, task_num_new, pkl_file) in s]:
        with open(os.path.join(self.logit_data_dir, 'pick_l_head-%d_data-%d_%s'%(task_num_old, task_num_new, pkl_file)), 'rb') as f:
            return pickle.load(f)

    def load_npy(self, npy_file):
        return np.load(os.path.join(self.data_path, npy_file))

    def custom_loss0(self, y_true, y_pred):
        loss = tf.zeros(tf.shape(y_true))*y_pred
        return loss

    def custom_loss1(self,y_true, y_pred):
        # y_exp = Softmax()(y_pred)
        # print("y_true:", y_true[2])
        loss = categorical_crossentropy(y_true, y_pred)
        return loss #tf.convert_to_tensor(loss)

    def custom_loss2(self, y_true, y_pred):
        logit_data = y_true
        logit_output = y_pred
        T = 2
        logit_data_T = Lambda(lambda x: x/T)(logit_data)
        logit_output_T = Lambda(lambda x: x/T)(logit_output)
        logit_data_T_soft = Softmax()(logit_data_T)
        logit_output_T_soft = Softmax()(logit_output_T)
        # y_hat = tf.math.pow(logit_data, tf.constant([1/T]*10))
        # y_hat = tf.math.divide(y_hat, tf.math.reduce_sum(y_hat))
        # y_output_hat = tf.math.pow(logit_output, tf.constant([1/T]*10))
        # y_output_hat = tf.math.divide(y_output_hat, tf.math.reduce_sum(y_output_hat))

        loss = categorical_crossentropy(logit_data_T_soft, logit_output_T_soft)
        return loss

        # logit_data = y_true[0]
        # logit_output = y_output[0]
        # logit_data2 = y_true[1]
        # logit_output2 = y_output[1]
        #
        # y_hat = [k**(.5) for k in logit_data]
        # y_hat /= np.sum(y_hat)
        # y_hat2 = [k**(.5) for k in logit_data2]
        # y_hat2 /= np.sum(y_hat2)
        #
        # y_output_hat = [k**(.5) for k in logit_output]
        # y_output_hat /= np.sum(y_output_hat)
        # y_output_hat2 = [k**(.5) for k in logit_output2]
        # y_output_hat2 /= np.sum(y_output_hat2)
        #
        # loss = categorical_crossentropy(y_true[1],y_output[1]) + categorical_crossentropy(y_hat, y_output_hat)+ categorical_crossentropy(y_hat2, y_output_hat2)

def main():
    eval = True
    screen_width = 192 #264
    screen_height = 192 #64
    crop = 128
    rgbd = True
    object_type = obj
    env = robosuite.make(
        "BaxterPush",
        bin_type='table',
        object_type=object_type,
        ignore_done=True,
        has_renderer=bool(render),  # True,
        has_offscreen_renderer=not bool(render),  # added this new line
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

     #'data/processed_data'
    model = ResNet_LwF(task, 10, model_name='ResNet', batch_size=batch_size, repetition =[3,4,6,3])
    model.set_datapath(args.task_num_old, args.task_num_new, args.task_num_newer, data_path, data_path2, data_path3)
    model.set_env(env, 0, init=1)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    # with tf.Session(config=gpu_config) as sess:

    model.train(task_order=0)

    model.save_task_logits(0,args.task_num_old,args.task_num_new)
    model.train(1)

    # model.save_task_logits(0,args.task_num_old,args.task_num_newer)
    # model.save_task_logits(1,args.task_num_new,args.task_num_newer)
    # model.train(2)


if __name__=='__main__':
    main()
