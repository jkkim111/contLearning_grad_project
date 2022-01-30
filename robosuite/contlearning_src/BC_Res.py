import os, sys
import argparse

## argument parsing ##
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=int, default=1)
parser.add_argument('--task', type=str, default="pick")
parser.add_argument('--save_view', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--viewpoint1', type=str, default="rlview1")
parser.add_argument('--viewpoint2', type=str, default="rlview2")
parser.add_argument('--viewpoint3', type=str, default="birdview")
parser.add_argument('--data_path', type=str, default="data")
parser.add_argument('--data_path2', type=str, default="data")
parser.add_argument('--data_path3', type=str, default="data")
parser.add_argument('--load_model', type=int, default=0)
parser.add_argument('--load_loc', type=str, default="binary_crossentropy")
parser.add_argument('--folder_name', type=str, default="pick_ResNet_10dim")
parser.add_argument('--only_test', type=int, default=0)
parser.add_argument('--data_alpha', type=float, default=1.0)
parser.add_argument('--data_beta', type=float, default=1.0)
parser.add_argument('--lrdecay', type=float, default=0.02)
parser.add_argument('--network', type=str, default='ResNet')
parser.add_argument('--mov_dist', type=float, default=0.02)
parser.add_argument('--task_num_train', type=int, default=0)
parser.add_argument('--task_num_eval', type=int, default=0)

parser.add_argument('--grasp_env', type=int, default=1)
parser.add_argument('--down_level', type=int, default=1)
parser.add_argument('--max_pool', type=int, default=1)

parser.add_argument('--obj', type=str, default="smallcube")
parser.add_argument('--num_episodes', type=int, default=50)
parser.add_argument('--action_type', type=str, default="2D")
parser.add_argument('--random_spawn', type=int, default=0)
parser.add_argument('--block_random', type=int, default=1)
parser.add_argument('--num-blocks', type=int, default=1)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--loss_fn', type=str, default="binary_crossentropy")
parser.add_argument('--reg', type=str, default=None)
parser.add_argument('--reg_coeff', type=float, default=None)
parser.add_argument('--starting_epoch', type=int, default=0)
parser.add_argument('--epoch_dsize', type=int, default=1280)


args = parser.parse_args()
render = bool(args.render)
task = args.task
task_num_train = args.task_num_train
task_num_eval = args.task_num_eval

save_view = args.save_view
batch_size = args.batch_size
viewpoint1 = args.viewpoint1
viewpoint2 = args.viewpoint2
viewpoint3 = args.viewpoint3
data_path = args.data_path
data_path2 = args.data_path2
data_path3 = args.data_path3
load_model = bool(args.load_model)
load_loc = args.load_loc
folder_name = args.folder_name
only_test = args.only_test
data_alpha = args.data_alpha
data_beta = args.data_beta
lrdecay = args.lrdecay
network = args.network
mov_dist = args.mov_dist

grasp_env = args.grasp_env
down_level = args.down_level
max_pool = args.max_pool

obj = args.obj
num_episodes = args.num_episodes
action_type = args.action_type
random_spawn = bool(args.random_spawn)
block_random = bool(args.block_random)
num_blocks = args.num_blocks

lr= args.lr
loss_fn= args.loss_fn
reg = args.reg
reg_coeff = args.reg_coeff
starting_epoch = args.starting_epoch
epoch_dsize = args.epoch_dsize

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
from keras.layers import Dense, Dropout, Activation, Flatten, Add, Lambda
from keras.layers import Conv2D, MaxPooling2D, Input, Multiply, Reshape, BatchNormalization, AveragePooling2D
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

class ResNet:
    def __init__(self, task, action_size, model_name=None, batch_size=50, repetition = [3,3,4]):
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
        self.test_freq = 1
        self.eval_freq = 6
        self.eval_from = 1
        self.num_test_ep = 10
        self.env = None
        self.batch_size = batch_size
        self.repetition = repetition
        self.s_test = None
        self.a_test = None
        self.only_test = only_test
        self.data_alpha = data_alpha
        self.data_beta = data_beta
        self.task_num_eval = None

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
        self.checkpoint_dir = os.path.join(FILE_PATH, 'bc_train_log/', 'keras_saved_model/')
        self.model = None
        self.folder_name = folder_name+'_'+self.now.strftime("%m%d_%H%M%S")

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
        # inputs1 = Input(inputs[0,:,:,:])
        # inputs2 = Input(inputs[1, :, :, :])
        # inputs = K.layers.concatenate([inputs1, inputs2], axis=-1)

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
        x = Flatten()(x)
        output_action = Dense(units=output_dim,
                        activation="softmax")(x)

        model = Model(inputs=inputs, outputs=output_action)
        return model


    def simpleCNN(self, _input_shape, output_dim):
        inputs = Input(shape=_input_shape)
        x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer="glorot_normal",kernel_regularizer= reg,name='first_conv')(inputs)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer="glorot_normal",kernel_regularizer= reg,name='second_conv')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer="glorot_normal",kernel_regularizer= reg,name='third_conv')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer="glorot_normal",kernel_regularizer= reg,name='fourth_conv')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        x = Flatten()(x)
        x = Dense(units=256,activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(units=128,activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        output_action = Dense(units=output_dim,activation="softmax")(x)

        model = Model(inputs=inputs, outputs=output_action)
        return model

    def set_datapath(self, data_path, data_path2, data_path3='None', data_type='pkl'): # data_type='npy' / 'pkl'
        #set path to data and load the collected data to self.a_list and self.s_list
        self.data_type = data_type
        file_list = os.listdir(os.path.join(os.getcwd(),data_path+'/pkls'))
        data_list = [f for f in file_list if data_type in f and self.task in f]
        file_list2 = os.listdir(os.path.join(os.getcwd(), data_path2 + '/pkls'))
        data_list2 = [f for f in file_list2 if data_type in f and self.task in f]
        file_list3 = os.listdir(os.path.join(os.getcwd(), data_path3 + '/pkls'))
        data_list3 = [f for f in file_list3 if data_type in f and self.task in f]
        if len(data_list) == 0:
            print('No data files exist. Wrong data path!!')
            return
        self.data_list = sorted([os.path.join(data_path, p) for p in data_list])
        self.a_list1 = sorted([p for p in data_list if self.task + '_a_' in p])
        self.s_list1 = sorted([p for p in data_list if self.task + '_s_' in p])
        self.a_list2 = sorted([p for p in data_list2 if self.task + '_a_' in p])
        self.s_list2 = sorted([p for p in data_list2 if self.task + '_s_' in p])
        self.a_list3 = sorted([p for p in data_list3 if self.task + '_a_' in p])
        self.s_list3 = sorted([p for p in data_list3 if self.task + '_s_' in p])

        self.a_list2, self.s_list2 = shuffle(self.a_list2, self.s_list2)
        self.a_list3, self.s_list3 = shuffle(self.a_list3, self.s_list3)

        self.a_list, self.s_list = self.a_list1, self.s_list1

        #add all datasets
        count = self.data_alpha
        while True:
            if count >=1.0:
                self.a_list = self.a_list + self.a_list2
                self.s_list = self.s_list + self.s_list2
                count -=1
            else:
                self.a_list = self.a_list + self.a_list2[:int(len(self.a_list2)*count)]
                self.s_list = self.s_list + self.s_list2[:int(len(self.s_list2)*count)]
                break

        count = self.data_beta
        while True:
            if count >=1.0:
                self.a_list = self.a_list + self.a_list3
                self.s_list = self.s_list + self.s_list3
                count -=1
            else:
                self.a_list = self.a_list + self.a_list3[:int(len(self.a_list3)*count)]
                self.s_list = self.s_list + self.s_list3[:int(len(self.s_list3)*count)]
                break

        # print(self.a_list)
        print(self.s_list)
        assert len(self.a_list) == len(self.s_list)
        self.data_path = data_path+'/pkls'
        self.data_path2 = data_path2+'/pkls'
        self.data_path3 = data_path3+'/pkls'

        if not os.path.exists(os.path.join(FILE_PATH,'bc_train_log/failed',self.folder_name)):
            os.makedirs(os.path.join(FILE_PATH,'bc_train_log/failed',self.folder_name))

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
            self.task_num_eval = task_num
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
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = mov_dist, object_type = "smallcube")

    def load_pkl(self, pkl_file):
        if pkl_file in self.a_list1+self.s_list1 :
            with open(os.path.join(self.data_path, pkl_file), 'rb') as f:
                return pickle.load(f)
        elif pkl_file in self.a_list2+self.s_list2 :
            with open(os.path.join(self.data_path2, pkl_file), 'rb') as f:
                return pickle.load(f)
        elif pkl_file in self.a_list3+self.s_list3 :
            with open(os.path.join(self.data_path3, pkl_file), 'rb') as f:
                return pickle.load(f)

    def load_npy(self, npy_file):
        return np.load(os.path.join(self.data_path, npy_file))

    def test_agent(self, num_rep = 10, task_num=0):
        if self.task_num_eval != task_num:
            self.set_env(self.env, task_num)

        print("EVAL for TASK  %d  STARTED"%(task_num))
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
                action = np.argmax(self.model.predict(np.expand_dims(obs_re, axis=0)))
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
                im1.save(os.path.join(FILE_PATH, 'bc_train_log', 'failed/'+self.folder_name+'/test_'
                                      + datetime.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(stucked)+'-view1.png'))
                im2.save(os.path.join(FILE_PATH, 'bc_train_log', 'failed/'+self.folder_name+'/test_'
                                      + datetime.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(stucked)+'-view2.png'))


            print("Episode : "+ str(n) + " / Step count : "+ str(step_count) + " / Cum Rew : " + str(cumulative_reward))
            action_dist = np.zeros((self.env.action_size), dtype=int)
            for ind in range(self.env.action_size):
                action_dist[ind] = action_his.count(ind)
            print('action distribution (this episode) :')
            print(list(action_dist))
        print('SUCCESS RATE on picking?:', np.mean(success_log))
        print('test outcome dist (success, wrong grasp, not going down,  , max step,  , out-bound, stuck):')
        print(list(failed_case_dist))
        print()

        return np.mean(success_log), failed_case_dist

    def train1(self):
        print('Training starts..')
        folder_name = self.folder_name
        vp3 = 0 if viewpoint3 == "None" else 1
        #load or make a keras model
        if load_model:
            # self.model = self.ResNetwork(_input_shape=[128, 128, 4 * (2 + vp3)], output_dim=self.env.action_size,
            #                              repetition=self.repetition)
            # self.model.load_weights(os.path.join(FILE_PATH, 'bc_train_log', 'keras_saved_model', load_loc))
            # self.model = LoadModel(os.path.join(FILE_PATH, 'bc_train_log', 'keras_saved_model', load_loc))
            self.model = self.ResNetwork(_input_shape=[128, 128, 4*(2+vp3)], output_dim=self.env.action_size, repetition=self.repetition)

            self.model.load_weights(os.path.join(FILE_PATH, 'bc_train_log', 'keras_saved_model', load_loc+".h5"))

            self.folder_name = load_loc.replace(".h5", '') if ".h5" in load_loc else load_loc
            folder_name = self.folder_name
            optimizer = Adam(lr=self.lr)
            print("loaded the saved keras model!")
            # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        elif self.model is None:
            if self.network == self.simpleCNN:
                self.model = self.simpleCNN(_input_shape=[128, 128, 4*(2+vp3)], output_dim=self.env.action_size)
            elif self.network == self.ResNetwork:
                self.model = self.ResNetwork(_input_shape=[128, 128, 4*(2+vp3)], output_dim=self.env.action_size, repetition=self.repetition)
            self.folder_name = folder_name

            optimizer = Adam(lr=self.lr, decay =self.lrdecay)
            self.model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics= ['accuracy'])

        #tensorboard and mkdir
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(os.path.join(FILE_PATH,'bc_train_log','failed')):
            os.makedirs(os.path.join(FILE_PATH, 'bc_train_log', 'failed'))

        #test only
        if self.only_test:
            success_rate, _ = self.test_agent(50)
            writer.add_scalar('train-%s/success_rate' % self.task, success_rate, starting_epoch + epoch + 1)
            print("only test finished!")
            return

        #load data
        if self.data_type=='pkl':
            load_data = self.load_pkl
        elif self.data_type == 'npy':
            load_data = self.load_npy
        data_size = len(load_data(self.a_list[0]))

        self.a_list, self.s_list = shuffle(self.a_list, self.s_list)

        train_dataset = CustomDataset(self.s_list[:-10], self.a_list[:-10],
                                (self.s_list1,self.s_list2, self.s_list3),
                                (self.a_list1,self.a_list2, self.a_list3),
                                (self.data_path,self.data_path2,self.data_path3),
                                self.batch_size, self.env.action_size)
        val_dataset = CustomDataset(self.s_list[-10:], self.a_list[-10:],
                                (self.s_list1,self.s_list2, self.s_list3),
                                (self.a_list1,self.a_list2, self.a_list3),
                                (self.data_path,self.data_path2,self.data_path3),
                                1280, self.env.action_size)

        # print(len(dataset), tound(0.9*len(dataset)),round(0.1*len(dataset)))
        # train_dataset, val_dataset = random_split(dataset, [round(0.9*len(dataset)),round(0.1*len(dataset))])

        # train_loader = DataLoader(train_dataset, batch_size=256)
        # val_loader = DataLoader(val_dataset, batch_size=1)
        # print(next(iter(val_loader)))
        # val_dataset = CustomDataset(test_s_list, test_a_list,(self.data_path,self.data_path2,self.data_path3))
        # val_dataloader = DataLoader(val_dataset, batch_size=256)

        # for epoch in range(self.num_epochs+3):
            # print()
            # print('[Epoch %d]' % epoch)

            #shuffle the dataset before each epoch and iterate over the division of the dataset
            # ep_train_acc = []
            # ep_loss = []

        # print("train_s_data", np.shape(train_s_data[0]))
        mycallbacks= myCallback(self.test_agent, self.eval_freq, self.num_epochs)
        best_model = ModelCheckpoint(filepath = os.path.join(FILE_PATH, 'bc_train_log/', 'keras_saved_model', folder_name+".h5"), monitor='val_acc', save_best_only=True, verbose=0)

        print(train_dataset[0][1])
        hist = self.model.fit_generator(generator = train_dataset,
                              epochs=self.num_epochs,
                              steps_per_epoch = len(train_dataset)//256,
                              validation_data=val_dataset[0],
                              callbacks=[mycallbacks, best_model])

            #evaluate and append accuracy
            # test_accuracy1 = self.model.evaluate(self.s_test, self.a_test)
            # test_accuracy = test_accuracy1[1]
            # print("test accuracy for EP {} is :".format(epoch), test_accuracy)



            # save the model parameters
            # if test_accuracy > self.max_accur:
            #     self.model.save(os.path.join(FILE_PATH, 'bc_train_log/', 'keras_saved_model', folder_name+".h5"))
            #     self.max_accur = test_accuracy


        print('Training done!')
        return


    def schedule_lr(self, epoch, round, num_rounds, decay_rate = 0.1):
        initial_lrate = self.lr
        k = decay_rate
        lrate = initial_lrate * np.exp(-k*(epoch+float(round)/num_rounds))
        print("learning rate for this round is : ", lrate)
        return lrate

    def train(self):
        print('Training starts..')
        folder_name = self.folder_name
        vp3 = 0 if viewpoint3 == "None" else 1
        #load or make a keras model
        if load_model:
            # self.model = self.ResNetwork(_input_shape=[128, 128, 4 * (2 + vp3)], output_dim=self.env.action_size,
            #                              repetition=self.repetition)
            # self.model.load_weights(os.path.join(FILE_PATH, 'bc_train_log', 'keras_saved_model', load_loc))
            # self.model = LoadModel(os.path.join(FILE_PATH, 'bc_train_log', 'keras_saved_model', load_loc))
            self.model = self.ResNetwork(_input_shape=[128, 128, 4*(2+vp3)], output_dim=self.env.action_size, repetition=self.repetition)

            self.model.load_weights(os.path.join(FILE_PATH, 'bc_train_log', 'keras_saved_model', load_loc+".h5"))

            self.folder_name = load_loc.replace(".h5", '') if ".h5" in load_loc else load_loc
            folder_name = self.folder_name
            optimizer = Adam(lr=self.lr)
            print("loaded the saved keras model!")
            # self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        elif self.model is None:
            if self.network == self.simpleCNN:
                self.model = self.simpleCNN(_input_shape=[128, 128, 4*(2+vp3)], output_dim=self.env.action_size)
            elif self.network == self.ResNetwork:
                self.model = self.ResNetwork(_input_shape=[128, 128, 4*(2+vp3)], output_dim=self.env.action_size, repetition=self.repetition)
            self.folder_name = folder_name

            optimizer = Adam(lr=self.lr, decay =self.lrdecay)
            self.model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics= ['accuracy'])

        #tensorboard and mkdir
        writer = SummaryWriter(os.path.join(FILE_PATH, 'bc_train_log/', 'tensorboard', folder_name))
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(os.path.join(FILE_PATH,'bc_train_log','failed')):
            os.makedirs(os.path.join(FILE_PATH, 'bc_train_log', 'failed'))

        #test only
        if self.only_test:
            success_rate, _ = self.test_agent(50)
            writer.add_scalar('train-%s/success_rate' % self.task, success_rate, starting_epoch + epoch + 1)
            print("only test finished!")
            return

        #load data
        if self.data_type=='pkl':
            load_data = self.load_pkl
        elif self.data_type == 'npy':
            load_data = self.load_npy
        data_size = len(load_data(self.a_list[0]))

        self.a_list, self.s_list = shuffle(self.a_list, self.s_list)

        hist= None
        if self.a_test is None:
            s_test = np.empty([0,128,128,4*(2+vp3)], float)
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
                train_s_data = np.empty([0,128,128,4*(2+vp3)], float)
                train_a_data = np.empty([0,], int)

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
                # fit
                if np.shape(train_s_data)[-1] ==4:
                    train_s_data = np.concatenate([train_s_data[:,k,:,:,:] for k in range(np.shape(train_s_data)[1])], axis=-1)

                train_a_data = np.asarray([np.eye(self.env.action_size)[int(k)] for k in train_a_data], dtype=int)
                # print("train_s_data", np.shape(train_s_data[0]))
                K.set_value(self.model.optimizer.lr, self.schedule_lr(epoch, round, num_rounds, self.lrdecay))
                hist = self.model.fit(train_s_data,
                                      train_a_data,
                                      epochs=1, verbose=0,
                                      batch_size=self.batch_size, validation_split=0.0)
                ep_train_acc += hist.history["acc"]
                ep_loss += hist.history["loss"]

            #evaluate and append accuracy
            ep_train_acc = np.sum(ep_train_acc)/num_rounds
            ep_loss = np.sum(ep_loss)/num_rounds
            test_accuracy1 = self.model.evaluate(self.s_test, self.a_test)
            test_accuracy = test_accuracy1[1]
            print("test accuracy for EP {} is :".format(epoch), test_accuracy)

            #append
            writer.add_scalar('train-%s/test_accuracy' % self.task, test_accuracy, starting_epoch+epoch + 1)
            writer.add_scalar('train-%s/train_cost'%self.task, ep_loss, starting_epoch+epoch+1)
            writer.add_scalar('train-%s/train_accuracy'%self.task, ep_train_acc, starting_epoch+epoch+1)


            print('cost: %.3f\t/   train accur: %.3f\t/   test accur:[%.3f]' \
                  % (ep_loss,
                     ep_train_acc,
                     test_accuracy))

            # save the model parameters
            if test_accuracy > self.max_accur:
                self.model.save(os.path.join(FILE_PATH, 'bc_train_log/', 'keras_saved_model', folder_name+".h5"))
                self.max_accur = test_accuracy

            # performance evaluation
            if  (epoch+1) % self.eval_freq == 0: #(epoch+1 > self.eval_from) and
                if (epoch+1) == self.num_epochs:
                    if task_num_eval == 5:
                        if task_num_train in [1,2]:
                            task_range=[0,1,2]
                        elif task_num_train in [3,4]:
                            task_range = [0,3,4]
                        else:
                            task_range = [0,1,2,3,4]
                        for i in task_range:
                            success_rate, _ = self.test_agent(50, i)
                            writer.add_scalar('train-%s/success_rate_task%d' % (self.task, i), success_rate, epoch + 1)
                    else:
                        success_rate, _ = self.test_agent(50, task_num_eval)
                        writer.add_scalar('train-%s/success_rate_task%d' % (self.task, task_num_eval), success_rate, epoch + 1)

                else:
                    success_rate, _ = self.test_agent(10, task_num_eval)
                    writer.add_scalar('train-%s/success_rate' % (self.task), success_rate, epoch + 1)



        print('Training done!')
        return


class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_agent, eval_freq, num_epochs, writer=None):
        super().__init__()
        self.test_agent = test_agent
        self.num_epochs= num_epochs
        self.eval_freq = eval_freq
        self.writer = SummaryWriter(os.path.join(FILE_PATH, 'bc_train_log/', 'tensorboard', folder_name+'_'+datetime.datetime.now().strftime("%m%d_%H%M%S")))

    def on_epoch_end(self, epoch, logs={}):
        self.writer.add_scalar('train-%s/test_accuracy' % self.task, logs.get("val_acc"), epoch + 1)
        self.writer.add_scalar('train-%s/train_cost'%self.task, logs.get("loss"), epoch+1)
        self.writer.add_scalar('train-%s/train_accuracy'%self.task, logs.get("acc"), epoch+1)
        print('cost: %.3f\t/   train accur: %.3f\t/   test accur:[%.3f]' \
              % (logs.get("loss"), logs.get("acc"),logs.get("val_acc")))
        if  (epoch+1) % self.eval_freq == 0: #(epoch+1 > self.eval_from) and
            if (epoch+1) == self.num_epochs:
                success_rate, _ = self.test_agent(50)
            else:
                success_rate, _ = self.test_agent(10)
            self.writer.add_scalar('train-%s/success_rate' % self.task, success_rate, epoch + 1)


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, s_list, a_list, s_lists, a_lists, data_paths, batch_size, action_dim, shuffle=True):
        self.batch_size =  batch_size
        self.s_list= s_list
        self.a_list= a_list
        self.data_path = data_paths[0]
        self.data_path2 = data_paths[1]
        self.data_path3 = data_paths[2]
        self.s_list1= s_lists[0]
        self.s_list2= s_lists[1]
        self.s_list3= s_lists[2]
        self.a_list1= a_lists[0]
        self.a_list2= a_lists[1]
        self.a_list3= a_lists[2]
        self.shuffle=shuffle
        self.on_epoch_end()
        self.data_size = len(self.load_pkl(self.a_list[0]))
        self.action_dim = action_dim

    def __len__(self):
    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분

        print(np.shape(self.load_pkl(self.s_list[0])))
        len_out = len(self.a_list)*self.data_size//self.batch_size
        # print("the len is {}".format(len_out))
        return len_out

    def __getitem__(self, idx):
  #데이터셋에서 특정 1개의 샘플을 가져오는 함수
        reps = int(self.batch_size / self.data_size)
        print(self.batch_size, self.data_size)
        train_s_data = np.empty([0,128,128,8], float)
        train_a_data = np.empty([0,], int)
        for p_idx in range(reps):  # -1
            buff_actions = self.load_pkl(self.a_list[int(reps*idx+p_idx)])
            buff_states = self.load_pkl(self.s_list[int(reps*idx+p_idx)])
            # print("just loaded : ", buff_actions)
            train_s_data = np.append(train_s_data,buff_states,axis=0)
            train_a_data = np.append(train_a_data,buff_actions,axis=0)

        print("the index {} has shape {},{}".format(idx,np.shape(train_s_data),np.shape(train_a_data)))
        return self.pres_s(train_s_data, train_a_data)

    def on_epoch_end(self):
        if self.shuffle:
            self.s_list, self.a_list = shuffle(self.s_list, self.a_list)

    def load_pkl(self, pkl_file):
        if pkl_file in self.a_list1+self.s_list1 :
            with open(os.path.join(self.data_path, pkl_file), 'rb') as f:
                return pickle.load(f)
        elif pkl_file in self.a_list2+self.s_list2 :
            with open(os.path.join(self.data_path2, pkl_file), 'rb') as f:
                return pickle.load(f)
        elif pkl_file in self.a_list3+self.s_list3 :
            with open(os.path.join(self.data_path3, pkl_file), 'rb') as f:
                return pickle.load(f)

    def load_npy(self, npy_file):
        return np.load(os.path.join(self.data_path, npy_file))

    def pres_s(self, imgdata, label):
        if np.shape(imgdata)[-1]== 4:
            out =  np.concatenate([imgdata[:, k, :, :, :] for k in range(np.shape(imgdata)[1])],axis=-1)
        else:
            out = imgdata
        label = np.asarray([np.eye(self.action_dim)[k] for k in label])
        return out, label


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
    model = ResNet(task, 10, model_name='ResNet', batch_size=batch_size, repetition =[3,4,6,3])
    model.set_datapath(data_path, data_path2, data_path3, data_type='pkl')
    model.set_env(env, 0, init=1)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    # with tf.Session(config=gpu_config) as sess:
    model.train()


if __name__=='__main__':
    main()
