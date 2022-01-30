import tensorflow as tf
slim = tf.contrib.slim
graph_replace = tf.contrib.graph_editor.graph_replace

import datetime
import sys, os
sys.path.extend([os.path.expanduser('..')])
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..'))
sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
from cont_learning_picking import *
from pathint import utils
from data_loader import CustomDataLoader
import seaborn as sns
sns.set_style("white")
import numpy as np

from tqdm import trange, tqdm

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Network params
activation_fn = tf.nn.relu

# Optimization params
file_size = 1024 #512
batch_size = 64
num_epochs = 10
learning_rate = 1e-4
xi = 0.1

# Reset optimizer after each age
reset_optimizer = False

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Add, Lambda
from keras.layers import Conv2D, MaxPooling2D, Input, Multiply, Reshape, BatchNormalization, AveragePooling2D
import keras.activations as activations
import keras.backend as K
from keras.regularizers import l2

def get_shortcut(inputs, residual):
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
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.001))(inputs) #0.0001
    return shortcut

def block_lv(name, inputs, num_filters, init_strides, is_first_block=False):
    if is_first_block:
        x = Conv2D(num_filters, (3, 3), padding='same', name=name + 'conv_1')(inputs)
    else:
        x = BatchNormalization(axis=-1)(inputs)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, (3, 3), strides=init_strides, padding='same', name=name + 'conv_1')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same', name=name + 'conv_2')(x)
    shortcut = get_shortcut(inputs, x)
    return Add()([shortcut, x])

def res_block(inputs, num_filters, repetition, group_idx, is_first_layer=False):
    x = inputs
    for i in range(repetition):
        init_strides = (1, 1)
        if i==0 and not is_first_layer:
            init_strides = (2,2)
        x = block_lv('group_%d_block_%d_'%(group_idx, i), x, num_filters, init_strides, (i==0) and is_first_layer)
    return x


def ResNet(_input_shape, output_dim):
    inputs = Input(shape=_input_shape)
    repetitions = [3, 4, 6, 3]

    x = Conv2D(64, (7, 7), strides=(2,2), padding='same', name='first_conv')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    filters = 64
    for i, r in enumerate(repetitions):
        x = res_block(x, filters, r, i, is_first_layer=(i == 0))
        filters *= 2

    # Last activation
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    # Classifier block
    x_shape = K.int_shape(x)
    pool2 = AveragePooling2D(pool_size=(x_shape[1], x_shape[2]))(x)
    x = Flatten()(x)
    outputs = Dense(units=output_dim, kernel_initializer="he_normal",
                  activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

from pathint import protocols
from pathint.optimizers import KOOptimizer
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback
from pathint.keras_utils import LossHistory
import psutil

class MyCallback(Callback):
    def __init__(self, task, render=True):
        super().__init__()
        self.num_test_ep = 20
        screen_width = 192
        screen_height = 192
        crop = 128
        rgbd = True

        random_spawn = False
        action_type = '2D'

        object_type = 'smallcube' if task == 'pick' or task == 'place' else 'cube'
        num_blocks = 2 if task == 'place' else 1
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
        self.env = BaxterEnv(env, task=task, render=render, using_feature=False, random_spawn=random_spawn, rgbd=rgbd,
                        action_type=action_type)
        action_size = self.env.action_size
        self.success_rate = []

    def test_agent(self):
        success_log = []
        for n in range(self.num_test_ep):
            obs = self.env.reset()
            done = False
            cumulative_reward = 0.0
            step_count = 0
            while not done:
                step_count += 1
                clipped_obs = np.clip(obs, 0.0, 5.0)
                s_0 = clipped_obs[0, :, :, :].copy()
                s_1 = clipped_obs[1, :, :, :].copy()
                state = np.concatenate([s_0, s_1], axis=-1)

                action = self.model.predict(state[np.newaxis, :])
                obs, reward, done, _ = self.env.step(np.argmax(action))
                cumulative_reward += reward

            if cumulative_reward >= 90:
                success_log.append(1)
            else:
                success_log.append(0)

            print(step_count, cumulative_reward)

        print(success_log)
        print('success rate?:', np.mean(success_log))
        return np.mean(success_log)

    def on_batch_end(self, batch, logs):
        process = psutil.Process(os.getpid())
        print(process.memory_percent())

    def on_epoch_end(self, epoch, logs):
        self.success_rate.append(self.test_agent())


def run_fits(cvals, task, training_generator, validation_generator, eval_on_train_set=False):
    if task == 'reach' or task == 'push':
        output_dim = 8
    elif task == 'pick':
        output_dim = 12
    elif task == 'place':
        output_dim = 10
    else:
        print('Wrong TASK!!')
        sys.exit()

    model = ResNet([128, 128, 8], output_dim)
    model._ckpt_saved_epoch = None
    # print(model.summary())
    protocol_name, protocol = protocols.PATH_INT_PROTOCOL(omega_decay='sum', xi=xi)
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    opt_name = 'adam'

    oopt = KOOptimizer(opt, model=model, **protocol)
    model.compile(loss="categorical_crossentropy", optimizer=oopt, metrics=['accuracy'])

    diag_vals = dict()
    all_evals = dict()
    reg = 5e-4

    history = LossHistory()
    now = datetime.datetime.now()
    model_name = "models/" + task + '_' + now.strftime("%m%d_%H%M")
    os.makedirs(model_name)
    checkpoint_path = model_name + "/{epoch:02d}-{val_acc:.2f}.hdf5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_best_only=False, period=1)
    mycallback = MyCallback(task, render=False)
    callbacks = [history, cp_callback, mycallback]
    for cidx, cval_ in enumerate(cvals):
        evals = []
        sess.run(tf.global_variables_initializer())
        cstuffs = []
        cval = cval_
        print("setting cval")
        oopt.set_strength(cval)
        print("cval is %e" % sess.run(oopt.lam))

        for age, tidx in enumerate(range(1)): #range(n_tasks)
            print("Age %i, cval is=%f" % (age, cval))

            stuffs = model.fit_generator(training_generator, epochs=num_epochs, validation_data=validation_generator, callbacks=callbacks, verbose=1)

            ftask = []
            f_ = model.evaluate_generator(validation_generator, verbose=1)
            ftask.append(np.mean(f_[1]))

            evals.append(ftask)
            cstuffs.append(stuffs)

            # Re-initialize optimizer variables
            if reset_optimizer:
                oopt.reset_optimizer()

        evals = np.array(evals)
        all_evals[cval_] = evals

        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(stuffs.history['loss'], 'y', label='train loss')
        loss_ax.plot(stuffs.history['val_loss'], 'r', label='val loss')
        acc_ax.plot(stuffs.history['acc'], 'b', label='train acc')
        acc_ax.plot(stuffs.history['val_acc'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        fig.savefig(model_name + "/plot.png")

        fig, ax = plt.subplots()
        ax.plot(mycallback.success_rate, 'r', label='success_rate')
        ax.set_xlabel('epoch')
        ax.set_ylabel('success rate')
        # ax.legend(loc='upper left')
        fig.savefig(model_name + "/performance.png")


if __name__=='__main__':
    flags = tf.app.flags
    flags.DEFINE_string('task', 'reach', 'name of task: [ reach / push / pick / place ]')
    flags.DEFINE_string('data_path', None, 'data path')
    flags.DEFINE_string('data_type', 'pkl', 'type of data file: [ pkl / npy ]')
    flags.DEFINE_integer('num_data', 0, 'number of data files to use')
    FLAGS = flags.FLAGS

    task = FLAGS.task
    data_path = FLAGS.data_path
    data_type = FLAGS.data_type
    num_data = FLAGS.num_data

    training_generator = CustomDataLoader(task, data_path, data_type, validation=False, batch_size=batch_size,
                                          file_size=file_size, shuffle=True, n_files=num_data)
    validation_generator = CustomDataLoader(task, data_path, data_type, validation=True, batch_size=batch_size,
                                            file_size=file_size, shuffle=False, n_files=num_data)

    cvals = [0.1]
    print(cvals)

    run_fits(cvals, FLAGS.task, training_generator, validation_generator)

    # model.save('models/' + model_name)
    # utils.save_zipped_pickle(all_evals, datafile_name)