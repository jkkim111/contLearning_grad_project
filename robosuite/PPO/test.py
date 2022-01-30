import os
import sys
import random
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(FILE_PATH, '../..'))
#sys.path.append(os.path.join(FILE_PATH, '..', 'scripts'))
#from demo_baxter_rl_pushing import *


task = 'pick-place-v1' #'push'
render = True
using_feature = False #True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU is not efficient here

'''
screen_width = 192 #64 #96
screen_height = 192 #64 #96
crop = 128 #None #64 #None

env = robosuite.make(
    "BaxterPush",
    bin_type='table',
    object_type='cube',
    ignore_done=True,
    has_renderer=True,
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
env = BaxterEnv(env, task=task, render=render, using_feature=using_feature, action_type='2D', random_spawn=True)
'''
import metaworld
ml1 = metaworld.ML1(task)
env = ml1.train_classes['pick-place-v1']()
task = random.choice(ml1.train_tasks)
env.set_task(task)
env.global_done = False
env.action_dim = env.action_space.n

obs = env.reset()
for _ in range(1000):
    action = int(input('action? '))
    # action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Action: {},  reward: {}".format(action, reward))
    env.render()
    #print('arm pos:', env.arm_pos)
    #print('obj pos:', env.obj_pos)
    if done:
        obs = env.reset()
        print('Episode restart.')
