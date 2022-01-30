import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=int, default=1)
parser.add_argument('--tag', type=str, default="torch")

#task config
parser.add_argument('--task_num_list', type = int, nargs='+', required=True)

# train-test config
parser.add_argument('--only_test', type=int, default=0)
parser.add_argument('--only_train', type=int, default=0)
parser.add_argument('--load_data_name', type=str, default=0)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--num_test_ep', type=int, default=10)

#network config
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--betas', type=float, nargs=2, default=[0.9,0.999])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lambda_', type=float, default=0.4)

args = parser.parse_args()

import os, sys
os.environ["CYMJ_RENDER"] = "0" if not args.render else "1"

import yaml
config = yaml.load(open('./config_start.yaml'), Loader=yaml.FullLoader)
config['view']['render'] = args.render
config['experiment']['tag'] = args.tag
config['experiment']['task_num_list'] = args.task_num_list
config['eval']['only_test'] = args.only_test
config['eval']['load_data_name'] = args.load_data_name
config['eval']['num_test_ep'] = args.num_test_ep
config['network']['num_epochs'] = args.num_epochs
config['network']['lr'] = args.lr
config['network']['weight_decay'] = args.weight_decay
config['network']['betas'] = args.betas
config['network']['batch_size'] = args.batch_size
config['network']['lambda'] = args.lambda_

FILE_PATH = os.getcwd()
import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import time, datetime
import copy
import pickle
import matplotlib.pyplot as plt
import random

if not args.only_train :
    from cont_learning_picking import *
    from ops import *

reset_optimizer = False
# tf.reset_default_graph()
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.InteractiveSession(config=config)
# sess.run(tf.global_variables_initializer())

def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
	labels = Variable(labels.data, requires_grad=False).cuda()
	outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(labels/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
	return Variable(outputs.data, requires_grad=True).cuda()

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class lwfDataset(Dataset):
    """ lwf dataset of s-a pairs."""

    def __init__(self, root_dir=None, task=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.task= task
        self.root_dir = root_dir if root_dir is not None else os.path.join(os.getcwd(),f'data_10dim_task{self.task}_sliced')
        self.list_dir = os.listdir(self.root_dir)
        
    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(os.path.join(self.root_dir,self.list_dir[idx]), 'rb') as f:
            sample = pickle.load(f)
        return {'state':np.rollaxis(sample[0],2,0).astype(np.float32), 'action':sample[1]}

    
class lwf_network(nn.Module):
    def __init__(self, config_network, action_size=10, config_dir = None, checkpoint_dir = None, folder_name = None):
        super(lwf_network, self).__init__()
        model = models.resnet34(pretrained=False)
#         print(self.model)
        model.apply(kaiming_normal_init)
        num_features = model.fc.in_features
        
        self.task_num_list = []
        self.action_size = action_size
        
        model.fc = nn.Linear(num_features, 1*self.action_size, bias=False)
        self.fc = model.fc
        
#         print(*list(self.model.children())[:1])
        first_conv = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_extractor = nn.Sequential(first_conv,*list(model.children())[1:-1])
#         self.feature_extractor = nn.Sequential(first_conv,*list(self.model.children())[:-1])
        self.feature_extractor = nn.DataParallel(self.feature_extractor)
        
        self.config = config_network
        self.num_epochs=config_network['num_epochs']
        self.batch_size = config_network['batch_size']
        self.lambda_ = config_network['lambda']
        self.softmax = nn.Softmax()
        self.config_dir = config_dir
        self.checkpoint_dir = checkpoint_dir
        self.folder_name = folder_name

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _add_task(self, task_num):
        """Add task to the final fc layer
        and add to the task_num_list"""
        if len(self.task_num_list) !=0:
            in_features = self.fc.in_features
            out_features = self.fc.out_features

            #save existing weights
            weight = self.fc.weight.data
            new_out_features = out_features + self.action_size

            #make new fc layer
            self.fc = nn.Linear(in_features, new_out_features, bias=False)

            kaiming_normal_init(self.fc.weight)
            self.fc.weight.data[:out_features] = weight
        self.task_num_list.append(task_num)
        print(f"Added task to the list - it's now {self.task_num_list}")

    def classify(self, images):
        """
        Take actions for each tasks

    Args:
            x: input image batch
        Returns:
            preds: actions for each task
        """
        output_ = self.forward(images)
#         print(output_)
        preds=[0]*len(self.task_num_list)
        for i in range(len(self.task_num_list)):
            preds[i] = torch.argmax(self.softmax(output_[i*self.action_size:(i+1)*self.action_size]),dim=1).item()
#             print(preds[i])
#         print(self.task_num_list)
        return preds

    def add_task_and_update(self, task_num, new_dataloader, test_num_list, test_dataloader_list, writer):

        self.compute_means = True
        
        # Save a copy to compute distillation outputs
        prev_model = copy.deepcopy(self)
        prev_model.cuda()

        self._add_task(task_num)
        self.cuda()
        
        if self.config['optim']=='Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.config['lr'], betas = self.config['betas'], eps=1e-8, weight_decay=self.config['weight_decay'])
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.config['lr'], momentum = self.config['momentum'], weight_decay=self.config['weight_decay'])

        with tqdm(total=self.num_epochs) as pbar:
            for epoch in range(self.num_epochs):

                # Modify learning rate
                # if (epoch+1) in lower_rate_epoch:
                # 	self.lr = self.lr * 1.0/lr_dec_factor
                # 	for param_group in optimizer.param_groups:
                # 		param_group['lr'] = self.lr
                
                for i, sample_batched in enumerate(new_dataloader):        
                    state = Variable(torch.FloatTensor(sample_batched['state'])).cuda()
                    action = Variable(torch.LongTensor(sample_batched['action'])).cuda()

                    optimizer.zero_grad()
                    logits = self.forward(state)
                    
                    cls_loss = nn.CrossEntropyLoss()(logits[:,-self.action_size:], action)
                    logit_loss =0
                    if len(self.task_num_list) >= 2:
                        dist_target = prev_model.forward(state)
                        logits_dist = logits[:,:-self.action_size]
                        logit_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
                        loss = self.lambda_*logit_loss + cls_loss
                    else:
                        loss = cls_loss

                    loss.backward()
                    optimizer.step()

                    if (i+1) % 1 == 0:
                        tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                            %(epoch+1, self.num_epochs, i+1, np.ceil(len(new_dataloader.dataset)/self.batch_size), loss.data))
                
                for task_num_, test_dataloader in zip(test_num_list,test_dataloader_list):
                    test_correct = 0
                    device = torch.device('cuda')
                    for i, sample_batched in enumerate(test_dataloader):
                        state = Variable(torch.FloatTensor(sample_batched['state'])).to(device=device)
                        action = torch.LongTensor(sample_batched['action']).to(device=device)

                        logits = self.forward(state)[:,-10:]
                        action_output = torch.argmax(nn.Softmax()(logits), axis=-1)
    #                     print(action_output)
    #                     print(action)
                        assert(action_output.shape == action.shape)
    #                     print((action_output==action).sum())
                        test_correct += (action_output==action).sum()
                    
                    test_accuracy = float(test_correct)/len(test_dataloader.dataset)
                    writer.add_scalar('order=%d/task%d-test_accuracy' % (len(self.task_num_list),task_num_), test_accuracy, epoch + 1)
                    writer.add_scalar('order=%d/task%d-loss' % (len(self.task_num_list),task_num_), loss, epoch + 1)
                    writer.add_scalar('order=%d/task%d-logit_loss' % (len(self.task_num_list),task_num_), logit_loss, epoch + 1)
                    writer.add_scalar('order=%d/task%d-cls_loss' % (len(self.task_num_list),task_num_), cls_loss, epoch + 1)
                    config['network']['cur_epoch'] = epoch
                    with open(os.path.join(self.config_dir,self.folder_name+".yaml"),"w") as file:
                        yaml.dump(config,file)
        torch.save(self,os.path.join(self.checkpoint_dir,self.folder_name+'.pt'))
        config['network']['task_head'] = self.task_num_list
        with open(os.path.join(self.config_dir,self.folder_name+".yaml"),"w") as file:
            yaml.dump(config,file)
#                 pbar.update(1)
        torch.cuda.empty_cache()


class experiment_world:
    def __init__(self, config):
        #config_experiment
        config_exp = config['experiment']
        self.task_num_list = config_exp['task_num_list']      
        self.num_episodes = config['network']['num_epochs']
        self.batch_size = config['network']['batch_size']
        self.model_time = datetime.datetime.now()
        self.tags = config_exp['tag']
        self.folder_name = 'tasks{}+{}+{}_{}_{}'.format(*config_exp['task_num_list'],
                                                        self.model_time.strftime("%m%d_%H%M%S"),self.tags)
        
        #config env, network and eval
        config_env = config['env']
        self.mov_dist = config_env['mov_dist']
        self.task = config_env['task']
        self.random_spawn = config_env['random_spawn']
        self.block_random = config_env['block_random']
        self.action_size = config_env['action_size']
        self.only_test = config['eval']['only_test']

        config_network = config['network']

        
        self.render = config['view']['render']
        
        config_eval = config['eval']
        self.num_test_ep = config_eval['num_test_ep']
        self.load_data_name = config_eval['load_data_name']
          
        #log_dir
        self.log_dir = os.path.join(FILE_PATH, 'lwf_train_log_torch/')
        self.checkpoint_dir = os.path.join(self.log_dir, 'torch_saved_model/')
        self.config_dir = os.path.join(self.log_dir, 'configs/')
        self.failed_dir = os.path.join(self.log_dir, 'failed/')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.failed_dir):
            os.makedirs(self.failed_dir)
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        with open(os.path.join(self.config_dir,self.folder_name+".yaml"),"w") as file:
            yaml.dump(config,file)
        
        self.env = None
        self.test_dataloader_list = []
        self.max_success_rate = []

        #make lwf network
        self.model = lwf_network(config_network, action_size = self.action_size, 
            config_dir = self.config_dir, checkpoint_dir = self.checkpoint_dir, folder_name = self.folder_name)

        self.task_order = 0

        if self.only_test:
            for i in self.task_num_list:
                self.model._add_task(i)
                self.max_success_rate.append(0.0)
            self.task_order = len(self.task_num_list)
            self.model = torch.load(self.checkpoint_dir+self.load_data_name)
            # self.model = torch.load(self.checkpoint_dir+self.load_data_name)
            # print(self.model)
        

        if args.only_train:
            pass
        else:
            #render options
            screen_width = 192 #264
            screen_height = 192 #64
            crop = 128
            rgbd = True

            #robosuite env
            env = robosuite.make(
            "BaxterPush",
            bin_type='table',
            object_type=config['env']['obj'],
            ignore_done=True,
            has_renderer=config['view']['render'],  # True,
            has_offscreen_renderer=not config['view']['render'],  # added this new line
            camera_name="eye_on_right_wrist",
            gripper_visualization=False,
            use_camera_obs=False,
            use_object_obs=False,
            camera_depth=True,
            num_objects=config['env']['num_blocks'],
            control_freq=100,
            camera_width=screen_width,
            camera_height=screen_height,
            crop=crop
            )
            env = IKWrapper(env)
            self.set_env(env, 0, init=1)
            
        #tensorboard and mkdir
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'tensorboard', self.folder_name))
        
        
    def _add_task_and_test(self, task_num=1):
        train_frac = 0.8
        self.task_order +=1
        self.max_success_rate.append(0.0)
        
        #add dataset and dataloader
        dataset = lwfDataset(None, task_num)
        train_data, test_data = torch.utils.data.random_split(dataset,[round(train_frac*len(dataset)), round((1-train_frac)*len(dataset))])
        self.test_dataloader_list.append(DataLoader(test_data, batch_size=self.batch_size, shuffle=True))
        
        #add fc layer and update
        self.model.add_task_and_update(task_num, DataLoader(train_data, batch_size=self.batch_size, shuffle=True), self.task_num_list[:self.task_order],
                                      self.test_dataloader_list,self.writer)
        
        if not os.path.exists(os.path.join(FILE_PATH,'lwf_train_log_torch/failed',self.folder_name)):
            os.makedirs(os.path.join(FILE_PATH,'lwf_train_log_torch/failed',self.folder_name))
        

    def test_model(self):
        for test_num in self.task_num_list[:self.task_order]:
            success_rate, _ = self.test_agent(self.num_test_ep, test_num)
            print(f'SUCCESS RATE FOR TASK{test_num} : {success_rate}%')
    
    def set_env(self, env, task_num, init=0):
        if init:
            self.test_env_task_num = 0
            self.env = BaxterTestingEnv1(env, task=self.task, render=self.render,
                                using_feature=False,
                                random_spawn=self.random_spawn, block_random = self.block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = self.mov_dist, object_type = "smallcube")
        else:
            self.test_env_task_num = task_num
            if task_num == 0:
                self.env = BaxterTestingEnv1(env.env, task=self.task, render=self.render,
                                using_feature=False,
                                random_spawn=self.random_spawn, block_random = self.block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = self.mov_dist, object_type = "smallcube")
            elif task_num == 1:
                self.env = BaxterTestingEnv1(env.env, task=self.task, render=self.render,
                                using_feature=False,
                                random_spawn=self.random_spawn, block_random = self.block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = self.mov_dist, object_type = "lemon_1")

            elif task_num == 2:
                self.env = BaxterTestingEnv1(env.env, task=self.task, render=self.render,
                                using_feature=False,
                                random_spawn=self.random_spawn, block_random = self.block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1",
                                viewpoint2="rlview2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = self.mov_dist, object_type = "stick")
            elif task_num == 3:
                self.env = BaxterTestingEnv1(env.env, task=self.task, render=self.render,
                                using_feature=False,
                                random_spawn=self.random_spawn, block_random = self.block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1_1",
                                viewpoint2="rlview2_1", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = self.mov_dist, object_type = "smallcube")
            elif task_num == 4:
                self.env = BaxterTestingEnv1(env.env, task=self.task, render=self.render,
                                using_feature=False,
                                random_spawn=self.random_spawn, block_random = self.block_random,
                                rgbd=True, action_type="2D", viewpoint1="rlview1_2",
                                viewpoint2="rlview2_2", viewpoint3 = "None",
                                grasping_env=False, down_level=1,
                                is_test=True, arm_near_block = False, only_above_block = False,
                                down_grasp_combined = True, mov_dist = self.mov_dist, object_type = "smallcube")


    def test_agent(self, num_rep = 10, task_num = 0):
        if self.test_env_task_num != task_num:
            self.set_env(self.env, task_num)

        print("EVAL for TASK  %d (ORDER %d) STARTED"%(task_num, self.task_order))

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
                obs_re = np.concatenate([k for i,k in enumerate(obs)], axis=-1)
                action = self.model.classify(torch.FloatTensor(np.rollaxis(np.expand_dims(obs_re, axis=0),3,1)))
                action = action[self.task_num_list.index(task_num)]
                print("action is : " , action)
                
                action_his.append(action)
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
                im1.save(os.path.join(FILE_PATH, 'lwf_train_log_torch', 'failed/'+self.folder_name+'/test_'
                                      + datetime.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(stucked)+'-view1.png'))
                im2.save(os.path.join(FILE_PATH, 'lwf_train_log_torch', 'failed/'+self.folder_name+'/test_'
                                      + datetime.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(stucked)+'-view2.png'))


            # print("Episode : "+ str(n) + " / Step count : "+ str(step_count) + " / Cum Rew : " + str(cumulative_reward))
            action_dist = np.zeros((self.env.action_size), dtype=int)
            for ind in range(self.env.action_size):
                action_dist[ind] = action_his.count(ind)
            print('dist:',list(action_dist))
        success_rate = np.mean(success_log)
        print('SUCCESS RATE on picking?:', success_rate)
        print('test outcome dist (succ,wrong grasp,not going down, ,max step, ,out-bound,stuck):')
        print(list(failed_case_dist))
        print()
        
        self.writer.add_scalar('order=%d/task%d-success_rate' % (self.task_order, task_num), success_rate, 0)

        # save the model parameters
        if success_rate >= self.max_success_rate[self.task_num_list.index(task_num)]:
            torch.save(self.model,self.checkpoint_dir,self.folder_name+'.pt')
            self.max_success_rate[self.task_num_list.index(task_num)] = success_rate
            file = open(os.path.join(self.config_dir,self.folder_name+".yaml"),"w")
            yaml.dump(config,file)
            file.close()

        return success_rate, failed_case_dist
    
    def schedule_lr(self, epoch, round, num_rounds, decay_rate = 0.1):
        initial_lrate = self.lr
        k = decay_rate
        lrate = initial_lrate * np.exp(-k*(epoch+float(round)/num_rounds))
        # print("learning rate for this round is : ", lrate)
        return lrate

def main():
    #'data/processed_data'
    experiment_ = experiment_world(config)
#     gpu_config = tf.ConfigProto()
#     gpu_config.gpu_options.allow_growth = True
    # with tf.Session(config=gpu_config) as sess:
    if config['eval']['only_test']:
        experiment_.test_model()
    else:
        for num in config['experiment']['task_num_list']:
            experiment_._add_task_and_test(num)
    


if __name__=='__main__':
    main()
