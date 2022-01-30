import numpy as np
from bc_real_multitask import my_selector, my_network, my_module2

# n_tasks: total number of tasks
# t_idx: task index
# if the order of training is like below,
# reach -> push -> pick
#   0       1       2     <- t_idx
selector = my_selector([128, 128, 8], n_tasks)
task_network = my_network([128, 128, 8], [1, 1, 1, 6])
model = my_module2([128, 128, 8], selector, task_network)
model.load_weights(model_name)

if task=='reach' or task=='push':
    task_labels = range(t_idx * 12, t_idx * 12 + 8)
elif task=='pick':
    task_labels = range(t_idx * 12, t_idx * 12 + 12)
elif task=='place':
    task_labels = range(t_idx * 12, t_idx * 12 + 10)

set_active_selection(t_idx)
set_active_outputs(task_labels)

def get_action(obs):
    clipped_obs = np.clip(obs, 0.0, 5.0)
    obs0 = clipped_obs[0, :, :, :]
    obs1 = clipped_obs[1, :, :, :]
    obs = np.concatenate([obs0, obs1], axis=-1)
    obs = np.expand_dims(obs, axis=0)
    obs = np.tile(obs, [64,1,1,1])
    action_prob = model.predict(obs, 64)[1][0]
    action = np.argmax(action_prob) % 12
    return action
