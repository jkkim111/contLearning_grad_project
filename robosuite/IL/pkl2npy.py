import os
import numpy as np

DATA_PATH = 'data/processed_data'
SAVE_PATH = 'data/npy_data'
pkl_list = sorted([f for f in os.listdir(DATA_PATH) if '.pkl' in f])

for pkl_name in pkl_list:
    x = np.load(os.path.join(DATA_PATH, pkl_name), allow_pickle=True)
    npy_name = pkl_name.split('.')[0]
    np.save(os.path.join(SAVE_PATH, npy_name), x)
