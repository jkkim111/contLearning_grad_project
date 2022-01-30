import os
import numpy as np
import psutil

DATA_PATH = 'data/processed_data'
SAVE_PATH = 'data/small_data'
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

task = 'reach'
batch_size = 512

pkl_list = sorted([f for f in os.listdir(DATA_PATH) if '.pkl' in f])

a_list = sorted([p for p in pkl_list if task+'_a_' in p])
s_list = sorted([p for p in pkl_list if task+'_s_' in p])
assert len(a_list) == len(s_list)

npy_count = 0
buff_a = []
buff_s = []
for i in range(len(a_list)):
    # print(npy_count)
    a_name = a_list[i]
    s_name = s_list[i]

    if len(buff_a) > batch_size:
        pkl_a = np.load(os.path.join(DATA_PATH, a_name), allow_pickle=True)
        pkl_s = np.load(os.path.join(DATA_PATH, s_name), allow_pickle=True)
        os.remove(os.path.join(DATA_PATH, a_name))
        os.remove(os.path.join(DATA_PATH, s_name))
        if len(buff_a)==0:
            buff_a = pkl_a#.copy()
            buff_s = pkl_s#.copy()
        else:
            buff_a = np.concatenate([buff_a, pkl_a])#.copy()
            buff_s = np.concatenate([buff_s, pkl_s])#.copy()
    # del pkl_a, pkl_s
    # buff_a.append(pkl_a)
    # buff_s.append(pkl_s)

    while len(buff_a) < batch_size:
        # batch_a = np.concatenate(buff_a)
        # batch_s = np.concatenate(buff_s)
        batch_a = buff_a[:batch_size].copy()
        batch_s = buff_s[:batch_size].copy()
        np.save(os.path.join(SAVE_PATH, task+'_a_'+str(npy_count)), batch_a)
        np.save(os.path.join(SAVE_PATH, task+'_s_'+str(npy_count)), batch_s)
        # del batch_a, batch_s
        npy_count += 1

        process = psutil.Process(os.getpid())
        print(i, npy_count, process.memory_percent())

        buff_a = buff_a[batch_size:].copy()
        buff_s = buff_s[batch_size:].copy()

if False and len(buff_a) != 0:
    batch_a = np.concatenate(buff_a)
    batch_s = np.concatenate(buff_s)
    np.save(os.path.join(SAVE_PATH, task+'_a_'+str(npy_count)), batch_a)
    np.save(os.path.join(SAVE_PATH, task+'_s_'+str(npy_count)), batch_s)
