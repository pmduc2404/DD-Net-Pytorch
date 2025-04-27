#! /usr/bin/env python
#! coding:utf-8:w

from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
import sys
sys.path.insert(0, '..')
from utils import *  # noqa
current_file_dirpath = Path(__file__).parent.absolute()


def load_KIMORE_data(ex = 1):
        
    train_path=current_file_dirpath / Path(f"../KIMORE_pickle/ex{str(ex)}/exercise_{str(ex)}_train.pkl")
    test_path=current_file_dirpath / Path(f"../KIMORE_pickle/ex{str(ex)}/exercise_{str(ex)}_test.pkl")
    
    Train = pickle.load(open(train_path, "rb"))
    Test = pickle.load(open(test_path, "rb"))
    
    print("Loading KIMORE Dataset")
    return Train, Test


class KIMOREConfig():
    def __init__(self):
        self.frame_l = 50  # the length of frames
        self.joint_n = 25  # the number of joints
        self.joint_d = 7  # the dimension of joints
        # self.clc_num = 21  
        self.feat_d = 300
        self.filters = 64

# Genrate dataset
# T: Dataset  C:config le:labelEncoder


def KIMOREdata_generator(T, C):
    X_0 = []
    X_1 = []
    Y = []
    # labels = le.transform(T['label'])
    for i in tqdm(range(len(T['pose']))):
        p = np.copy(T['pose'][i])
        # p.shape (frame,joint_num,joint_coords_dims)
        p = zoom(p, target_l=C.frame_l,
                 joints_num=C.joint_n, joints_dim=C.joint_d)
        # p.shape (target_frame,joint_num,joint_coords_dims)
        # label = np.zeros(C.clc_num)
        # label[labels[i]] = 1
        label = np.copy(T['labels'][i])
        # M.shape (target_frame,(joint_num - 1) * joint_num / 2)
        M = get_CG(p, C)

        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)
    return X_0, X_1, Y
