
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='Malgun Gothic')

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

from glob import glob

import tensorflow as tf
from tensorflow.python.client import device_lib
import gc


predictions = []
for ar in glob('../data/dacon22/*.npy'):
    arr = np.load(ar)
    predictions.append(arr)

sample = pd.read_csv('../data/dacon2/open/sample_submission.csv')
sample['label'] = np.argmax(np.mean(predictions,axis=0), axis = 1)
sample.to_csv('../data/dacon22/submission.csv',index=False)


# with open('inputs5.pkl','rb') as f :
    # train_inputs, test_inputs, labels = pickle.load(f)