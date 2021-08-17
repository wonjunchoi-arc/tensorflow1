
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


train=pd.read_csv('../data/dacon2/open/train.csv')

train=pd.read_csv('../data/dacon2/train_okt.csv')
test=pd.read_csv('../data/dacon2/test_okt.csv')
sample_submission=pd.read_csv('../data/dacon2/open/sample_submission.csv')

