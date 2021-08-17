import cv2
import uuid
import os
import time
from icecream import ic
from os import listdir
from git import Repo
import git
import subprocess



labels =['discs','neck_discs']
number_imgs = 5

IMAGES_PATH = os.path.join('../data','disc','collect')
ic(os.name)

if not os.path.exists(IMAGES_PATH):
    if os.name =='posix':
        os.makedirs(IMAGES_PATH,exist_ok=True)
    if os.name =='nt':
        os.mkdir(IMAGES_PATH)
for label in labels :
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)


LABEL_PATH = os.path.join('../data','disc','label')
if not os.path.exists(LABEL_PATH):
    os.mkdir(LABEL_PATH)
    



LABELING_PATH = os.path.join('..\data', '\labeling')
if not os.path.exists(LABELING_PATH):
    os.mkdir(LABELING_PATH)
if os.name == 'posix':
    pass
if os.name =='nt':
    os.chdir(LABELING_PATH) and os.system('pyrcc5 -o libs/resources.py resources.qrc')
os.system('"d:\\data\\labeling\\python labelImg.py"')




