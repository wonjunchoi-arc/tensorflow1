import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

img_col = cv.imread('../data/disc/nor/0.jpg', cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img_col,cv.COLOR_BAYER_BG2RGB)

ret,img_binary = cv.threshold(img_gray,0,255, cv.TH | cv.THRESH_OTSU)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5) )
img_bi = cv.morphologyEx(img_binary,cv. MORPH_CLOSE,kernel)

cv.imshow('digit', img_bi)
cv.waitKey(0)
import PyQt5 