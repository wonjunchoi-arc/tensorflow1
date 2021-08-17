
from cv2 import ml_KNearest
import cv2 
import mediapipe as mp
import numpy as np
import csv
import pandas as pd


max_num_human = 1
discs_pose = {
0:'correct',1:'discs',2:'neck_discs',

}


#Media
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
     model_complexity=2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
)

#Gesture R model
file =np.genfromtxt('../down/discs_train.csv',delimiter=',')
angle =file[:,:-1].astype(np.float32)
label =file[:,-1].astype(np.float32)

knn = ml_KNearest.create()
knn.train(angle, cv2.ml.ROW_SAMPLE,label)

cap = cv2.imread('../down/nor/10.jpg')

if cap is not None:
    cap= cv2.flip(cap, 1)
    cap = cv2.cvtColor(cap,cv2.COLOR_BGR2RGB)

    results =pose.process(cap)

    cap =cv2.cvtColor(cap, cv2.COLOR_RGB2BGR)
    cv2.imshow(cap)

else:
    print('No image file.')

print(type(results.pose_landmarks))
a=[]
a.append(results.pose_landmarks)
print(len(a))

if results.pose_landmarks is not None:
     for res in a:
         joint =np.zeros((33,4))
         for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

v1 = joint[[11,11,12,12,11,12,23],:]
v2 =joint[[0,7,0,7,23,24,24],:]
v = v2 - v1
v= v/ np.linalg.norm(v, axis=1)[:,np.newaxis]


angle =np.arccos(np.einsum('nt,nt->n',
v[[0,1,2,3,4,5],:],
v[[1,2,3,4,5,6],:]))

angle =np.degrees(angle)

print(angle)
# print(type(angle))

# df = pd.DataFrame(angle)
# df.loc[6] = [1]
# df=df.transpose()
# print(df)

# df.to_csv('../down/discs_train.csv',mode='a', index=False,header = False)

data =np.array([angle], dtype=np.float32)
print(data)
ret, results, neighbors, dist = knn.findNearest(data, 1)
idx =int(results[0][0])
print(ret)
print(results)
print(neighbors)
print(dist)


#Draw
if idx in discs_pose.keys():
    cv2.putText(cap, text=discs_pose[idx].upper(),
                org=(int(res.landmark[25].x * cap.shape[1])
    , int(res.landmark[0].y * cap.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.5, color=(255,100,120), thickness=1)

print('여기까지 실행')

mp_drawing.draw_landmarks(cap, res, mp_pose.POSE_CONNECTIONS)
cv2.imshow('img',cap)
cv2.waitKey(0)  ##
cv2.destroyAllWindows() ## 이코드 안넣어주면 사진이 알아서 꺼짐