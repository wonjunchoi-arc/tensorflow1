from cv2 import ml_KNearest # import로 ml_KNearset 해줘야한다.
import cv2 
import mediapipe as mp
import numpy as np


max_num_hands = 1
gesture = {
0:'fist',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',
            7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',

}

rps_gesture = {0: 'rock', 5:'paper', 9:'scissors'}

#Media
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands =max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
)

from sklearn.neighbors import KNeighborsClassifier
#Gesture R model
file =np.genfromtxt('../data/disc/gesture_train.csv',delimiter=',')
angle =file[:,:-1].astype(np.float32)
label =file[:,-1].astype(np.float32)
knn = ml_KNearest.create()  ## cv2에서 knn모델 쓸려면 cv.ml(머신러닝).Knearest 대신 내가 쓴 코드 써야함
# knn = KNeighborsClassifier(n_neighbors=2)
# knn.fit(angle,label)
knn.train(angle, cv2.ml.ROW_SAMPLE,label)


cap = cv2.imread('../data/disc/nor/0.png')

if cap is not None:
    cap= cv2.flip(cap, 1)
    cap = cv2.cvtColor(cap,cv2.COLOR_BGR2RGB)

    results =hands.process(cap)

    cap =cv2.cvtColor(cap, cv2.COLOR_RGB2BGR)
    cv2.imshow('img',cap)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('No image file.')


if results.multi_hand_landmarks is not None:
    for res in results.multi_hand_landmarks:
        joint =np.zeros((21,3))
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z]

v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
v2 =joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
v = v2 - v1
v= v/ np.linalg.norm(v, axis=1)[:,np.newaxis]


angle =np.arccos(np.einsum('nt,nt->n',
v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
v[[1,2,3,5,6,7,9,10,11,13,14,15,7,18,19],:]))


angle =np.degrees(angle)


data =np.array([angle], dtype=np.float32)
ret, results, neighbors, dist = knn.findNearest(data, 3)
idx =int(results[0][0])

#Draw
if idx in rps_gesture.keys():
    cv2.putText(cap, text=rps_gesture[idx].upper(),
                org=(int(res.landmark[0].x * cap.shape[1])
    , int(res.landmark[0].y * cap.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1, color=(255,255,255), thickness=2)

print('여기까지 실행')

mp_drawing.draw_landmarks(cap, res, mp_hands.HAND_CONNECTIONS)
cv2.imshow('img',cap)
cv2.waitKey(0)  ##
cv2.destroyAllWindows() ## 이코드 안넣어주면 사진이 알아서 꺼짐