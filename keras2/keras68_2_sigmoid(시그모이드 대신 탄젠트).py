import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
#-5부터 5까지 0.1 단위로 숫자 생성

y = np.tanh(x)

plt.plot(x,y)
plt.grid()
plt.show()

'''
 tanh와 Sigmoid의 차이점은 Sigmoid의 출력 범위가 0에서 1 사이인 반면 
 tanh와 출력 범위는 -1에서 1사이라는 점입니다. 
 Sigmoid와 비교하여 tanh와는 출력 범위가 더 넓고 경사면이 큰 범위가 더 크기 때문에 
 더 빠르게 수렴하여 학습하는 특성이 있습니다.

 Sigmoid와 비교하여 중심점이 0이고 범위가 기울기 넓은 차이점이 있지만 Sigmoid의 
 치명적인 단점인 Vanishing gradient problem 문제를 그대로 갖고 있습니다.
'''