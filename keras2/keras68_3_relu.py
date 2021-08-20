import numpy as np
import matplotlib.pyplot as plt

def relu(x) : 
    return np.maximum(0,x)
# 0이하는 전부 0으로 그외의 값은 양수로!

x = np.arange(-5,5,0.1)
y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()

# 과제 
'''
relu의 파생중에 어느정도 음수값을 인정해주는 
elu, selu, reaky relu ... 등의 친구들을 활용하요
68_3_2,3,4 .... 로 만들것!!
'''
