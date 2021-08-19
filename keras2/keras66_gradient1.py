import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4 *x +6
x = np.linspace(-1,6,100)
#기울기 배열을 만듦  -1~6 까지 100개의 숫자
print(x)
y = f(x)

#그리기
plt.plot(x,y,'k-')
plt.plot(2,2,'sk')
plt.grid() #눈금 주는거
plt.xlabel('x')
plt.xlabel('x')
plt.show()
