import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4 *x +6  #2차함수 

gradient = lambda x: 2*x -4

x0 = 0.0
MaxIter = 20
learning_rate = 0.25

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

#  {:02d} => N자리로 포맷팅할 때,

# 정수부 : 0Nd

# 실수부 : 0.Nf
# {:6.5f} = > 6자리를 채우는데 소수점 5자리까지 표시한다.
#따로 표시가 없으면 순서대로 가져오는듯 하다.
#  포메팅링크 https://datascienceschool.net/01%20python/02.04%20%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%98%20%EB%AC%B8%EC%9E%90%EC%97%B4%20%ED%98%95%EC%8B%9D%ED%99%94.html

for i in range(MaxIter):# 여기는 그냥 돌아가는 횟수를 지정해줄 뿐
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1 #이전의 for문의 x0값이계속 없데이트 되어 맞추어 가는 것!
    print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))
'''

이식은 2차방정식에서 learning rate에 기울기를 곱해서 
값과 기울기를 구하는 것이다!  즉 기울기가 0이 되는 지점을 찾아가는 것!!

'''
