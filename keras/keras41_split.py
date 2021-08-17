import numpy as np

a = np.array(range(1,11))
size = 5

print(a.shape)
print(len(a))

def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset) - size +1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)


print(dataset.shape)

x = dataset[: , :4]

y = dataset[:,4]

print('x : ',x)
print('y : ',y)

##시계열 데이터를 만드는 함수다 !!!


