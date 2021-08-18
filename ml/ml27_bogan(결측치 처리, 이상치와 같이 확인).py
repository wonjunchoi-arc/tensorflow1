# [1, np.nan ,np.nan, 8 ,10]

# 결측치 처리
# 1. 행 삭제
# 2. 0 넣기 => [1, 0, 0, 8, 10] => but 특정상황에서는 값의 차이가 커질 수도 있다.
# 3. 앞자리 숫자 넣기  [1, 1, 1, 8, 10]
# 4. 뒷자리 숫자 넣기  [1, 8, 8, 8, 10] 
# 5. 중위값            [1 ,4.5 ,4.5 ,8 ,10]
# 사람들이 생각하는 대로 저 결측치 자리에 진행되가는 숫자의 값들을 넣는다면???
#                      [1, , , 8, 10] 
# 6. bogan
# 7. 모델형 predict (모델을 돌려서 나온 예측값을 결측치의 값에 넣는다)
# 8. 부스트계열은 결측치에 대해 자유(?)롭다! tree 계열 ... DT,RF,XG, LGBM

'''
bogan은 좌표에 그어진 선을 기준으로 데이터를 예측해서 찍음

'''


from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datastrs = ['8/13/2021','8/14/2021','8/15/2021','8/16/2021','8/17/2021'] 
dates = pd.to_datetime(datastrs)
print(dates)
print(type(dates))#<class 'pandas.core.indexes.datetimes.DatetimeIndex'>   

ts = Series([1, np.nan ,np.nan, 8 ,10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)