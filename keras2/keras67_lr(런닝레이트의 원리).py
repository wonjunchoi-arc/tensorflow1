weight = 0.5
input = 0.5
goal_prediction = 0.8
lr =0.005 # 0.001 # 0.1/ 1 / 0.001 / 100
epochs = 3000

for iteration in range(epochs):
    prediction = input * weight
    error = (prediction - goal_prediction) **2
    #입력값을 통해 예측값과 에러값(목표로하는 값에서 얼마만큼 먼지)을 측정

    print("Error : " + str(error)+ "\tPrediction:"+str(prediction))

    up_prediction = input * (weight +lr)
    up_error = (goal_prediction - up_prediction)**2
#가중치를 높였을때 예측값과 에러값을 측정

    down_prediction = input * (weight -lr)
    down_error = (goal_prediction - down_prediction)**2
#가중치를 낮췄을 때 예측값과 에러값을 뽑아낸다.
    if(down_error < up_error):
        weight = weight -lr
    if(down_error > up_error):
        weight = weight +lr
#낮췄을 때의 에러가 더 크다면 가중치를 조금씩 계속해서 더해가는 것이다!!