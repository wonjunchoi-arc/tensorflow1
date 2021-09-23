import autokeras as ak

from tensorflow.keras.datasets import mnist

#1. 데이터

(x_train,y_train),(x_test,y_test)=mnist.load_data()

#2. 모델
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2 #이 모델을 두번 돌리겠다
)

#3. 컴파일
model.fit(x_train,y_train, epochs=5)

#4. 평가 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test,y_test)
print(results)