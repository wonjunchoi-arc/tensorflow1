
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='Malgun Gothic')

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

from glob import glob

import tensorflow as tf
from tensorflow.python.client import device_lib
import gc

device_lib.list_local_devices()
tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)

# 텐서 생성
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)



# 주로 사용하는 코드 2 : 인식한 GPU 개수 출력
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# 방법 2-2
tf.config.list_physical_devices('GPU')

# 방법 2-3
tf.config.experimental.list_physical_devices('GPU')

# 방법 2-4
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

def get_input_dataset(data, index, train = False) : 
    input0 = tf.convert_to_tensor(data[0][index].toarray(), tf.float32)
    input1 = tf.convert_to_tensor(data[1][index].toarray(), tf.float32)
    input2 = tf.convert_to_tensor(data[2][index].toarray(), tf.float32)
    
    if train : 
        label = labels[index]

        return input0, input1, input2, label
    else:
        return input0, input1, input2

def plot_loss(history):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(history.history['acc'], 'b', label='train accuracy')
    acc_ax.plot(history.history['val_acc'], 'g', label='val accuracy')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.show()

def single_dense(x, units):
    fc = Dense(units, activation = None, kernel_initializer = 'he_normal')(x)
    batch = BatchNormalization()(fc)
    relu = ReLU()(batch)
    dr = Dropout(0.25)(relu)
    
    return dr

def create_model(input_shape0,input_shape1,input_shape2, num_labels, learning_rate):
    x_in0 = Input(input_shape0,)
    x_in1 = Input(input_shape1,)
    x_in2 = Input(input_shape2,)
    
    fc0 = single_dense(x_in0, 512)
    fc0 = single_dense(fc0, 256)
    fc0 = single_dense(fc0, 128)
    fc0 = single_dense(fc0, 64)
    
    fc1 = single_dense(x_in1, 1024)
    fc1 = single_dense(fc1, 512)
    fc1 = single_dense(fc1, 256)
    fc1 = single_dense(fc1, 128)
    fc1 = single_dense(fc1, 64)
    
    fc2 = single_dense(x_in2, 512)
    fc2 = single_dense(fc2, 256)
    fc2 = single_dense(fc2, 128)
    fc2 = single_dense(fc2, 64)
    
    fc = Concatenate()([fc0,fc1,fc2])
    
    fc = single_dense(fc, 128)
    fc = single_dense(fc, 64)
    
    x_out = Dense(num_labels, activation = 'softmax')(fc)
    
    model = Model([x_in0,x_in1,x_in2], x_out)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    
    return model

with open('inputs.pkl','rb') as f :
    train_inputs, test_inputs, labels = pickle.load(f)

num_labels = 46
learning_rate = 2e-2
seed = np.random.randint(2**16-1)
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed)




for train_idx, valid_idx in skf.split(train_inputs[0], labels):
    print('!!!!!!|||||||||||||||||||||||||||||~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~여기')
    X_train_input0, X_train_input1, X_train_input2, X_train_label = get_input_dataset(train_inputs, train_idx, train = True)
    X_valid_input0, X_valid_input1, X_valid_input2, X_valid_label = get_input_dataset(train_inputs, valid_idx, train = True)
    
    # X_valid_input0 = np.array(X_valid_input0)
    # X_valid_input1 = np.array(X_valid_input1)

    # X_valid_input2 = np.array(X_valid_input2)
    # # X_valid_input0 = np.array(X_valid_input0)


    now = datetime.now()
    now = str(now)[11:16].replace(':','h')+'m'
    ckpt_path = f'./{now}.ckpt'
    
    input_shape0 = X_train_input0.shape[1]
    input_shape1 = X_train_input1.shape[1]
    input_shape2 = X_train_input2.shape[1]


    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor = 'val_acc', save_best_only= True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.9, patience = 2,),
                ]
    model = create_model(input_shape0,input_shape1,input_shape2, num_labels, learning_rate)
    model.fit(
                        [X_train_input0,X_train_input1,X_train_input2],
                        X_train_label,
                        epochs=200,
                        callbacks=callbacks,
                        validation_data=([X_valid_input0, X_valid_input1, X_valid_input2], X_valid_label),
                        verbose=1,  # Logs once per epoch.
                        batch_size=2048)
    
    model.load_weights(ckpt_path)
    prediction = model.predict([test_inputs[0], test_inputs[1], test_inputs[2]])
    np.save(f'../data/dacon2{now}_prediction.npy', prediction)
    del X_train_input0
    del X_train_input1
    del X_train_input2
    del prediction
    del X_train_label
    del X_valid_input0
    del X_valid_input1
    del X_valid_input2
    del X_valid_label
    gc.collect()

predictions = []
for ar in glob('../data/dacon2/*.npy'):
    arr = np.load(ar)
    predictions.append(arr)

sample = pd.read_csv('../data/sample_submission.csv')
sample['label'] = np.argmax(np.mean(predictions,axis=0), axis = 1)
sample.to_csv('../data/Day0810_JayHong.csv',index=False)


# with open('inputs5.pkl','rb') as f :
#     train_inputs, test_inputs, labels = pickle.load(f)