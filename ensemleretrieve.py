#ансамблиевая модель для определения схожести векторов. Ансамблирование позволило снизить значение mape до 4-7%, добиться 100% точности ранжирования на тестовой выборке

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, concatenate, Input
from kbs import Elements
from kbs import db
import numpy
import json
import os
from kbs import UPLOAD_FOLDER

# для нечеткого случая
dataset = numpy.loadtxt("train.csv", delimiter=",")
s = dataset.shape[1]
print(s)
a = int((s - 1) / 2)
print(f"размер {a}")
X1 = dataset[:, 0:a].astype(int)
X = dataset[:, 0:a*2].astype(int)
X2 = dataset[:, a:a * 2].astype(int)
print(X1.shape)
print(X2.shape)
Y = dataset[:, a * 2]
print(Y)

# формируем входной вектор - разница между соответствующими позициями векторов X1, X2
# X_in1 = 1-abs(X1-X2)
X_in1 = abs(X1 - X2)

X_in2 = abs(X1 + X2)
X_in3 = X


model_in1 = Input(shape=(a,))
model_out1111 = Dense(512, input_dim=a, activation="relu", name="layer1111")(model_in1)
model_out111 = Dense(256, activation="relu", name="layer111")(model_out1111)
model_out11 = Dense(128, activation="swish", name="layer11")(model_out111)
model_out1 = Dense(56, activation="relu", name="layer1")(model_out11)
# model1 = Model(model_in1, model_out1)

model_in2 = Input(shape=(a,))
model_out22 = Dense(512, input_dim=a, activation="relu", name="layer2")(model_in2)
model_out222 = Dense(256, activation="swish", name="layer22")(model_out22)
model_out2222 = Dense(128, activation="relu", name="layer222")(model_out222)
model_out2 = Dense(56, activation="swish", name="layer2222")(model_out2222)
# model2 = Model(model_in2, model_out2)

model_in3 = Input(shape=(a,))
model_out33 = Dense(128, input_dim=a, activation="relu", name="layer3")(model_in3)
model_out3 = Dense(56, activation="relu", name="layer333")(model_out33)
# model1 = Model(model_in1, model_out1)

model_in4 = Input(shape=(a,))
model_out44 = Dense(256, input_dim=a, activation="relu", name="layer4")(model_in4)
model_out444 = Dense(128, activation="swish", name="layer444")(model_out44)
model_out4 = Dense(56, activation="relu", name="layer44")(model_out444)

concatenated = concatenate([model_out1, model_out2, model_out3,model_out4])
out = Dense(1, activation="relu", name="outputlayer")(concatenated)
merged_model = Model([model_in1, model_in2, model_in3,model_in4], out)
merged_model.compile(loss='MeanSquaredError', optimizer='RMSprop', metrics=['mape'])
merged_model.fit([X_in1, X_in2, X_in1,X_in1], Y, batch_size=15, epochs=15, verbose=1, validation_split=0.3)
merged_model.save('adress')




