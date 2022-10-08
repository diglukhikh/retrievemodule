from keras.models import Sequential
#adress = "C:\\Users\Хозяин\PycharmProjects\pythonProject01\\admin\education\\80_sit.csv"
import numpy
from tensorflow import keras
import numpy
import pandas as pd
import time

def retrtrain(adress):

    # для нечеткого случая
    dataset = numpy.loadtxt(adress, delimiter=",")
    s = dataset.shape[1]
    print(s)
    a = int((s-1)/2)

    # смотрим, что из бея представляет датасет
    # print(dataset)
    #X = dataset[:, 0:224]
    #print(X)
    X1 = dataset[:, 0:a].astype(int)
    X2 = dataset[:, a:a*2].astype(int)
    print(X1.shape)
    print(X2.shape)
    Y = dataset[:, a*2]
    #print(Y)

    # формируем входной вектор - разница между соответствующими позициями векторов X1, X2
    #X_in1 = 1-abs(X1-X2)
    X_in1 = abs(X1 - X2)
    X_in2 = abs(X1 + X2)
    # Это  нейросеть  для прогноза Симилирити с числом входов - 112 =  абсолютная разница между элементами двух векторов X1, X2 двух ситуаций
    from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
    from keras import losses
    # create model
    model = Sequential()
    model.add(Dense(a, input_dim=a, activation='relu'))

    model.add(Dense(a/2, activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(a/4, activation='relu'))

    model.add(Dense(a/8, activation='swish'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    model.compile(loss='MeanSquaredError', optimizer='RMSprop', metrics=['mape'])
    # Fit the model
    #history = model.fit(X_in, Y, epochs=67, validation_split=0.3, batch_size=10) #нечеткая логика
    history1 = model.fit(X_in1, Y, epochs=35, validation_split=0.5, batch_size=10) #четкая логика
    # evaluate the model
    scores = model.evaluate(X_in1, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))
    model.save('C:/Users/Хозяин/PycharmProjects/pythonProject01/admin/education/13')




retrtrain("C:\\Users\Хозяин\PycharmProjects\pythonProject01\\admin\education\\train.csv")

def apply():


    model_loaded = keras.models.load_model("C:\\Users\Хозяин\PycharmProjects\pythonProject01\\admin\education\\13")

    # адрес валидационного
    dataset_valid = numpy.loadtxt("/Users/Public/files/valid20_7.csv", delimiter=",")
    s = dataset_valid.shape[1]
    print(s)
    a = int((s - 1) / 2)
    X1 = dataset_valid[:, 0:a].astype(int)
    X2 = dataset_valid[:, a:a * 2].astype(int)
    print(X1.shape)
    print(X2.shape)
    Y = dataset_valid[:, a * 2]
    #X_in_valid1 = 1 - abs(X1 - X2)
    X_in_valid1 = abs(X1 - X2)
    # X_in_valid = 1-abs(X1_valid - X2_valid)

    # print(X_in_valid)

    # calculate predictions1

    predictions = model_loaded.predict(X_in_valid1)
    # predictions = preprocessing.normalize(predictions1)
    # print(predictions1)
    # print(predictions.T)

    pred1 = model_loaded.predict(X_in_valid1)
    # print(pred1.T)  # это предсказание НС
    print(Y)  # это фактические значения из файла
    # print('Округленные предсказания "похож-непохож" с порогом 0,85, как в дата сете')
    # print(    pred1.T)  # - печать в транспонированном виде - это то, что спрогнозировал на выходе сумматор , т..е оценки Похоже1, непохоже0 с преодолением порога
    # print('predictions')
    aa = numpy.around(predictions, decimals=3)
    print(aa)
    tn = time.strftime("%Y%m%d-%H%M%S")
    #a = pd.DataFrame(numpy.around(predictions.T, decimals=3))
    #print(a)
    #a.to_excel('/Users/Public/files/' + tn + '.xlsx')
    scores_valid = model_loaded.evaluate(X_in_valid1, Y)
    print("\n%s: %.2f%%" % (model_loaded.metrics_names[1], scores_valid[1]))



apply()