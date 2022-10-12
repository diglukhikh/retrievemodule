import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, concatenate, Input
import numpy
model_loaded = keras.models.load_model("C:\\Users\Хозяин\PycharmProjects\pythonProject01\\admin\education\\133")

# адрес валидационного
dataset_valid = numpy.loadtxt("/Users/Public/files/valid20_7.csv", delimiter=",")
s = dataset_valid.shape[1]
print(s)
a = int((s - 1) / 2)
X1 = dataset_valid[:, 0:a].astype(int)
X2 = dataset_valid[:, a:a * 2].astype(int)
X = dataset_valid[:, 0:a * 2].astype(int)
print(X1.shape)
print(X2.shape)
Y = dataset_valid[:, a * 2]

X_in_valid1 = abs(X1 - X2)
X_in_valid2 = abs(X1 + X2)
X_in_valid3 = X

predictions = model_loaded.predict([X_in_valid1,X_in_valid2,X_in_valid1,X_in_valid1])

print(Y)  # это фактические значения из файла

aa = numpy.around(predictions, decimals=3)
print(aa)

scores_valid = model_loaded.evaluate([X_in_valid1,X_in_valid2,X_in_valid1,X_in_valid1], Y)
print("\n%s: %.2f%%" % (model_loaded.metrics_names[1], scores_valid[1]))

