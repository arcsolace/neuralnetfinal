import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import *
import os
import sys
#gets rid of silly error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#maximizes print output
np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('https://raw.githubusercontent.com/arcsolace/neuralnetfinal/master/leafshape_edit.csv', index_col=0)

df2 = pd.read_csv('https://raw.githubusercontent.com/arcsolace/neuralnetfinal/master/leafshape_scale.csv', index_col=0)


#editing main dataset
train_label = pd.DataFrame(df[["location"]].copy(deep=False))
train_input = df.drop(columns=["location", "arch", "latitude"])
train_label = to_categorical(train_label)
train_label = np.delete(train_label, np.s_[0], axis=1)
del df

#editing normalised dataset
train_label2 = pd.DataFrame(df2[["location"]].copy(deep=False))
train_input2 = df2.drop(columns=["location", "arch", "latitude"])
train_label2 = to_categorical(train_label2)
train_label2 = np.delete(train_label2, np.s_[0], axis=1)
del df2

pd.options.display.max_columns = train_input.shape[1]
print(train_input.describe(include = 'all'))


#normalizing datameans
train_means = train_input.mean(axis=0)
train_stds = train_input.std(axis=0)
print("Means:")
print(train_means.head(5))
print("Stds:")
print(train_stds.head(5))
train_input = train_input - train_means
train_input = train_input / train_stds

#doing it for this one too just in case
train_means2 = train_input2.mean(axis=0)
train_stds2 = train_input2.std(axis=0)
train_input2 = train_input2 - train_means2
train_input2 = train_input2 / train_stds2

#splitting dataset
(traindata, testdata, trainlocs, testlocs) = train_test_split(
	train_input, train_label, test_size=0.05, random_state=42)
(traindata2, testdata2, trainlocs2, testlocs2) = train_test_split(
	train_input2, train_label2, test_size=0.05, random_state=42)

#model specs
model = Sequential()
model.add(Dense(12, input_dim=traindata.shape[1], activation="relu",
                kernel_initializer="random_uniform",
                bias_initializer="zeros"))
model.add(Dense(6, activation="relu", kernel_initializer="random_uniform",
                bias_initializer="zeros"))
model.add(Dense(3, activation="relu", kernel_initializer="random_uniform",
                bias_initializer="zeros"))
model.add(Dense(6, activation="softmax"))

#compile model
print("compiling model...")
model.compile(loss="categorical_crossentropy", optimizer="adam",
	metrics=["accuracy"])
model.fit(traindata.values,trainlocs, epochs=50, batch_size=1)

#evaluate model
print("evaluating on testing set...")
test_loss_and_metrics = model.evaluate(testdata.values, testlocs)
train_loss_and_metrics = model.evaluate(traindata.values, trainlocs)
print("")
print("test accuracy: " + str(test_loss_and_metrics[1]))
print("train accuracy: " + str(train_loss_and_metrics[1]))

print("evaluating on scaled testing set...")
test_loss_and_metrics2 = model.evaluate(testdata2.values, testlocs2)
train_loss_and_metrics2 = model.evaluate(traindata2.values, trainlocs2)
print("")
print("test accuracy: " + str(test_loss_and_metrics2[1]))
print("train accuracy: " + str(train_loss_and_metrics2[1]))

#creating array of real values from dataset, expected output = "Sabah"
#or [0 0 0 0 1 0]
X = np.array([[0.594040778, -0.239388951, 0.03130408, 0.355633858,
            -0.0867434952, 0.703759068]]) #sabah
Y = np.array([[-0.350023280, 0.123138731, -0.196972400, 0.045077371,
            0.9635150995, -0.108321847]]) #n queensland
Z = np.array([[-0.519582873, -0.206006784, -0.771287042, -1.215173889,
            0.0736911444, -0.305097361]]) #tasmania

predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

predictions = model.predict(Y)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

predictions = model.predict(Z)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

#did not successfully predict...
