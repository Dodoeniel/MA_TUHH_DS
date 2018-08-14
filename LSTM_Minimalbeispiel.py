
import csv
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load dataset
X = []
with open('testData.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        X.append(row)

# load labelset
y = []
with open('labelData.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        y.append(row)

numExamples     = 4          # = number of samples in the sense of how many independend data sets
numValid        = 2
numTimeSteps    = len(X[0])  # length of a sequence
inputLength     = 1          # in each time step the length of the input sequence // number of Features

X_train = np.array(X[0:4])
y_train = np.array(y[0:4])

X_valid = np.array(X[4:6])
y_valid = np.array(y[4:6])

# data needs to be in format [Examples, TimeSteps, Features]
X_train = X_train.reshape(numExamples,numTimeSteps,inputLength)
X_valid = X_valid.reshape(numValid,numTimeSteps,inputLength)

print(X_train.shape)
print(y_train.shape)

numBatch = 1



model = Sequential()

model.add(LSTM(32,input_shape=(numTimeSteps,inputLength)))

# last layer, activation sigmoid for classification
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train,y_train,epochs = 2,batch_size=numBatch)

score = model.evaluate(X_valid, y_valid, batch_size=1)

print(score)

