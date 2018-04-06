import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]],"float32")
Y = np.array([[0],[1],[1],[0]],"float32")


model = Sequential()
model.add(Dense(3,input_dim = 2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
print(model.summary())

model.fit(X,Y,nb_epoch=2000,verbose=2)
print(model.predict(X).round())