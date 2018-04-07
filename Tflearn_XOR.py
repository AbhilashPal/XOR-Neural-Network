import numpy as np 
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression

X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]
model = input_data(shape=[None,2])
model = fully_connected(model,2,activation='tanh')
model = fully_connected(model,1,activation='tanh')
regression=regression(model,optimizer='sgd',learning_rate=5,
		loss = 'binary_crossentropy')
model = tflearn.DNN(regression)
model.fit(X,Y,n_epoch=5000,show_metric = True)
L = [i[0] > 0 for i in model.predict(X)]
print(L)