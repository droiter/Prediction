'''
Created on 13/02/2017

@author: smas255
'''
import time
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
def get_data(filename):    
    priceDataset=np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)  
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))    
    X_dataset=scaler.fit_transform(priceDataset[...,0])
    Y_dataset=scaler.fit_transform(priceDataset[...,1])
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, Y_dataset, test_size=0.3, random_state=1234)
    return scaler,X_train, X_test, y_train, y_test
def build_model():
    model = Sequential()
    layers = [1, 15, 7,1]
    model.add(Dense(
            input_dim=layers[0],
            output_dim=layers[1], activation='tanh'))    
    model.add(Dense(layers[2],activation='tanh'))    
    model.add(Dense(output_dim=layers[3], activation='linear'))    
    
    model.compile(loss="mse", optimizer="rmsprop")   
    return model
if __name__ == '__main__':    
    global_start_time = time.time()
    epochs = 500
    scaler, X_train, X_test, y_train, y_test =get_data('./data/aapl.csv')
    model = build_model()    
    try:
        model.fit(X_train, y_train,batch_size=5, nb_epoch=epochs) 
        # make predictions
        trainPredict = model.predict(X_train)
        testPredict = model.predict(X_test)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([y_train])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([y_test])
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))        
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
    except KeyboardInterrupt:
        print('Training duration (s) : ', time.time() - global_start_time)        
    pass