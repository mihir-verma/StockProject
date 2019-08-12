import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#from sklearn.svm import SVC
#np.set_printoptions(threshold=np.inf)

def normalize(dataset):
    normalized = (dataset - dataset.mean()) / dataset.std()
    return normalized
    
def scale(dataset,min=0,max=1):
    scaler = preprocessing.MinMaxScaler((min,max))
    dataset = scaler.fit_transform(dataset)
    return dataset
    
def getX():
    data = pd.read_pickle('reliance_prepared.pickle')
    data.dropna(inplace=True)
    data.drop(data.last_valid_index(), inplace=True)
    return data
    
def getY(choice=1): #1:AsItIsTomorrow 2:Diff 3:1HotCol 4:1HotCategorical
    data = pd.read_pickle('reliance_prepared.pickle')
    data.dropna(inplace=True)
    #
    if choice==1:
        y = data['Close'].shift(-1)
        y.dropna(inplace=True)
        return y
    #
    diff = data['Close'].diff()
    y = diff.shift(-1)
    y.dropna(inplace=True)
    #
    if choice==2:
        return y
    #
    y[y > 0] = 1
    y[y < 0] = 0
    #
    if choice==3:
        return y
    #
    y = to_categorical(y)
    #
    if choice==4:
        return y
    #
    print("Error")

X = getX()
X = X.filter(items=['bol_upp','bol_low','rsi_val', 'macd_histogram'])
X_scale = scale(X,-1,1)
#X = normalize(X)
y = getY(4)
#y = normalize(y)
#y = scale(y)
#y = np.array(y).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_scale,y,train_size=0.9, shuffle=True)


print(y_train)


n = X_train.shape[1]

#model = Sequential()
model = load_model('model_sup.h5')

#model.add(Dense(32, activation='relu', input_shape=(n,)))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(2, activation='softmax'))
#model.add(Dense(1, activation='linear'))
#model.compile(optimizer='adam' ,loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train, validation_split=0.2, epochs=20, verbose=2)

#model = SVC()
#model ter.fit(X_train, y_train)

err, acc = model.evaluate(X_test, y_test, verbose=1)
print(err, acc)
#print(model.metrics_names)

predictions = model.predict(X_test)
predictions = pd.DataFrame(predictions)
#predictions[predictions > 0] = 1
#predictions[predictions < 0] = 0
#if predictions['1'] > predictions['0']:
#    predictions['z'] = 1
#else:
#    predictions['z'] = 0

#y_test = pd.DataFrame(y_test)
#y_test.reset_index(inplace=True)
#y_test.drop('Date', axis=1, inplace=True)

#predictions['actual'] = y_test['Close']

#predictions = predictions - X_test['Close']
#predictions['actual'] = y_test['Close'] - X_test['Close']


print(predictions)
y_test = pd.DataFrame(y_test)
y_test.to_csv('TestPredicions.csv')
print(X_train)
#print(y_test)
#predictions.to_csv('predictions.csv')
model.save('model_sup.h5')
#model.summary()'''