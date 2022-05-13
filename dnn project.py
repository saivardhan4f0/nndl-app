import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import keras  #Keras is the deep learning library that helps you to code Deep Neural Networks with fewer lines of code
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop,Adadelta,SGD,Adagrad,Adam,Adamax,Nadam
#import pylab as plt
#import seaborn as sns #For data visualization
import pandas as pd # For Data manipulation

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
dataframe = pd.read_csv("Crop_recommendation modi2.csv")
dataframe.head()
dataset = dataframe.values
X = dataset[0:,0:7].astype(float)
Y = dataset[0:,7]
print(X[0:])
from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
red_wine_data_X=ms.fit_transform(X)
print(red_wine_data_X)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(red_wine_data_X, dummy_y, test_size=0.2, random_state=0)
print(y_test[0:5])
print(X_train)
First_Layer_Size = 64 # Number of neurons in first layer
model=Sequential()
model.add(Dense(First_Layer_Size,activation='tanh', input_shape=(7,)))
model.add(Dense(64,activation='tanh'))
model.add(Dense(64,activation='tanh'))
model.add(Dense(22,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=1,epochs=10,verbose=1)
# Write the testing input and output variables
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Write the index of the test sample to test
prediction = model.predict(X_test[70].reshape(1,7))
print(prediction[0])
print(np.round(prediction[0]))
print(y_test[70])
features=np.array([[77,55,88,55,44,54,6]])
print(model.predict(features))
model.save('CropPrediction.h5')
list_of_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
input_data=pd.DataFrame(columns=list_of_columns)


input_data.at[0,'N']=int(input('enter nitrogen level(N)'))
input_data.at[0,'P']=int(input('enter phosphorus level(P)'))
input_data.at[0,'K']=int(input('enter potassium level(K)'))
input_data.at[0,'temperature']=float(input ('enter temperature'))
input_data.at[0,'humidity']=float(input ('enter humidity '))
input_data.at[0,'ph']=float(input('enter pH value of soul'))
input_data.at[0,'rainfall']=float(input('enter rainfall value'))

from keras.models import load_model
model2=load_model('CropPrediction.h5')
prediction = model2.predict(np.asarray(input_data).astype(np.float32).reshape(1,7))
result = prediction[0]
#print('crop  ',result)

label=['rice','banana','black gram','chickpea','coconut','coffee','cotton','grapes','jute','kidneybeans','lentil','maize','mango','mothbeans','mungbean','muskmelon','orange','papaya','pegionpeas','promagranate','rice','watermelon']
thresholded = (result>0.5)*1
ind = np.argmax(thresholded)
print("Suggested Crop is",label[ind])