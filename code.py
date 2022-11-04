
# import packages
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Convolution2D, MaxPooling2D 
from keras.utils import np_utils
from keras.datasets import mnist 
 
# freezes randomization state
np.random.seed(123) 

# loads mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# displays sample image from training set 
plt.imshow(X_train[0])
plt.show()

# reshaping data to fit our model's input shape
X_train = X_train.reshape(X_train.shape[0],28, 28, 1) 
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# changes the datatype of the training and testing sets
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# standardizes data by reducing the range to between 0 and 1
X_train /= 255
X_test /= 255

# one-hot encoding
Y_train = np_utils.to_categorical(y_train, 10) 
Y_test = np_utils.to_categorical(y_test, 10)

# builds the neural network using keras
model = Sequential()

# convolutional layers to extract features from image data
# =============================================================================
# Add as many convolutional layers and play around with the parameters (number
# of filters and kernel size)
# =============================================================================
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1))) 
model.add(Convolution2D(32, (3, 3), activation='relu'))

# further condenses the important information extracted from the image
model.add(MaxPooling2D(pool_size=(2,2)))

# prevents overfitting by randomly dropping layers
model.add(Dropout(0.25))

# flattens the data into a vector
model.add(Flatten())

# densely connected layers that apply weights to the feature vector
# =============================================================================
# Add as many dense layers and play around with the number of nodes and type of
# activation function
# =============================================================================
model.add(Dense(128, activation='relu'))


model.add(Dropout(0.5))

# final dense layer has to have as many nodes as classes
model.add(Dense(10, activation='softmax'))

# compiles the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# runs the training
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)
           
 
for i in np.random.choice(np.arange(0, len(Y_test)), size = (10,)):
	
	probs = model.predict(X_test[np.newaxis, i])
	prediction = probs.argmax(axis=1)
    
	image = (X_test[i] * 255).reshape((28, 28)).astype("uint8")
    
	print("Actual digit is {0}, predicted {1}".format(Y_test[i], prediction[0]))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)


