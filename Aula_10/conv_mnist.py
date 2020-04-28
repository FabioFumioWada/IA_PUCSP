from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])
print(X_train[0].shape)

#reshape data to fit the model (6000 to train / 10000 to test). Size is 28,28,1 - grayscale
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#one-hot encode target column
print(y_train[0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[0])

#create model
model = Sequential()
#add model layers 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2'))
model.add(Flatten())
model.add(Dense(10, activation='softmax', name='dense_1'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 4 images in the test set
model.predict(X_test[:4])
print(y_test[:4])


