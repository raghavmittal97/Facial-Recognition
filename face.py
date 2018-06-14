import os
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
#CNN for Handwritten digit recognition
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import itertools
from keras import optimizers
from scipy.misc import imresize
import cv2

path="/Users/mlproject/Desktop/DatafacesAshoka"
names=[]
X=[]
Y=[]
Z=[]
seed = 7
for x,y,z in os.walk(path):
	for i in y:
		for a,b,c in os.walk(path+"/"+i):
			for j in c:
				img=cv2.imread(path+"/"+i+"/"+j)
				resized = imresize(img, (150,150), 'bilinear')
				norm_image=cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
				X.append(norm_image)
				Y.append(i)




Z=os.listdir("/Users/mlproject/Desktop/DatafacesAshoka")
# print(len(Z))
Z.remove(Z[0])
# print(len(Z))
# print(Z[0])
# print (Z[1])
# print(Z[71])
x=0


'''for i in Z:
	for j in range(len(Y)):
		if(Y[j]==i):
			Y[j]=[0]*len(Z)
			Y[j][x]=1

	x=x+1'''

#print(Y[1])
for i in Z:
	for j in range(len(Y)):
		if(Y[j]==i):
			Y[j]=[x]

	x=x+1
#
X = np.array(X)
Y = np.array(Y)

#print(Y)
#print(len(Y))
# print((Y))


#for image in X:
#print(image.shape)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=seed)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train)
# print(X_train[0])
# print(len(y_train[1]))
num_classes = y_test.shape[1]

X_train = X_train.reshape(X_train.shape[0], 150, 150, 3)
X_test = X_test.reshape(X_test.shape[0], 150, 150, 3)


#print(y_test[56])


# change learning rate to 0.001, 0.0001

model = Sequential()
#model.load_weights('weights.h5')
#model.load_weights('plotting.h5')
model.add(Conv2D(3, (2, 2), padding='same',kernel_initializer = 'normal', input_shape=(150, 150, 3),activation='elu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
#model.add(Dropout(0.3))
#model.add(Conv2D(32, (5, 5), activation='relu'))
#model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Dropout(0.3))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer = 'normal')) 
model.add(Dense(12, activation='sigmoid', kernel_initializer = 'normal'))
sgd = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.add(Activation('softmax'))
model.load_weights('weights3.h5')
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#history=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=32)

#epsilon=1e-08,

#model.save_weights('weights.h5')
#model.save_weights('weights3.h5')

img2=cv2.imread("/Users/mlproject/Desktop/TestImages/Shourya.jpg")
resized2 = imresize(img2, (150,150), 'bilinear')
norm_image2=cv2.normalize(resized2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
test = norm_image2
test = test.reshape(1, 150, 150, 3)
l = model.predict(test)
print(l)

c=0
ht=0
for i in range(len(l[0])):
	if(l[0][i]>ht):
		ht=l[0][i]
		c=i

print(c)
zz=os.listdir("/Users/mlproject/Desktop/DatafacesAshoka")
print("The person is "+ zz[c+1])



'''plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.show()'''




#online model

# model = Sequential()
# model.add(Conv2D(32, (5, 5), padding='valid', input_shape=(200, 180, 3),activation='relu'))
# model.add(Conv2D(32, (5, 5), activation='relu'))
# model.add(Conv2D(32, (5, 5), activation='relu'))
# model.add(Dense(512,input_shape=(200, 180, 3)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(15))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))
# model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=32)



#print(X_train.shape)
#print(X_test.shape)
