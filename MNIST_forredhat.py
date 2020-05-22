#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import Convolution2D


# In[ ]:


from keras.datasets import mnist


# In[ ]:


from keras.layers import Dense


# In[ ]:


from keras.layers import Flatten


# In[ ]:


from keras.models import Sequential


# In[ ]:


from keras.layers import MaxPooling2D


# In[ ]:


from keras.utils import np_utils


# In[ ]:


(x_train, y_train), (x_test, y_test)  = mnist.load_data()
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image 
inputshape = (img_rows, img_cols, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[ ]:


model=Sequential()


# In[ ]:


model.add(Convolution2D(filters=32 ,kernel_size=(3,3), activation='relu',input_shape=inputshape))


# In[ ]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[ ]:


model.add(Convolution2D(filters=32 ,kernel_size=(3,3), activation='relu'))


# In[ ]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[ ]:


model.add(Flatten())


# In[ ]:


model.add(Dense(units=128 , activation='relu'))


# In[ ]:


model.add(Dense(units=32 , activation='relu'))


# In[ ]:


model.add(Dense(units=10,activation='softmax'))


# In[ ]:


from keras.optimizers import Adam


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


# Training Parameters
batch_size = 128
epochs = 1

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model.save("mnist_LeNet.h5")

# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:




