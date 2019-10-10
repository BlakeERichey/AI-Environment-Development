#learn classifiers from a video

import gym
import os, datetime, random
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt
from   tensorflow.keras.optimizers import Adam
from   collections                 import deque
from   tensorflow.keras            import backend
from   tensorflow.keras.models     import Sequential
from   tensorflow.python.client    import device_lib
from   tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint
from   tensorflow.keras.layers     import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM

#-------------------- Load images --------------------
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

num_images = 119
images = []
for i in range(num_images):
  filename = f'./images/image{i}.jpg'
#  filename = f'./test_image.jpg'
  image = load_img(filename, target_size=(256, 256))
  # print('PIL image size',image.size) #(256,256)
  numpy_image = img_to_array(image)
  # print('numpy array size',numpy_image.shape) #(256,256, 3)
  image_batch = np.expand_dims(numpy_image, axis=0) 
  # print('image batch size', image_batch.shape) #(1, 256,256, 3)
  images.append(numpy_image)
#images_batch = np.array([np.expand_dims(image, axis=0) for image in images])
images_batch = np.array(images)
images_batch.reshape(num_images,256,256,3)

#print('image:', images[0])
#plt.imshow(np.uint8(images_batch[0])) #render an image

classes = [0 for i in range(23)]
classes += [1 for i in range(23, 77)]
classes += [0 for i in range(77, 119)]

classes = to_categorical(classes)
print('Classes:', classes[0], classes[56])
print('Shape:', classes.shape)



#-------------------- Create Model --------------------
model = Sequential()

num_layers = 3
nodes_per_layer = [128, 128, 64]
filter_size = 3
num_classes = 2

#pooling options
stride_size = 2
pool_size = 2
for layer in range(num_layers):

  try:
    nodes=nodes_per_layer[layer]
  except IndexError:
    nodes = None

  if nodes is None:
    nodes = 32

  if layer == 0:
    #input layer
    model.add(Conv2D(nodes, kernel_size=filter_size, activation='relu', \
      input_shape=(256,256,3)))
  else:
    #add hidden layers
    model.add(Conv2D(nodes, kernel_size=filter_size, activation='relu'))

  model.add(MaxPooling2D(pool_size=pool_size, strides=stride_size))

model.add(Flatten())
#output layer
model.add(Dense(num_classes, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer=Adam(lr=0.001), \
  loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#-------------------- train model --------------------
model.fit(images_batch, classes, verbose=1, batch_size=12, epochs=10, validation_split=0.2)
model.save_weights('./model/'+f'model.h5')

#-------------------- Test model --------------------
#model.load_weights('./model/'+f'model.h5')
#print(model.predict(np.expand_dims(images_batch[0], axis=0)))
#print(model.predict(images_batch[73:82]))