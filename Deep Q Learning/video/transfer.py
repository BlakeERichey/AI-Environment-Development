# %matplotlib inline
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.applications import ResNet50
from keras.datasets import cifar10
from keras.models import Model
from keras import layers

batch_size = 64
epochs = 100

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

num_images = 607
images = []
for i in range(num_images):
  filename = f'./images/image{i}.jpg'
#  filename = f'./test_image.jpg'
  image = load_img(filename, target_size=(32, 32))
  # print('PIL image size',image.size) #(256,256)
  numpy_image = img_to_array(image)
  # print('numpy array size',numpy_image.shape) #(256,256, 3)
  image_batch = np.expand_dims(numpy_image, axis=0) 
  # print('image batch size', image_batch.shape) #(1, 256,256, 3)
  images.append(numpy_image)
#images_batch = np.array([np.expand_dims(image, axis=0) for image in images])
images_batch = np.array(images)
images_batch.reshape((num_images,32,32,3))

#print('image:', images[0])
#plt.imshow(np.uint8(images_batch[0])) #render an image

zeros = [(0,4), (75, 116), (208, 277), (387, 456), (576, 607)]
ones =  [(4, 75), (116, 208), (277, 387), (456, 576)]

classes = [0 for _ in range(num_images)]

for tup in zeros:
  a = tup[0]
  b = tup[1]
  for i in range(a, b):
    classes[i] = 0

for tup in ones:
  a = tup[0]
  b = tup[1]
  for i in range(a, b):
    classes[i] = 1

classes = to_categorical(classes)

pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=images_batch.shape[1:])

print(pretrained.input_shape)
print(pretrained.output_shape)

flattened = layers.Flatten()(pretrained.output)
output = layers.Dense(2, activation='softmax')(flattened)
pretrained = Model(pretrained.inputs, output)
pretrained.summary()

print(pretrained.output_shape)

pretrained.compile(RMSprop(lr=1e-3), 'categorical_crossentropy', metrics=['acc'])\

pretrained.load_weights('./model/transfer_model.h5')

pretrained_history = pretrained.fit(images_batch, classes, validation_split=0.1,
#                                    batch_size=batch_size, epochs=epochs)

score = pretrained.evaluate(images_batch, classes, batch_size=batch_size)
print(f'Pretrained model accuracy: {score[1]}')
pretrained.save_weights("./model/transfer_model.h5")

fig, ax = plt.subplots()

ax.plot(pretrained_history.epoch, pretrained_history.history['val_acc'], label='Pretrained')

ax.legend()
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Accuracy')               

#test on new images
#num_images = 1
#images = []
#filename = f'./tests/img1.jpg'
##  filename = f'./test_image.jpg'
#image = load_img(filename, target_size=(32, 32))
## print('PIL image size',image.size) #(256,256)
#numpy_image = img_to_array(image)
## print('numpy array size',numpy_image.shape) #(256,256, 3)
#image_batch = np.expand_dims(numpy_image, axis=0) 
## print('image batch size', image_batch.shape) #(1, 256,256, 3)
#images.append(numpy_image)
##images_batch = np.array([np.expand_dims(image, axis=0) for image in images])
#images_batch = np.array(images)
#images_batch.reshape((num_images,32,32,3))
#
#classes = [1]
#
#print(pretrained.predict(np.expand_dims(images_batch[0], axis=0)))