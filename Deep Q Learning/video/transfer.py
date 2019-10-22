#%matplotlib inline
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import axes
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.applications import ResNet50
from keras.datasets import cifar10
from keras.models import Model
from keras import layers
from tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint, EarlyStopping

batch_size = 32
epochs = 35
dim = 192

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

num_images = 1009
images = []
for i in range(num_images):
  filename = f'./images/image{i}.jpg'
#  filename = f'./test_image.jpg'
  image = load_img(filename, target_size=(dim, dim))
  numpy_image = img_to_array(image)
  images.append(numpy_image)

images_batch = np.array(images)
images_batch.reshape((num_images,dim,dim,3))

#print('image:', images[0])
#plt.imshow(np.uint8(images_batch[700])) #render an image

#zeros = [(0,4), (75, 116), (208, 277), (387, 456), (576, 607)]
#ones =  [(4, 75), (116, 208), (277, 387), (456, 576)]
#
#classes = [0 for _ in range(num_images)]
#
#for tup in zeros:
#  a = tup[0]
#  b = tup[1]
#  for i in range(a, b):
#    classes[i] = 0
#
#for tup in ones:
#  a = tup[0]
#  b = tup[1]
#  for i in range(a, b):
#    classes[i] = 1

classes = [0 for _ in range(0, 272)]
classes += [1 for _ in range(272, 715)]
classes += [2 for _ in range(715, 1009)]

classes = to_categorical(classes)

images_batch, images_valid, classes, classes_valid = train_test_split(images_batch, classes, test_size=.1)
pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=images_batch.shape[1:])

flattened = layers.Flatten()(pretrained.output)
output = layers.Dense(3, activation='softmax')(flattened)
pretrained = Model(pretrained.inputs, output)
pretrained.summary()



pretrained.compile(RMSprop(lr=1e-3), 'categorical_crossentropy', metrics=['acc'])\


#ckpt = ModelCheckpoint('./model/best_model.h5', monitor='val_loss', \
#    verbose=0, save_weights_only=True, save_best_only=True)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
pretrained_history = pretrained.fit(images_batch, classes, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(images_valid, classes_valid), callbacks=[es])

#pretrained.load_weights('./model/transfer_model.h5')
score = pretrained.evaluate(images_batch, classes, batch_size=batch_size)
print(pretrained.input_shape)
print(pretrained.output_shape)
print(f'Pretrained model accuracy: {score[1]}')
pretrained.save_weights("./model/transfer_model.h5")

#fig, ax = plt.subplots()

#ax.plot(pretrained_history.epoch, pretrained_history.history['val_acc'], label='Pretrained')

#ax.legend()
#ax.set_xlabel('Epoch Number')
#ax.set_ylabel('Accuracy')               

#test on new images
num_images = 1
images = []
filename = f'./test_image.jpg'
#  filename = f'./test_image.jpg'
image = load_img(filename, target_size=(dim, dim))
numpy_image = img_to_array(image)
images.append(numpy_image)
images_batch = np.array(images)
images_batch.reshape((num_images,dim,dim,3))

classes = [1]

#pretrained.load_weights('./model/transfer_model.h5')
print(pretrained.predict(np.expand_dims(images_batch[0], axis=0)))
plt.imshow(np.uint8(images_batch[0])) #render an image
#pretrained.load_weights('./model/best_model.h5')