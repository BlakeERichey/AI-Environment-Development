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
epochs = 25
subset_frac = .99

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train_subset, _, y_train_subset, _ = train_test_split(x_train, y_train, train_size=subset_frac)

print(x_train_subset.shape)

random_start = ResNet50(weights=None, input_shape=x_train.shape[1:], classes=10)

print(random_start.input_shape)

print(random_start.output_shape)

pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=x_train.shape[1:])

print(pretrained.input_shape)
print(pretrained.output_shape)

flattened = layers.Flatten()(pretrained.output)
output = layers.Dense(10, activation='softmax')(flattened)
pretrained = Model(pretrained.inputs, output)
pretrained.summary()

print(pretrained.output_shape)

def get_weights_resnet(model):
    layer_weights = []
    for i, l in enumerate(model.layers):
        if isinstance(l, (layers.Conv2D, layers.Dense)):
            layer_weights.append({
                'layer': i,
                'weights': l.get_weights()[0]
            })
    return pd.DataFrame(layer_weights)

pretrained_start = get_weights_resnet(pretrained)
random_weights_start = get_weights_resnet(random_start)

pretrained.compile(RMSprop(lr=1e-3), 'categorical_crossentropy', metrics=['acc'])

pretrained_history = pretrained.fit(x_train_subset, y_train_subset, validation_split=0.1,
                                    batch_size=batch_size, epochs=epochs)

score = pretrained.evaluate(x_test, y_test, batch_size=batch_size)
print(f'Pretrained model accuracy: {score[1]}')

random_start.compile(RMSprop(lr=1e-3), 'categorical_crossentropy', metrics=['acc'])
random_history = random_start.fit(x_train_subset, y_train_subset,
                                  validation_data=(x_test, y_test),
                                  batch_size=batch_size, epochs=epochs)

score = random_start.evaluate(x_test, y_test, batch_size=batch_size)
print(f'Randomized start model accuracy: {score[1]}')

fig, ax = plt.subplots()

ax.plot(pretrained_history.epoch, pretrained_history.history['val_acc'], label='Pretrained')
ax.plot(random_history.epoch, random_history.history['val_acc'], label='Randomized Start')

ax.legend()
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Accuracy')


pretrained_end = get_weights_resnet(pretrained)
pretrained_end['change'] = (pretrained_end['weights'] - pretrained_start['weights']).apply(lambda x: np.linalg.norm(x))
random_weights_end = get_weights_resnet(random_start)
random_weights_end['change'] = (random_weights_end['weights'] 
                                - random_weights_start['weights']).apply(lambda x: np.linalg.norm(x))

fig, ax = plt.subplots()

ax.semilogy(pretrained_end['layer'], pretrained_end['change'], label='Pretrained')
ax.semilogy(random_weights_end['layer'], random_weights_end['change'], label='Randomized Start')

ax.legend()
ax.set_xlabel('Layer Number')
ax.set_ylabel('Mean Absolute Difference')                             