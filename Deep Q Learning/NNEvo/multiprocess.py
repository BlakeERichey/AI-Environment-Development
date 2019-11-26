import numpy as np

xdata = np.random.randn(100, 8)
ytrue = np.random.randint(0, 2, 100)

class MyClass():

  def __init__(self):
    self.models = [self.create_model() for _ in range(6)]

  def fit(self, arr, i):
      with tf.Session(graph=tf.Graph()) as sess:
          K.set_session(sess)
          model = Sequential()
          model.add(Dense(12, input_dim=8, activation='relu'))
          model.add(Dense(8, activation='relu'))
          model.add(Dense(1, activation='sigmoid'))

          model.compile(loss='binary_crossentropy', optimizer='adam')
          model.fit(xdata, ytrue, verbose=0)
          
          res = model.evaluate(xdata, ytrue, verbose=0)
          print(res)
          arr[i] = res
          return res

  def create_model(self):
      model = Sequential()
      model.add(Dense(512, input_dim=8, activation='relu'))
      model.add(Dense(512, activation='relu'))
      model.add(Dense(1, activation='sigmoid'))

      model.compile(loss='binary_crossentropy', optimizer='adam')
      print('Model created.')
      return model

def quality(arr=None, i=0, model=None):
    print('testing model...')
    res = model.evaluate(xdata, ytrue, verbose=0)
    arr[i] = res
    print('Model result', res)

#if __name__ != '__mp_main__': #This line solves the problem
import keras
import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
    
from multiprocessing import Pool, Process, Queue, Array

if __name__ == '__main__':
  obj = MyClass()
  threads = []
  arr = Array('f', range(6))

  # models = []
  # for _ in range(6):
  #     models.append(create_model())

  for i in range(6):
      d = {'arr': arr, 'i': i, 'model': obj.models[i]}
      print(d['model'])
      p = Process(target=quality, kwargs=d)
      threads.append(p)
      p.start()

  for thread in threads:
      thread.join()

  test = [x for x in arr]
  print('Results', test)
  print(keras.backend)