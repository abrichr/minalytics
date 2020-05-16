#coding=utf-8

try:
    import keras
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import os
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import uniform
except:
    pass

try:
    from keras.callbacks import ModelCheckpoint, LambdaCallback
except:
    pass

try:
    from keras.datasets import cifar10
except:
    pass

try:
    from keras.layers import Dense, Dropout, Activation, Flatten
except:
    pass

try:
    from keras.layers import Conv2D, MaxPooling2D
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.preprocessing.image import ImageDataGenerator
except:
    pass

try:
    from matplotlib import pyplot as plt
except:
    pass

try:
    from random import shuffle
except:
    pass

try:
    from gdal_grid import GDALGrid
except:
    pass

try:
    from minalytics import get_full_data, get_patch_from_grid, get_lat_lon_cols, myrepr, cache, MAG_PATCH_SIZE_M, BINARY_MAGNETIC_GRID_PATH
except:
    pass

try:
    import ipdb
except:
    pass

try:
    import ipdb
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

(x_train, y_train), (x_test, y_test) = load_data(
    #read_cache=False,
    nodata_to_mean=False
)
if truncate:
  def truncate(z):
    print('truncate() before: %s' % str(z.shape))
    z = z[:int(len(z) * TRUNCATE), ...]
    print('truncate() after: %s' % str(z.shape))
    return z
  x_train = truncate(x_train)
  y_train = truncate(y_train)

if normalize:
  # XXX accuracy goes to 50%
  new_min = 0
  new_max = 1
  mn, mx = get_min_max('before normalize', [x_train, x_test])
  # shift to [0, 1]
  x_train = (x_train - mn) / (mx - mn)
  x_test = (x_test - mn) / (mx - mn)
  # shift to [new_min, new_max]
  x_train = (x_train * (new_max - new_min)) + new_min
  x_test = (x_test * (new_max - new_min)) + new_min
  get_min_max('after normalize', [x_train, x_test])

if log:
  mn, mx = get_min_max('before log', [x_train, x_test])
  x_train = np.log(x_train)
  x_test = np.log(x_test)
  get_min_max('after log', [x_train, x_test])

# XXX the network doesn't learn without this for some reason
if norm_hack:
  x_train /= 255
  x_test /= 255

datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of the dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=True,
)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Convert class vectors to binary class matrices.
num_classes = len(set(np.concatenate((y_train, y_test))))
print('num_classes: %s' % num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



def keras_fmin_fnct(space):

  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same',
                   input_shape=x_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # TODO: hyperparam search
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # TODO: hyperparam search
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  # TODO: hyperparam search
  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
  opt = keras.optimizers.Adam(
      lr=space['lr'],
      beta_1=0.9,
      beta_2=0.999,
      epsilon=None,
      decay=space['decay'],
      amsgrad=False
  )

  # train model
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

  '''
  model_path = os.path.join(save_dir, 'partial_' + model_name)
  checkpoint = ModelCheckpoint(
      model_path,
      monitor='loss',
      verbose=1,
      save_best_only=True,
      mode='min'
  )
  plot_callback = LambdaCallback(
      #on_epoch_end=lambda epoch, logs: plot_grids(
      on_train_end=lambda logs: plot_grids(
        x_test,
        y_test.argmax(axis=1),
        model.predict(x_test, verbose=1).argmax(axis=1)
      ),
  )
  callbacks = [
      #checkpoint,
      #plot_callback
  ]
  '''
  if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        #callbacks=callbacks
                        )
  else:
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=batch_size),
                                  epochs=epochs,
                                  validation_data=(x_test, y_test),
                                  workers=4,
                                  #callbacks=callbacks
                                  )

  score, acc = model.evaluate(x_test, y_test, verbose=1)

  return {'loss': -acc, 'status': STATUS_OK, 'model': model}#, 'history': history}

def get_space():
    return {
        'lr': hp.choice('lr', [0.001, 0.003, 0.1]),
        'decay': hp.choice('decay', [0.0, 1e-6, 3e-6]),
    }
