'''
Binary classification: mine vs. non-mine

TODO:
- DeepSat:
    http://csc.lsu.edu/~saikat/deepsat/
    https://github.com/romanegloo/deepsat 
    https://github.com/kkgadiraju/SAT-Classification-Using-CNN
'''

from minalytics import (
    get_full_data,
    get_patch_from_grid,
    get_lat_lon_cols,
    myrepr,
    cache,
    MAG_PATCH_SIZE_M,
    BINARY_MAGNETIC_GRID_PATH
)

from gdal_grid import GDALGrid
from random import shuffle

import numpy as np
import percache

def get_empty_patches(grid, nonempty_lon_lat_tups, patch_size_m):
  empty_patches = []

  cols = int(patch_size_m / abs(grid.dx))
  rows = int(patch_size_m / abs(grid.dy))
  arr = grid.arr
  ignore_mask = np.ones(arr.shape) * False
  for lon, lat in nonempty_lon_lat_tups:
    cx, cy = grid.lonlat2pixel(lon, lat)
    y0 = max(cy-rows, 0)
    y1 = min(cy+rows, arr.shape[0])
    x0 = max(cx-cols, 0)
    x1 = min(cx+cols, arr.shape[1])
    arr[x0:x1, y0:y1] = True

  min_lon, max_lat = grid.pixel2lonlat(0, 0)
  max_lon, min_lat = grid.pixel2lonlat(grid.x_size-1, grid.y_size-1)
  lon_range = max_lon - min_lon
  lat_range = max_lat - min_lat
  grid_min = grid.arr.min()
  
  while len(empty_patches) < len(nonempty_lon_lat_tups):
    lon = min_lon + np.random.random_sample() * lon_range
    lat = min_lat + np.random.random_sample() * lat_range
    rval = get_patch_from_grid(
        grid,
        lon,
        lat,
        patch_size_m=patch_size_m,
        return_idxs=True,
        square_only=True
    )
    if rval is None:
      continue
    patch, x0, x1, y0, y1 = rval 
    if np.any(patch == grid_min):
      print('Empty vals in patch, ignoring...')
      continue
    ignore_mask_patch = ignore_mask[x0:x1, y0:y1]
    if any(np.nonzero(ignore_mask_patch)):
      print('Patch overlapped with knowon mine, ignoring...')
      continue
    empty_patches.append(patch)

  return empty_patches

@cache
def load_data(visualize=False):
  mine_patches, df, cols_by_type, target_cols = get_full_data(
      patch_size_m=MAG_PATCH_SIZE_M
  )
  grid = GDALGrid(BINARY_MAGNETIC_GRID_PATH)

  lat_col, lon_col = get_lat_lon_cols(df.columns)

  nonempty_lon_lat_tups = [tuple(x) for x in df[[lon_col, lat_col]].values]
  empty_patches = get_empty_patches(
      grid,
      nonempty_lon_lat_tups,
      patch_size_m=MAG_PATCH_SIZE_M
  )

  if visualize:
    from matplotlib import pyplot as plt
    N = 5
    patches = mine_patches[:N] + empty_patches[:N]
    for i, patch in enumerate(patches):
      plt.subplot(N, N, i+1)
      plt.show(patch)
    plt.show()
    import sys; sys.exit()

  X = np.array(mine_patches + empty_patches)
  X = X[:,:,:,np.newaxis]
  y = np.array([1 for _ in mine_patches] + [0 for _ in empty_patches])

  TRAIN_TEST_SPLIT = 0.9

  # Split at the given index
  n_images = len(y)
  split_index = int(TRAIN_TEST_SPLIT * n_images)
  shuffled_indices = np.random.permutation(n_images)
  train_indices = shuffled_indices[0:split_index]
  test_indices = shuffled_indices[split_index:]

  # Split the images and the labels
  x_train = X[train_indices, :, :]
  y_train = y[train_indices]
  x_test = X[test_indices, :, :]
  y_test = y[test_indices]

  return (x_train, y_train), (x_test, y_test)

###

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 2
epochs = 100
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_magnetism_trained_model.h5'

# The data, split between train and test sets:
USE_CIFAR = False
print('load_data(), USE_CIFAR: %s' % USE_CIFAR)
if USE_CIFAR:
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
else:
  (x_train, y_train), (x_test, y_test) = load_data(visualize=True)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
