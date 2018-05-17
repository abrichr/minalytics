'''
Binary classification: mine vs. non-mine

TODO:
- DeepSat:
    http://csc.lsu.edu/~saikat/deepsat/
    https://github.com/romanegloo/deepsat 
    https://github.com/kkgadiraju/SAT-Classification-Using-CNN
'''
import keras
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from random import shuffle

from gdal_grid import GDALGrid
from minalytics import (
    get_full_data,
    get_patch_from_grid,
    get_lat_lon_cols,
    myrepr,
    cache,
    MAG_PATCH_SIZE_M,
    BINARY_MAGNETIC_GRID_PATH
)

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
def load_data(nodata_to_mean=False):
  mine_patches, df, cols_by_type, target_cols = get_full_data(
      patch_size_m=MAG_PATCH_SIZE_M
  )
  grid = GDALGrid(BINARY_MAGNETIC_GRID_PATH)
  arr = grid.arr
  if nodata_to_mean:
    arr[arr == arr.min()] = arr.mean()  # 0

  lat_col, lon_col = get_lat_lon_cols(df.columns)

  nonempty_lon_lat_tups = [tuple(x) for x in df[[lon_col, lat_col]].values]
  empty_patches = get_empty_patches(
      grid,
      nonempty_lon_lat_tups,
      patch_size_m=MAG_PATCH_SIZE_M
  )

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

def plot_grids(x, y, y_pred=None, nrows=6, ncols=3):
  N = nrows * ncols

  mine_patch_idxs = y == 1
  empty_patch_idxs = y == 0
  mine_patches = x[mine_patch_idxs][:,:,:,0]
  empty_patches = x[empty_patch_idxs][:,:,:,0]

  data_vmin = min((mine_patches.min(), empty_patches.min()))
  data_vmax = max((mine_patches.max(), empty_patches.max()))
  print('data_vmin: %.2f, data_vmax: %.2f' % (data_vmin, data_vmax))

  mine_patches = mine_patches[:N]
  empty_patches = empty_patches[:N]
  show_vmin = min((mine_patches.min(), empty_patches.min()))
  show_vmax = min((mine_patches.max(), empty_patches.max()))
  print('show_vmin: %.2f, show_vmax: %.2f' % (show_vmin, show_vmax))

  for title, patches in [
      ('mine patches', mine_patches),
      ('non-mine patches', empty_patches)
  ]:
    print('len(patches): %s' % len(patches))
    plt.figure()
    for i, patch in enumerate(patches):
      print('i: %s, patch.shape: %s, min: %.2f, max: %.2f' % (
        i, patch.shape, patch.min(), patch.max()))
      def plot(pos, vmin, vmax):
        plt.subplot(nrows, ncols*2, pos)
        plt.imshow(patch, vmin=vmin, vmax=vmax)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if y_pred is not None:
          if y[i] != y_pred[i]:
            for spine in ax.spines.values():
              spine.set_edgecolor('red')
              spine.set_linewidth(3)
      plot(i*2+1, data_vmin, data_vmax)
      plot(i*2+2, show_vmin, show_vmax)
    title = '%s\n(%.2f, %.2f)\n(%.2f, %.2f)' % (
        title, data_vmin, data_vmax, show_vmin, show_vmax)
    plt.suptitle(title)

  plt.show()

batch_size = 32
num_classes = 2
epochs = 10
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_magnetism_trained_model.h5'

(x_train, y_train), (x_test, y_test) = load_data(
    #read_cache=False,
    nodata_to_mean=False
)

SHOW_HIST = False
if SHOW_HIST:
  vals = np.concatenate((x_train.flatten(), x_test.flatten()))
  plt.hist(vals, log=True)
  plt.show()

TRUNCATE = None  #0.05
if TRUNCATE:
  def truncate(z):
    print('truncate() before: %s' % str(z.shape))
    z = z[:int(len(z) * TRUNCATE), ...]
    print('truncate() after: %s' % str(z.shape))
    return z
  x_train = truncate(x_train)
  y_train = truncate(y_train)

def get_min_max(name, arrs):
  mn = min([arr.min() for arr in arrs])
  mx = max([arr.max() for arr in arrs])
  print('%s, mn: %.2f, mx: %.2f' % (name, mn, mx))
  return mn, mx

NORMALIZE = True
if NORMALIZE:
  # XXX accuracy goes to 50%
  new_min = 1
  new_max = 10
  mn, mx = get_min_max('before normalize', [x_train, x_test])
  # shift to [0, 1]
  x_train = (x_train - mn) / (mx - mn)
  x_test = (x_test - mn) / (mx - mn)
  # shift to [new_min, new_max]
  x_train = (x_train * (new_max - new_min)) + new_min
  x_test = (x_test * (new_max - new_min)) + new_min
  get_min_max('after normalize', [x_train, x_test])

LOG = True
if LOG:
  mn, mx = get_min_max('before log', [x_train, x_test])
  x_train = np.log(x_train)
  x_test = np.log(x_test)
  get_min_max('after log', [x_train, x_test])

PLOT_BEFORE = True
if PLOT_BEFORE:
  plot_grids(x_test, y_test)

# XXX the network doesn't learn without this for some reason
NORM_HACK = True
if NORM_HACK:
  x_train /= 255
  x_test /= 255


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

#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.Adam(
    lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=0.0,
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
'''
callbacks = [
    #checkpoint,
    #plot_callback
]

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
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

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=batch_size),
                                  epochs=epochs,
                                  validation_data=(x_test, y_test),
                                  workers=4,
                                  callbacks=callbacks)

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

plot_grids(
  x_test,
  y_test.argmax(axis=1),
  model.predict(x_test, verbose=1).argmax(axis=1)
)

val_acc = history.history['val_acc']
plt.plot(val_acc)
plt.title('Validation Accuracy')
plt.show()

import ipdb; ipdb.set_trace()
