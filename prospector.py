'''
Binary classification: mine vs. non-mine

TODO:
- DeepSat:
    http://csc.lsu.edu/~saikat/deepsat/
    https://github.com/romanegloo/deepsat 
    https://github.com/kkgadiraju/SAT-Classification-Using-CNN
'''

import random
random.seed(123)
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

import keras
import math
import os
from hipsterplot import plot as hipplot
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from pprint import pprint, pformat
from random import shuffle

try:
  from .gdal_grid import GDALGrid
  from .common import (
      get_full_data,
      get_patch_from_grid,
      get_lat_lon_cols,
      myrepr,
      cache,
      MAG_PATCH_SIZE_M,
      BINARY_MAGNETIC_GRID_PATH
  )
  from .utils import LR_Find
except Exception as exc:
  print('exception: %s' % exc)
  from gdal_grid import GDALGrid
  from common import (
      get_full_data,
      get_patch_from_grid,
      get_lat_lon_cols,
      myrepr,
      cache,
      MAG_PATCH_SIZE_M,
      BINARY_MAGNETIC_GRID_PATH
  )
  from utils import LR_Find

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_magnetism_trained_model.h5'

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
  
  num_empty = 0
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
      num_empty += 1
      continue
    ignore_mask_patch = ignore_mask[x0:x1, y0:y1]
    if any(np.nonzero(ignore_mask_patch)):
      print('Patch overlapped with knowon mine, ignoring...')
      continue
    empty_patches.append(patch)
  print('Ignored %d empty patches' % num_empty)

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

  rval = []
  for title, patches in [
      ('mine patches', mine_patches),
      ('non-mine patches', empty_patches)
  ]:
    print('len(patches): %s' % len(patches))
    rval.append(plt.figure())
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

  return rval

def get_min_max(name, arrs):
  mn = min([arr.min() for arr in arrs])
  mx = max([arr.max() for arr in arrs])
  print('%s, mn: %.2f, mx: %.2f' % (name, mn, mx))
  return mn, mx

def do_hist(before_or_after, x_train, y_train, x_test, y_test):
  bins_by_log = {True: None, False: None}
  for i, (log_title, hist_log) in enumerate([('', False), ('log', True)]):
    plt.figure()
    plt.suptitle('%s %s' % (log_title, before_or_after))
    for j, (cls_title, y_targ) in enumerate([('empty', 0), ('mine', 1)]):
      for k, (grp_title, x, y) in enumerate([('train', x_train, y_train), ('test', x_test, y_test)]):
        plt.subplot(2,2,j*2+k+1)
        idxs = y == y_targ
        _x = x[idxs]
        bins = bins_by_log[hist_log]
        _, _bins, _ = plt.hist(_x.flatten(), log=hist_log, bins=bins)
        plt.title('%s %s' % (cls_title, grp_title))
        bins_by_log[hist_log] = bins if bins is not None else _bins

def run(
    lr=0.0001,
    decay=1e-6,

    batch_size=32,
    num_classes=2,
    epochs=10,
    data_augmentation=True,

    show_hist=False,
    truncate_ratio=None,
    norm_bounds=(-1, 1),
    log=False,
    norm_hack=False,
    plot_before=False,
    show_plots=False,
    save_plots=True,
    save_final_model=False,

    find_lr=False,
    stop_early=False,

    featurewise_center=False,
    featurewise_std_normalization=False,

    double_layers=True,

    rotation_range=0,

    batch_norm=True,
    reduce_lr_on_plateau=True,
    record_stats=True
):

  (x_train, y_train), (x_test, y_test) = load_data(
      #read_cache=False,
      nodata_to_mean=False
  )

  if show_hist:
    do_hist('before', x_train, y_train, x_test, y_test)

  if truncate_ratio:
    def truncate(name, z):
      print('truncate() %s before: %s' % (name, str(z.shape)))
      z = z[:int(len(z) * truncate_ratio), ...]
      print('truncate() %s after: %s' % (name, str(z.shape)))
      return z
    x_train = truncate('x_train', x_train)
    y_train = truncate('y_train', y_train)
    x_test = truncate('x_test', x_test)
    y_test = truncate('y_test', y_test)

  if log:
    mn, mx = get_min_max('before log', [x_train, x_test])
    if mn <= 0:
      x_train = x_train - mn + np.finfo(float).eps
      x_test = x_test - mn + np.finfo(float).eps
      mn, mx = get_min_max('before log, after adjusting', [x_train, x_test])
    x_train = np.log(x_train)
    x_test = np.log(x_test)
    get_min_max('after log', [x_train, x_test])

  if norm_bounds:
    # XXX accuracy goes to 50%
    new_min, new_max = norm_bounds
    # technically shouldn't use training data here...
    mn, mx = get_min_max('before normalize', [x_train, x_test])
    # shift to [0, 1]
    x_train = (x_train - mn) / (mx - mn)
    x_test = (x_test - mn) / (mx - mn)
    # shift to [new_min, new_max]
    x_train = (x_train * (new_max - new_min)) + new_min
    x_test = (x_test * (new_max - new_min)) + new_min
    get_min_max('after normalize', [x_train, x_test])

  # XXX the network doesn't learn without this for some reason
  if norm_hack:
    x_train /= 255
    x_test /= 255

  if plot_before:
    plot_grids(x_test, y_test)
    plt.show()

  if show_hist:
    do_hist('after', x_train, y_train, x_test, y_test)
    plt.show()

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
  if batch_norm:
    model.add(BatchNormalization())
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  if double_layers:
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    if batch_norm:
      model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  if batch_norm:
    model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  #opt = keras.optimizers.rmsprop(lr=lr, decay=decay)
  opt = keras.optimizers.Adam(
      lr=lr,
      decay=decay,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=None,
      amsgrad=False
  )

  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

  callbacks = []

  SAVE_MODEL_CB = False
  if SAVE_MODEL_CB:
    model_path = os.path.join(save_dir, 'partial_' + model_name)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks.append(checkpoint)

  PLOT_CB = False
  if PLOT_CB:
    plot_callback = LambdaCallback(
        #on_epoch_end=lambda epoch, logs: plot_grids(
        on_train_end=lambda logs: plot_grids(
          x_test,
          y_test.argmax(axis=1),
          model.predict(x_test, verbose=1).argmax(axis=1)
        ) and plt.show(),
    )
    callbacks.append(plot_callback)

  if stop_early or find_lr:
    assert not (stop_early and find_lr)

  if stop_early:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=2,
        verbose=1,
        mode='auto',
    )
    callbacks.append(early_stopping)

  if find_lr:
    lr_find = LR_Find(len(x_train) / batch_size)
    callbacks.append(lr_find)

  if reduce_lr_on_plateau:
    lr_plateau = keras.callbacks.ReduceLROnPlateau()
    callbacks.append(lr_plateau)

  class MyCallback(Callback):

    def __init__(self, model):
      self.epoch = 0
      self.model = model
      self.stat_funcs_by_name = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'std': np.std
      }
      self.stats_by_name = {
        name: [] for name in self.stat_funcs_by_name.keys()
      }

    def on_batch_end(self, batch, logs=None):
      flat_weights = []
      weights = self.model.get_weights()
      for layer_weights in weights:
        flat_weights.append(layer_weights.flatten())
      for stat_name, stat_func in self.stat_funcs_by_name.items():
        stat_val = stat_func(np.concatenate(flat_weights))
        self.stats_by_name[stat_name].append(stat_val)

    # TODO: plot learning curves
    def print_weight_stats(self):
      try:
        for stat_name, stat_vals in self.stats_by_name.items():
          print('*' * 90)
          print('%s:' % stat_name)
          hipplot(stat_vals)
          print('*' * 90)
      except Exception as exc:
        import traceback
        traceback.print_exc()
        import ipdb; ipdb.set_trace()
        foo = 1

  if record_stats:
    mcb = MyCallback(model)
    callbacks.append(mcb)

  print('x_train min: %.5f, max: %.5f, mean: %.5f, std: %.5f' % (
    x_train.min(), x_train.max(), x_train.mean(), x_train.std()))

  '''
  #from bashplotlib.scatterplot import plot_scatter
  from bashplotlib.histogram import plot_hist 
  import io
  plot_hist(io.StringIO('\n'.join(['%s' % _x for _x in x_train.flatten()])))
  '''

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
          featurewise_center=featurewise_center,
          # set each sample mean to 0
          samplewise_center=False,
          # divide inputs by std of the dataset
          featurewise_std_normalization=featurewise_std_normalization,
          # divide each input by its std
          samplewise_std_normalization=False,
          # apply ZCA whitening
          zca_whitening=False,
          # randomly rotate images in the range (degrees, 0 to 180)
          rotation_range=rotation_range,
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

      vals = []
      f = datagen.flow(x_train, y_train, batch_size=batch_size)
      for _ in range(math.ceil(len(x_train) // batch_size)):
        vals.append(next(f))
      xs = np.array([_x for _x, _y in vals])
      print('after ImageDataGenerator, min: %.5f, max: %.5f, mean: %.5f, std: %.5f' % (
        xs.min(), xs.max(), xs.mean(), xs.std()))
        

      # Fit the model on the batches generated by datagen.flow().
      history = model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    workers=4,
                                    callbacks=callbacks)

  if record_stats:
    mcb.print_weight_stats()

  # Save model and weights
  if save_final_model:
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

  # Score trained model.
  scores = model.evaluate(x_test, y_test, verbose=1)
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])

  figs = []

  '''
  figs += plot_grids(
    x_test,
    y_test.argmax(axis=1),
    model.predict(x_test, verbose=1).argmax(axis=1)
  )
  '''

  if record_stats:
    figs.append(plt.figure())
    acc = history.history.get('acc')
    loss = history.history.get('loss')
    val_acc = history.history.get('val_acc')
    val_loss = history.history.get('val_loss')

    if acc or loss or val_acc or val_loss:
      print('acc: %s' % acc)
      hipplot(acc)
      print('loss: %s' % loss)
      hipplot(loss)
      print('val_acc: %s' % val_acc)
      hipplot(val_acc)
      print('val_loss: %s' % val_loss)
      hipplot(val_loss)
      assert len(acc) == len(loss) == len(val_acc) == len(val_loss)
      get_x = lambda y: np.linspace(0, len(y), num=len(y))
      figs.append(plt.figure())
      try:
        handles = [
          plt.plot(get_x(acc), acc, label='Accuracy (acc)'),
          plt.plot(get_x(loss), loss, label='Loss (loss)'),
          plt.plot(get_x(val_acc), val_acc, label='Validation Accuracy (val_acc)'),
          plt.plot(get_x(val_loss), val_loss, label='Validation Loss (val_loss)')
        ]
      except Exception as exc:
        import ipdb; ipdb.set_trace()
      plt.legend()

  if find_lr:
    figs.append(plt.figure())
    lr_find.plot()
    plt.title('loss vs. learning rate (log)')

    figs.append(plt.figure())
    lr_find.plot_lr()
    plt.title('learning rate vs. iterations')

  # TODO: show plots non-blocking
  # https://stackoverflow.com/questions/28269157/
  if show_plots:
    plt.show()

  if save_plots:
    import inspect
    import hashlib
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    params = [(i, values[i]) for i in args]
    params.sort(key=lambda tup: tup[0])
    digest = hashlib.sha224(str(params).encode('utf-8')).hexdigest()
    # TODO: print params in title
    if not os.path.exists('out'):
      os.mkdir('out')
    for i_fig, fig in enumerate(figs):
      filename = 'out/prospector_%s_%d.png' % (digest, i_fig)
      print('Saving to file %s...' % filename)
      plt.figure(fig.number)
      plt.savefig(filename)
      plt.close(fig)

  return scores[1]

'''
norm_hack = True
truncate_ratio = 0.2

[(0.49324324324324326, {'decay': 1e-06, 'lr': 0.003}),
 (0.5067567567567568, {'decay': 3e-07, 'lr': 0.003}),
 (0.5067567567567568, {'decay': 3e-06, 'lr': 0.003}),
 (0.5540540540540541, {'decay': 3e-06, 'lr': 3e-05}),
 (0.5743243243243243, {'decay': 3e-07, 'lr': 3e-05}),
 (0.595945945945946, {'decay': 1e-06, 'lr': 3e-05}),
 (0.6324324324324324, {'decay': 3e-06, 'lr': 0.0001}),
 (0.6337837837837837, {'decay': 1e-06, 'lr': 0.0001}),
 (0.6567567567567567, {'decay': 3e-07, 'lr': 0.0001})]

[(0.5324324317880579,
  {'decay': 0,
   'epochs': 10,
   'featurewise_std_normalization': True,
   'find_lr': False,
   'log': False,
   'lr': 1e-05,
   'norm_bounds': None,
   'norm_hack': False,
   'stop_early': True,
   'truncate_ratio': 0.5}),
 (0.5405405415071024,
  {'decay': 0,
   'epochs': 10,
   'featurewise_std_normalization': True,
   'find_lr': False,
   'log': False,
   'lr': 1e-05,
   'norm_bounds': None,
   'norm_hack': True,
   'stop_early': True,
   'truncate_ratio': 0.5})]
'''
'''
  {'decay': 3e-07,
   'epochs': 10,
   'featurewise_std_normalization': False,
   'find_lr': False,
   'log': False,
   'lr': 0.0003,
   'norm_bounds': (-14.0852577255, 144.4015012157),
   'norm_hack': False,
   'stop_early': True,
   'truncate_ratio': None})]

  {'decay': 3e-07,
   'epochs': 10,
   'featurewise_std_normalization': True,
   'find_lr': False,
   'log': False,
   'lr': 0.0003,
   'norm_bounds': (-14.0852577255, 144.4015012157),
   'norm_hack': False,
   'stop_early': True,
   'truncate_ratio': None})
'''

def hyperopt():
  from itertools import product

  def param_sets():
    param_grid = [
      {
        'log': [False],
        'truncate_ratio': [1],
        'find_lr': [True, False],
        'stop_early': [False],
        'epochs': [20],
        'lr': [1e-4, 3e-4, 1e-3],
        'decay': [1e-7, 3e-7, 1e-6],
        'norm_hack': [False],
        'double_layers': [False, True],
        'norm_bounds': [
          (-14.07, 144.3),
          (-100, 100),
        ],
        'featurewise_std_normalization': [False, True],
        'rotation_range': [0, 180],
        'batch_norm': [False, True],
        'reduce_lr_on_plateau': [True],
        'data_augmentation': [True],
        'record_stats': [False]
      }
    ]
    pprint(param_grid)
    for p in param_grid:
      items = sorted(p.items())
      if not items:
        yield {}
      else:
        keys, values = zip(*items)
        for v in product(*values):
          params = dict(zip(keys, v))
          yield params
  results = []
  for param_set in param_sets():
    if param_set:
      print('param_set:\n%s' % pformat(param_set))
      score = run(**param_set)
      #score = np.random.random()
      results.append((score, param_set))
    else:
      print('param set was None')
  results.sort(key=lambda tup: tup[0])
  print('results:')
  pprint(results)

  # TODO XXX pickle instead

  '''
  results_fname = 'hyperparam-scores.txt'
  with open(results_fname
  with open(results_fname, 'a+') as f:
    f.seek(-1, os.SEEK_END)
    filehandle.truncate()
    f.write(',\n' + pformat(results)[1:])
  print('results written to %s' % results_fname)
  '''

  import ipdb; ipdb.set_trace()

if __name__ == '__main__':
  hyperopt()
