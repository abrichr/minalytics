from __future__ import print_function

'''
This script performs a regression in columns imported from an excel workbook.

It performs the following tasks:
  - Import data from first worksheet in a workbook
  - Determine which columns are targets based on their names (first row)
  - Remove data columns that are highly correlated with target columns
  - Determine column types (one of Numerical, Textual, Categorical and
    Multi-categorical)
  - Convert Categorical and Multi-Categorical to One-Hot representations
  - Regress one target column at a time (ignoring other targets) via linear
    regression
    with k-fold cross-validation
  - Print Pearson's Correlation Coefficient and the standard deviation across
    folds for each target
'''

# This ratio is the number of empty values divided by the total number of rows
# in a column. Columns which have too many empty values are ignored.
MAX_EMPTY_RATIO = 0.7

# Any column whose name contains at least one of these words (case-insensitive)
# is treated as a target for the regression.
TARGET_COL_TERMS = ['capitalization', 'value']

# If the Pearson Correlation Coefficient between a data column and any target
# column is greater than this value, it is ignored.
CORR_THRESH = 0.9

# If any value in a column is longer than this many characters, it is treated
# as a text column and ignored.
TEXT_COL_MIN_LEN = 500

# Multi-category columns contain comma-delimited values. This is the minimum
# number of times a comma-delimited value needs to be found in order for that
# column to be treated as a multi-category column.
MIN_DUP_PART_COUNT = 2

# Category columns contain a single value. If the ratio of unique values in a
# column to the # total number of rows is too high, it won't contain any
# predictive value. This is the maximum allowed value before a column is
# ignored. Setting this value aggressively low (e.g. 0.01) reduces the number
# of columns, and thus the time required to complete the regression.
MAX_UNIQUE_VALS_TO_ROWS = 0.01

# Path to Excel workbook
EXCEL_WORKBOOK_PATH = 'snlworkbook_Combined.xls'

# Path to magnetic grid file (text)
MAGNETIC_GRID_PATH = '200 m mag deg.dat'

# Path to augmented data file
AUGMENTED_DATA_PATH = 'augmented_data.pkl'

# Path to non-augmented data file
DATA_PATH = 'data.pkl'

# Size of patch of magnetic data in metres
MAG_PATCH_SIZE_M = 10000

# Size of magnetic data stride in metres
MAG_STRIDE_SIZE_M = 5000

# Set to True for verbose logging
DEBUG = False


import logging
#log_format = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
log_format = '%(asctime)s : %(message)s'
logging.basicConfig(format=log_format, level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

from cachier import cachier
from collections import Counter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint, pformat
from sklearn import preprocessing, metrics, linear_model, model_selection
from sklearn.pipeline import Pipeline
from xlrd import open_workbook
import argparse
import cPickle as pickle
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

@cachier()
def load_dataframe():
  data_fname = EXCEL_WORKBOOK_PATH
  wb = open_workbook(data_fname)

  sheet = wb.sheets()[0]
  col_names = []
  values = []
  for row in range(sheet.nrows):
    row_values = []
    set_col_names = not bool(col_names)
    for col in range(sheet.ncols):
      value = sheet.cell(row, col).value
      if set_col_names:
        # handle duplicate col names
        while value in col_names:
          value += '_'
        col_names.append(value)
      else:
        row_values.append(value)
    if not set_col_names:
      values.append(row_values)

  df = pd.DataFrame(data=values, columns=col_names)
  empty_vals = ['', 'NA', 'NM']
  df.replace(empty_vals, [None for _ in empty_vals], inplace=True) 

  return df

def truncate(s, max_len=80):
  if len(s) <= max_len:
    return s
  return '%s...' % s[:max_len]

class ColType(object):
  EMPTY = 'EMPTY'
  NUM = 'NUMERIC'
  TEXT = 'TEXT'
  CAT = 'CATEGORICAL'
  MULTI = 'MULTICATEGORICAL'

def parse_cols(df):

  empty_cols = []
  num_cols = []
  text_cols = []
  cat_cols = []
  multi_cat_cols = []
  dup_part_counts_by_col = {}

  for i_col, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
    vals = df[col]
    logger.info('Col %s: [%s]' % (i_col, col))
    if dtype == np.float64:
      logger.info('\t==> Numerical')
      num_cols.append(col)
    elif dtype == np.object:
      nonempty_vals = [unicode(v) for v in vals if v not in ['', None]]
      unique_vals = list(set(nonempty_vals))
      if len(unique_vals) <= 1:
        empty_cols.append(col)
        continue

      logger.debug('first 4 unique_vals: %s' % '\n\t'.join(
        list([''] + [truncate(s) for s in unique_vals])[:5]))

      logger.debug('len(unique_vals): %s, len(nonempty_vals): %s' % (
        len(unique_vals), len(nonempty_vals)))
      max_unique_val_len = max(len(v) for v in unique_vals)
      longest_unique_val = [v for v in unique_vals
                            if len(v) == max_unique_val_len][0]
      logger.debug('max_unique_val_len: %s, longest_unique_val: %s' % (
        max_unique_val_len, longest_unique_val))
      if (len(unique_vals) == len(nonempty_vals) or \
          max_unique_val_len > TEXT_COL_MIN_LEN):
        cols_target = text_cols
      else:
        part_counts = Counter()
        for u in unique_vals:

          # remove non-delimiting commas
          replacements = [
            (', Ltd', ' Ltd'),
            (', Inc', ' Inc'),
            (', LLC', ' LLC'),
            (', L.L.C', ' L.L.C'),
            (', S.A. de C.V.', ' S.A. de C.V.'),
            (', S.A.B. de C.V.', ' S.A.B. de C.V.'),
            (', L.P.', ' L.P.'),
            (', S.A.', ' S.A.'),
          ]
          for frm, to in replacements:
            u = u.replace(frm, to)

          # remove thousands separator commas
          new_u = ''
          for i in range(len(u)):
            if (i > 0 and i < len(u) - 1
                and u[i] == ','
                and u[i-1].isdigit()
                and u[i+1].isdigit()):
              continue
            new_u += u[i]
          if u != new_u:
            logger.debug('\t%s  ==>  %s' % (u, new_u))
          u = new_u

          parts = [p.strip() for p in u.split(',')]
          part_counts.update(Counter(parts))

        dup_part_counts = {part: count for part, count in part_counts.items()
                           if count > MIN_DUP_PART_COUNT}
        dup_part_counts_by_col[col] = dup_part_counts
        logger.debug('dup_part_counts:\n%s' % pformat(dup_part_counts))
        if dup_part_counts:
          cols_target = multi_cat_cols
        else:
          cols_target = cat_cols
        
      val_counts = Counter(nonempty_vals)
      dup_val_counts = {val: count for val, count in val_counts.items()
                        if count > 1}
      logger.debug('dup_val_counts:\n%s' % pformat(dup_val_counts))

      if cols_target is cat_cols:
        logger.info('\t==> Categorical')
      elif cols_target is text_cols:
        logger.info('\t==> Textual')
      else:
        logger.info('\t==> Multi-Categorical')
      cols_target.append(col)

      logger.debug('*' * 40)

    else:
      raise 'Unknown column type'

  def truncate_keys(d):
    rval = {}
    for key, val in d.items():
      rval[truncate(key)] = val
    return rval

  logger.debug('num_cols:\n%s' % pformat(num_cols))
  logger.debug('text_cols:\n%s' % pformat(text_cols))
  logger.debug('cat_cols:\n%s' % pformat({
    col: truncate_keys(dup_part_counts_by_col[col]) for col in cat_cols}))
  logger.debug('multi_cat_cols:\n%s' % pformat({
    col: truncate_keys(dup_part_counts_by_col[col]) for col in multi_cat_cols}))

  return {
    ColType.NUM: num_cols,
    ColType.TEXT: text_cols,
    ColType.CAT: cat_cols,
    ColType.MULTI: multi_cat_cols,
    ColType.EMPTY: empty_cols
  }

  #return num_cols, text_cols, cat_cols, multi_cat_cols, empty_cols

def remove_empty_cols(df, max_empty_ratio=MAX_EMPTY_RATIO):
  col_ratios = []
  for col in df.columns:
    vals = df[col]
    num_vals = len(vals)
    if any(pd.notnull(vals)):
      nonempty = vals[pd.notnull(vals)]
    else:
      nonempty = []
    num_nonempty = len(nonempty)
    ratio = 1 - 1.0 * num_nonempty / num_vals
    col_ratios.append((ratio, col))

  col_ratios.sort(key=lambda x: x[0], reverse=True)
  logger.debug('col_ratios: %s' % pformat(col_ratios))

  cols_to_remove = set()
  for ratio, col in col_ratios:
    if ratio > max_empty_ratio:
      logger.info('\tCol is %.2f%% empty, removing: %s' % (ratio*100, col))
      cols_to_remove.add(col)

  remove_columns(df, cols_to_remove)

def remove_columns(df, cols):
  for col in cols:
    del df[col]

def convert_multi_to_onehot(df, cols_by_type):
  new_cols = []
  prev_num_cols = len(df.columns)
  num_added = 0
  for col in cols_by_type[ColType.MULTI]:
    logger.info('converting multi: %s' % col)
    vals = df[col]
    part_counts = Counter()
    for val in vals:
      if val:
        part_counts.update(Counter(p.strip() for p in val.split(',')))
    min_count = 1
    num_to_add = len([_ for _, count in part_counts.items()
                      if count > min_count])
    part_counts_to_add = {p: c for p, c in part_counts.items() if c > min_count}
    for i, (part, count) in enumerate(part_counts_to_add.items()):
      col_name = '%s_%s' % (col, part)
      new_cols.append(col_name)
      logger.debug('\tAdding column %d of %d: %s' % (i, num_to_add, col_name))
      num_added += 1
      df[col_name] = df[col].map(lambda x: 1 if x and part in x else 0)
    num_added -= 1
    del df[col]
  assert len(df.columns) == prev_num_cols + num_added, (
      len(df.columns), prev_num_cols, num_added)
  cols_by_type[ColType.MULTI].extend(new_cols)

def convert_cat_to_onehot(df, cols_by_type,
                          max_unique_vals_to_rows=MAX_UNIQUE_VALS_TO_ROWS):
  num_cols_to_add = 0
  #unique_vals_by_col = {}
  for col in sorted(cols_by_type[ColType.CAT]):
    vals = df[col]
    unique_vals = set(vals) - set([None])
    #unique_vals_by_col[col] = unique_vals
    unique_vals_to_rows = 1.0 * len(unique_vals) / df.shape[0]
    if 1.0 * len(unique_vals) / df.shape[0] > max_unique_vals_to_rows:
      logger.debug(('\tRemoving column, unique_vals_to_rows: %.4f, col: %s, '
                    'len(unique_vals): %s') % (
                  unique_vals_to_rows, col, len(unique_vals)))
      cols_by_type[ColType.CAT].remove(col)
      del df[col]
    else:
      num_cols_to_add += len(unique_vals) - 1
      cols_before = set(df.columns.values)
      logger.debug('\tlen(cols_before): %s' % len(cols_before))
      df = pd.get_dummies(df, columns=[col])
      cols_after = set(df.columns.values)
      logger.debug('\tlen(cols_after): %s' % len(cols_after))
      new_cols = cols_after - cols_before
      logger.debug('\tlen(new_cols): %s' % len(new_cols))
      assert len(new_cols) == len(unique_vals), (
          len(new_cols), len(unique_vals))
  return df

def remove_target_correlations(df, corr_thresh=CORR_THRESH):
  corrs = df.corr().abs()
  cols = corrs.columns.values.tolist()
  corr_col_tups = []
  for row in corrs.index:
    for col in corrs.columns:
      if row == col:
        continue
      corr = corrs.loc[row][col]
      corr_col_tups.append((corr, row, col))
  corr_col_tups.sort(key=lambda x: x[0], reverse=True)
  logger.debug('corr_col_tups:\n%s' % pformat(corr_col_tups))
  cols_to_remove = set()
  for corr, col1, col2 in corr_col_tups:
    if corr < corr_thresh:
      continue
    col1_is_target = any([word in col1.lower() for word in TARGET_COL_TERMS])
    col2_is_target = any([word in col2.lower() for word in TARGET_COL_TERMS])
    # ignore correlations between target columns
    if col1_is_target and col2_is_target:
      continue
    # ignore correlations between data columns
    # TODO: remove data columns with perfect correlation
    if not (col1_is_target or col2_is_target):
      continue

    col_to_remove = col1 if col2_is_target else col2 
    if col_to_remove in cols_to_remove:
      continue

    # display correlated values
    vals1 = df[col1].values
    vals2 = df[col2].values
    this_df = pd.DataFrame(data={col1: vals1, col2: vals2})
    logger.info(('Dropping data column with suspiciously large correlation '
      'with target column, corr=%.4f\n%s') % (corr, this_df.dropna()[:10]))

    cols_to_remove.add(col_to_remove)

  for col in cols_to_remove:
    del df[col]

def get_target_cols(cols_by_type):
  target_cols = [col for col in cols_by_type[ColType.NUM]
                 if any(word in col.lower() for word in TARGET_COL_TERMS)]
  logger.info('target_cols:\n\t%s' % '\n\t'.join(target_cols))
  return target_cols

def get_data(from_disk=True):
  if from_disk and os.path.exists(DATA_PATH):
    logger.info('reading %s...' % DATA_PATH)
    try:
      f = open(DATA_PATH)
      df, cols_by_type, target_cols = pickle.load(f)
      return df, cols_by_type, target_cols
    except Exception as e:
      logger.info('e: %s' % e)
      logger.info('recreating...')

  logger.info('Preparing data...')
  start_time = time.time()

  df = load_dataframe()
  remove_empty_cols(df)

  remove_target_correlations(df)

  cols_by_type = parse_cols(df)

  target_cols = get_target_cols(cols_by_type)

  logger.info('Converting multis...')
  convert_multi_to_onehot(df, cols_by_type)

  logger.info('Converting categories...')
  df = convert_cat_to_onehot(df, cols_by_type)

  logger.info('Removing text columns...')
  remove_columns(df, cols_by_type[ColType.TEXT])

  for col, dtype in zip(df.columns, df.dtypes):
    logger.debug('dtype: %s, col: %s' % (dtype, col))

  logger.info('Data preparation took: %.2fs' % (time.time() - start_time))

  logger.info('saving to %s' % DATA_PATH)
  f = open(DATA_PATH, 'w')
  pickle.dump([df, cols_by_type, target_cols], f)

  return df, cols_by_type, target_cols

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
regs = [
    #SVR(kernel='rbf', C=1e3, gamma=0.1),  # slow and innaccurate
    RandomForestRegressor()
]

def regress(df, target_cols, data_cols=None):
  logger.info('regressing, df.shape: %s' % str(df.shape))
  for target_col in target_cols:
    logger.info('target_col: %s' % target_col)
    this_df = df[pd.notnull(df[target_col].values)]
    this_df = this_df.fillna(0)
    logger.debug('this_df.shape: %s' % str(this_df.shape))
    y = this_df[target_col].values
    for _target_col in target_cols:
        del this_df[_target_col]
    if data_cols:
      for col in this_df.columns:
        if col not in data_cols:
          del this_df[col]
    X = this_df.values
    logger.info('\tX.shape: %s, y.shape: %s' % ((X.shape, y.shape)))

    for reg in regs:
      logger.info('\tFitting reg: %s...' % reg.__class__.__name__)
      scaler = preprocessing.StandardScaler()
      pipeline = Pipeline([('transformer', scaler), ('estimator', reg)])
      cv = model_selection.KFold(n_splits=5)
      scores = model_selection.cross_val_score(pipeline, X, y, cv=cv)
      score, std = scores.mean(), scores.std()
      logger.info('\t\tR2: %.4f, std: %.4f' % (score, std))

def do_simple_regression():
  df, cols_by_type, target_cols = get_data()

  rows, cols = df.shape
  logger.info('rows: %s, cols: %s' % (rows, cols))

  regress(df, target_cols)

def read_magnetic_data(
    magnetic_grid_path=MAGNETIC_GRID_PATH
):
  # N: positive latitude
  # S: negative latitude
  # E: positive longitude
  # W: negative longitude

  logger.info('reading grid file: %s' % magnetic_grid_path)
  start_time = time.time()
  df = pd.read_csv(
    magnetic_grid_path,
    sep=',',
    header=None,
    names='y x mag lon lat'.split(),
    dtype={
      'y': np.int64,
      'x': np.int64,
      'mag': np.float64,
      'lat': np.float64,
      'lon': np.float64
    }
  )
  logger.info('done, took: %.2fs' % (time.time() - start_time))

  if 0:
    logger.info('setting index...')
    df = df.set_index(['lat', 'lon'])

  if 0:
    logger.info('writing to outfile...')
    df.to_csv(magnetic_grid_path + '.degonly', header=False, index=False)
    logger.info('done')

  return df

def get_lat_lon_cols(cols):
  lat_cols = [c for c in cols if 'latitude' in c.lower()]
  lon_cols = [c for c in cols if 'longitude' in c.lower()]
  logger.info('lat_cols: %s' % lat_cols)
  logger.info('lon_cols: %s' % lon_cols)
  return lat_cols[0], lon_cols[0]

def time_func(func, name):
  start_time = time.time()
  rval = func()
  duration = time.time() - start_time
  logger.debug('func: %s, duration: %.2f' % (name, duration))
  return rval

def extract_grid(mag_df, cx, cy, box_size_m=MAG_PATCH_SIZE_M):
  half_size = box_size_m / 2.0
  x_mask = time_func(lambda: abs(mag_df.x - cx) < half_size,
      'abs(mag_df.x - cx) < half_size')
  y_mask = time_func(lambda: abs(mag_df.y - cy) < half_size,
      'abs(mag_df.y - cy) < half_size')
  mask = time_func(lambda: x_mask & y_mask,
      'x_mask & y_mask')
  region = time_func(lambda: mag_df[mask],
      'mag_df[mask]')
  n_cols = time_func(lambda: len(region.x.unique()),
      'len(region.x.unique())')
  n_rows = time_func(lambda: len(region.y.unique()),
      'len(region.y.unique())')
  grid = time_func(lambda: region.mag.values.reshape((n_rows, n_cols)),
      'region.mag.values.reshape((n_rows, n_cols))')

  # TOOD: remove
  #grid2 = region.pivot(index='x', columns='y', values='mag').values
  #assert np.allclose(grid, grid2)

  return grid

def get_mag_region(mag_df, row, lat_col, lon_col, box_size_m=MAG_PATCH_SIZE_M):
  try:
    d_lat = abs(mag_df.lat - row[lat_col])
    d_lon = abs(mag_df.lon - row[lon_col])
    d_tot = d_lat + d_lon
    argmin = d_tot.idxmin()
    row = mag_df.loc[argmin]
    cx = row.x
    cy = row.y
    grid = extract_grid(mag_df, cx, cy)
  except Exception as e:
    logger.warn('e: %s' % e)
    import ipdb; ipdb.set_trace()

  logger.info('grid.shape: %s' % str(grid.shape))
  return grid

def get_full_data(from_disk=True, max_rows=None):
  if from_disk and os.path.exists(AUGMENTED_DATA_PATH):
    logger.info('reading %s...' % AUGMENTED_DATA_PATH)
    try:
      f = open(AUGMENTED_DATA_PATH)
      grids, df, cols_by_type, target_cols = pickle.load(f)
      return grids, df, cols_by_type, target_cols
    except Exception as e:
      logger.info('e: %s' % e)
      logger.info('recreating...')

  mag_df = read_magnetic_data()

  df, cols_by_type, target_cols = get_data()
  lat_col, lon_col = get_lat_lon_cols(df.columns)

  min_lat = min(mag_df.lat)
  max_lat = max(mag_df.lat)
  min_lon = min(mag_df.lon)
  max_lon = max(mag_df.lon)

  logger.debug('min_lat: %s' % min_lat)
  logger.debug('max_lat: %s' % max_lat)
  logger.debug('min_lon: %s' % min_lon)
  logger.debug('max_lon: %s' % max_lon)

  try:
    logger.info('removing rows without lat/lon...')
    df = df[pd.notnull(df[lat_col]) &
            pd.notnull(df[lon_col])]
    logger.info('removing out of bounds...')
    df = df[(df[lat_col] >= min_lat) &
            (df[lat_col] <= max_lat) &
            (df[lon_col] >= min_lon) &
            (df[lon_col] <= max_lon)]
    logger.info('resetting index...')
    df = df.reset_index()
  except Exception as e:
    logger.warn('e:', e)
    import ipdb; ipdb.set_trace()

  #df['mag'] = None  # set dtype to Object
  grids = []
  for i, row in df.iterrows():
    if max_rows and i > max_rows:
      break
    logger.info('extracting magnetic region %d of %d' % (i, df.shape[0]))
    try:
      mag_region = get_mag_region(mag_df, row, lat_col, lon_col)
      #df.at[0, 'mag'] = mag_region
      grids.append(mag_region)
    except Exception as e:
      logger.warn('e:', e)
      import ipdb; ipdb.set_trace()

  logger.info('saving to %s' % AUGMENTED_DATA_PATH)
  f = open(AUGMENTED_DATA_PATH, 'w')
  pickle.dump([grids, df, cols_by_type, target_cols], f)
  #df.to_csv(AUGMENTED_DATA_PATH, encoding='utf-8')

  return grids, df, cols_by_type, target_cols

def add_magnetic_features(df, grids):
  df['mag_min'] = [grid.min() for grid in grids]
  df['mag_max'] = [grid.max() for grid in grids]
  df['mag_std'] = [grid.std() for grid in grids]

  # http://www.cspg.org/cspg/documents/Conventions/Archives/Annual/2011/176-Texture_Analysis.pdf
  # https://www.sciencedirect.com/science/article/pii/S2090997717300937
  # https://prism.ucalgary.ca/bitstream/handle/1880/51900/texture%20tutorial%20v%203_0%20180206.pdf?sequence=11&isAllowed=y

  # https://www.orfeo-toolbox.org/features-2/

  from skimage.feature import greycomatrix, greycoprops
  from skimage.measure import shannon_entropy
  from sklearn.decomposition import PCA
  feats = {name: [] for name in [
    'correlation',
    'variance',
    'contrast',
    'entropy',
    'homogeneity',
    'asm',
    #'pca'
  ]}
  #mn = min(g.min() for g in grids)
  #mx = max(g.max() for g in grids)
  for grid in grids:
    print('grid.shape: %s' % str(grid.shape))
    mn = grid.min()
    mx = grid.max()
    grid = ((grid - mn) / (mx - mn) * 256).astype('uint8')
    glcm = greycomatrix(grid, [5], [0], symmetric=True, normed=True)
    correlation = greycoprops(glcm, 'correlation')[0][0]
    variance = grid.var()
    contrast = greycoprops(glcm, 'contrast')[0][0]
    entropy = shannon_entropy(grid)
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    asm = greycoprops(glcm, 'ASM')[0][0]
    # TODO: take components which explain 85% variance 
    #pca = PCA().fit_transform(grid)

    #import ipdb; ipdb.set_trace()

    feats['correlation'].append(correlation)
    feats['variance'].append(variance)
    feats['contrast'].append(contrast)
    feats['entropy'].append(entropy)
    feats['homogeneity'].append(homogeneity)
    feats['asm'].append(asm)
    #feats['pca'].append(pca)

  for name, vals in feats.iteritems():
    logger.info('name: %s val: %s' % (name, vals[0]))
    df[name] = vals
  #import ipdb; ipdb.set_trace()

  return feats.keys() + ['mag_min', 'mag_max', 'mag_std']

def do_magnetic_regression():
  grids, df, cols_by_type, target_cols = get_full_data(from_disk=False)
  mag_feat_names = add_magnetic_features(df, grids)
  regress(df, target_cols, mag_feat_names)


def do_unsupervised_learning():
  mag_df = read_magnetic_data()

  min_y = min(mag_df.y)
  max_y = max(mag_df.y)
  min_x = min(mag_df.x)
  max_x = max(mag_df.x)

  # loop over top left corners
  X = []
  half_patch_size = MAG_PATCH_SIZE_M / 2
  x_range = range(min_x, max_x - MAG_PATCH_SIZE_M, MAG_STRIDE_SIZE_M)
  y_range = range(min_y, max_y - MAG_PATCH_SIZE_M, MAG_STRIDE_SIZE_M)
  nx = len(x_range)
  ny = len(y_range)
  n_total = nx * ny
  n = 0.0
  for ix, x in enumerate(x_range):
    for iy, y in enumerate(y_range):
      cx = x + half_patch_size
      cy = y + half_patch_size
      grid = extract_grid(mag_df, cx, cy)
      pct = n / n_total * 100
      logger.info('%d of %d, %d of %d: %s (%.4f%%)' % (ix, nx, iy, ny, str(grid.shape), pct))
      X.append(grid)
      n += 1.0

  logger.info("X.shape: %s" % str(np.array(X).shape))

  import ipdb; ipdb.set_trace()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-s',
      help='Do simple regression',
      action='store_true')
  parser.add_argument(
      '-m',
      help='Use magnetic data',
      action='store_true')
  parser.add_argument(
      '-u',
      help='Unsupervised feature learning',
      action='store_true')
  args = parser.parse_args()

  if args.s:
    do_simple_regression()

  if args.m:
    do_magnetic_regression()

  if args.u:
    do_unsupervised_learning()


if __name__ == '__main__':
  main()
