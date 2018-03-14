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

TODO:
  model:
    - blob detection
    - regression using blob features
    - autoencoder on log/blobs 
    - autoencoder grid search over hyperparameters
  code:
    - replace all manual caching
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

# Size of patch of magnetic data in metres
MAG_PATCH_SIZE_M = 10000

# Size of magnetic data stride in metres
MAG_STRIDE_SIZE_M = 5000

# Number of PCA components to use as features
NUM_PCA_COMPONENTS = 10

# Set to True for verbose logging
DEBUG = True

import logging
#log_format = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
log_format = '%(asctime)s : %(message)s'
logging.basicConfig(format=log_format, level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

from collections import Counter
from functools import partial
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint, pformat
from sklearn import preprocessing, metrics, linear_model, model_selection
from sklearn.decomposition import PCA, KernelPCA
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sklearn.pipeline import Pipeline
from xlrd import open_workbook
import argparse
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import percache
import time

cache = percache.Cache(".cache", livesync=True)

@cache
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

PY3 = False
try:
  unicode('')
except NameError:
  PY3 = True

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
      nonempty_vals = [v if PY3 else unicode(v) for v in vals if v not in ['', None]]
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

@cache
def get_data():
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

  return df, cols_by_type, target_cols

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
regs = [
    #SVR(kernel='rbf', C=1e3, gamma=0.1),  # slow and innaccurate
    RandomForestRegressor()
]

def regress(df, target_cols, data_cols=None):
  data_cols = data_cols or []
  logger.info('regressing, df.shape: %s, target_cols:\n%s, data_cols:\n%s' % (
    str(df.shape), pformat(sorted(target_cols)), pformat(sorted(data_cols))))
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

@cache
def read_magnetic_data(
    magnetic_grid_path=MAGNETIC_GRID_PATH,
    stride=None
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

  if stride is not None:
    logger.info('striding: %s' % stride)
    logger.info('before df.shape: %s' % str(df.shape))
    x = np.array(list(set(df.x))[::stride])
    y = np.array(list(set(df.y))[::stride])
    df = df[df.x.isin(x) & df.y.isin(y)]
    logger.info('after df.shape: %s' % str(df.shape))

  return df

def get_lat_lon_cols(cols):
  lat_cols = [c for c in cols if 'latitude' in c.lower()]
  lon_cols = [c for c in cols if 'longitude' in c.lower()]
  logger.info('lat_cols: %s' % lat_cols)
  logger.info('lon_cols: %s' % lon_cols)
  return lat_cols[0], lon_cols[0]

def time_func(func, name, verbose=False):
  print_func = logger.info if verbose else logger.debug
  start_time = time.time()
  print_func('timing func: %s...' % name)
  rval = func()
  duration = time.time() - start_time
  print_func('func: %s, duration: %.2f' % (name, duration))
  return rval

def get_mag_region(mag_df, pivot, row, lat_col, lon_col, box_size_m=MAG_PATCH_SIZE_M):
  try:
    d_lat = abs(mag_df.lat - row[lat_col])
    d_lon = abs(mag_df.lon - row[lon_col])
    d_tot = d_lat + d_lon
    argmin = d_tot.idxmin()
    row = mag_df.loc[argmin]
    cx = row.x
    cy = row.y
    grid = extract_grid_from_pivot(pivot, cx, cy)
  except Exception as e:
    logger.warn('e: %s' % e)
    import ipdb; ipdb.set_trace()

  logger.info('grid.shape: %s' % str(grid.shape))
  return grid

@cache
def get_full_data(max_rows=None):
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
  pivot = get_pivot(mag_df)
  for i, row in df.iterrows():
    if max_rows and i > max_rows:
      break
    logger.info('extracting magnetic region %d of %d' % (i, df.shape[0]))
    try:
      mag_region = get_mag_region(mag_df, pivot, row, lat_col, lon_col)
      #df.at[0, 'mag'] = mag_region
      grids.append(mag_region)
    except Exception as e:
      logger.warn('e:', e)
      import ipdb; ipdb.set_trace()

  return grids, df, cols_by_type, target_cols

@cache
def get_magnetic_features(df, grids, num_pca_components=NUM_PCA_COMPONENTS):
  feats = {name: [] for name in [
    'min',
    'max',
    'std',
    'sum',

    # http://www.cspg.org/cspg/documents/Conventions/Archives/Annual/2011/176-Texture_Analysis.pdf
    # https://www.sciencedirect.com/science/article/pii/S2090997717300937
    # https://prism.ucalgary.ca/bitstream/handle/1880/51900/texture%20tutorial%20v%203_0%20180206.pdf?sequence=11&isAllowed=y
    # https://www.orfeo-toolbox.org/features-2/
    'correlation',
    'variance',
    'contrast',
    'entropy',
    'homogeneity',
    'asm',
  ]}
  for i in range(num_pca_components):
    feats['pca_%d' % i] = []
    feats['kpca_%d' % i] = []

  pca = get_magnetic_pca()
  kpca = get_magnetic_kpca()

  logger.info('transforming %d grids of shape %s...' % (len(grids), str(grids[0].shape)))
  for i_grid, grid in enumerate(grids):
    pct_done = int(1.0 * i_grid / len(grids) * 100)
    if i_grid and i_grid % 100 == 0:
      logger.info('%d%% done' % pct_done)
    mn = grid.min()
    mx = grid.max()
    std = grid.std()
    grid = ((grid - mn) / (mx - mn) * 256).astype('uint8')
    glcm = greycomatrix(grid, [5], [0], symmetric=True, normed=True)
    correlation = greycoprops(glcm, 'correlation')[0][0]
    variance = grid.var()
    contrast = greycoprops(glcm, 'contrast')[0][0]
    entropy = shannon_entropy(grid)
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    asm = greycoprops(glcm, 'ASM')[0][0]
    _sum = grid.sum()

    try:
      grid_pca = pca.transform([grid.flatten()])[0]
      grid_kpca = kpca.transform([grid.flatten()])[0]
      for i in range(num_pca_components):
        feats['pca_%d' % i].append(grid_pca[i])
        feats['kpca_%d' % i].append(grid_kpca[i])
    except ValueError as e:
      for i in range(num_pca_components):
        feats['pca_%d' % i].append(0)
        feats['kpca_%d' % i].append(0)

    feats['min'].append(mn)
    feats['max'].append(mx)
    feats['std'].append(std)
    feats['correlation'].append(correlation)
    feats['variance'].append(variance)
    feats['contrast'].append(contrast)
    feats['entropy'].append(entropy)
    feats['homogeneity'].append(homogeneity)
    feats['asm'].append(asm)
    feats['sum'].append(_sum)

  return feats

def do_magnetic_regression():
  grids, df, cols_by_type, target_cols = get_full_data()
  feats = get_magnetic_features(df, grids)
  for name, vals in feats.items():
    logger.info('name: %s val: %s' % (name, vals[0]))
    df[name] = vals
  mag_feat_names = feats.keys()
  regress(df, target_cols, mag_feat_names)

@cache
def get_pivot(mag_df):
  logger.info('Pivoting...')
  pivot = mag_df.pivot(index='x', columns='y', values='mag')
  grid = pivot.values
  vals = grid.flatten()
  vals.sort()
  idxs = [i for i, v in enumerate(vals) if v != vals[0]]
  min_idx = idxs[0]
  clip_min = vals[min_idx]
  pivot = pivot.clip(clip_min)
  return pivot

def display_full_magnetic_grid(save=True, show=True, stride=5):
  if not (save or show):
    return

  dpi = [np.inf]
  def _display(f, do_log, title, pos):
    mag_df = read_magnetic_data(stride=stride)
    pivot = get_pivot(mag_df)
    x = pivot.columns
    y = pivot.index
    Z = pivot.values
    if do_log:
      Z = np.log(Z)
    ax = plt.subplot(2,2,pos)
    f(Z, x, y)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)
    dpi[0] = min(dpi[0], min(Z.shape))
    logger.info('dpi: %s' % dpi)

  def _intensity(Z, _, __):
    mn = Z.min()
    mx = Z.max()
    logger.info('mn: %s, mx: %s' % (mn, mx))
    logger.info('Z.shape: %s' % str(Z.shape))
    plt.imshow(Z)

  def _contours(Z, x, y):
    logger.debug('Z.dtype: %s, x.dtype: %s, y.dtype: %s' % (
      Z.dtype, x.dtype, y.dtype))
    logger.debug('Z.shape: %s, x.shape: %s, y.shape: %s' % (
      str(Z.shape), str(x.shape), str(y.shape)))
    logger.info('np.meshgrid()...')
    X, Y = np.meshgrid(x, y)
    nr, nc = Z.shape
    logger.info('plt.contourf()...')
    CS = plt.contourf(X, Y, Z, 10, cmap=plt.cm.bone)#, origin='lower')
    plt.colorbar(CS)

  _display(_intensity, False, 'Magnetic Intensity', 1)
  _display(_intensity, True, 'Magnetic Intensity (log)', 3)
  _display(_contours, False, 'Magnetic Contours', 2)
  _display(_contours, True, 'Magnetic Contours(log)', 4)

  if save:
    filename = 'grid.png'
    logger.info('Saving to file %s...' % filename)
    plt.savefig(filename, dpi=dpi[0])
  if show:
    logger.info('plt.show()...')
    plt.show()

def display_magnetic_grids(save=True, show=False):
  if not (save or show):
    return

  grids, df, cols_by_type, target_cols = get_full_data()
  lat_col, lon_col = get_lat_lon_cols(df.columns)

  mn = min([grid.min() for grid in grids])
  mx = max([grid.max() for grid in grids])
  logger.info('mn: %.2f, mx: %.2f' % (mn, mx))

  for i, grid in enumerate(grids):
    grid_min = grid.min()
    grid_max = grid.max()
    logger.info('grid_min: %s, grid_max: %s' % (grid_min, grid_max))
    row = df.iloc[i]
    lat = row[lat_col]
    lon = row[lon_col]

    ax = plt.subplot(1,2,1)
    plt.imshow(grid)
    ax.set_title('Relative [%.2f, %.2f]' % (grid_min, grid_max))

    ax = plt.subplot(1,2,2)
    plt.imshow(grid, vmin=mn, vmax=mx)
    ax.set_title('Absolute [%.2f, %.2f]' % (mn, mx))

    plt.suptitle('Magnetic Intensities at lat: %s, lon: %s' % (lat, lon))

    if save:
      filename = 'grid_%04d_%s_%s.png' % (i, lat, lon)
      logger.info("Saving to file %s" % filename)
      plt.savefig(filename)
    if show:
      plt.show()

def extract_grid_from_pivot(pivot, cx, cy, box_size_m=MAG_PATCH_SIZE_M):
  half_size = box_size_m / 2.0
  x0 = int(cx - half_size)
  x1 = int(cx + half_size)
  y0 = int(cy - half_size)
  y1 = int(cy + half_size)
  grid = pivot.loc[x0:x1,y0:y1].values
  return grid

@cache
def extract_all_grids(flatten=True):
  mag_df = read_magnetic_data()
  pivot = get_pivot(mag_df)

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
  n = 0
  logger.info('Extracting grids...')
  for ix, x in enumerate(x_range):
    for iy, y in enumerate(y_range):
      cx = x + half_patch_size
      cy = y + half_patch_size
      grid = extract_grid_from_pivot(pivot, cx, cy)
      pct = 1.0 * n / n_total * 100
      if n % 1000 == 0:
        logger.debug('x: %d of %d, y: %d of %d, shape: %s, %d%% complete' % (
          ix, nx, iy, ny, str(grid.shape), int(pct)))
      vals = grid.flatten() if flatten else grid
      X.append(vals)
      n += 1
  X = np.array(X)
  logger.info("X.shape: %s" % str(X.shape))

  return X

@cache
def _do_mag_pca(Klass, pct_rows=None):
  X = extract_all_grids()

  # Standardize
  mean = X.mean(axis=0)
  X -= mean
  std = X.std(axis=0)
  X /= std

  pca = Klass()  # PCA() or KernelPCA()

  if pct_rows is not None:
    n_rows = int(X.shape[0] * pct_rows)
    logger.debug('pct_rows: %s, n_rows: %s' % (pct_rows, n_rows))
    row_idxs = np.random.choice(range(X.shape[0]), size=(n_rows,))
    X_ = []
    for i in row_idxs:
      X_.append(X[i,:])
    X = np.array(X_)
    logger.debug('new X.shape: %s' % str(X.shape))

  # TODO: change name of func depending on Klass
  func_name = Klass.__name__ + '.fit()'
  time_func(partial(pca.fit, X), func_name, verbose=True)

  return pca

def get_magnetic_kpca():
  return _do_mag_pca(partial(KernelPCA, kernel='rbf'),
      pct_rows=0.1
  )

def get_magnetic_pca():
  return _do_mag_pca(PCA)

def _display_pca(pca, name, save=True, show=False, n_components=10):
  # show/save first N principal components
  for i, component in enumerate(pca.components_):
    if i > n_components:
      break
    grid_size = int(np.sqrt(component.shape[0]))
    grid = np.resize(component, (grid_size, grid_size))
    plt.imshow(grid)

    exp_var_ratio = pca.explained_variance_ratio_[i]
    plt.title('%dth principal component\nexplains %.2f variance' % (
      i, exp_var_ratio))

    if show:
      plt.show()
    if save:
      filename = '%s_%d.png' % (name, i)
      logger.info('Saving to %s...' % filename)
      plt.savefig(filename, dpi=min(grid.shape))
  import ipdb; ipdb.set_trace()

def do_pca():
  pca = get_magnetic_pca()
  _display_pca(pca, 'pca')
  import ipdb; ipdb.set_trace()

def do_kpca():
  kpca = get_magnetic_kpca()
  import ipdb; ipdb.set_trace()

def do_autoencoder():
  # https://blog.keras.io/building-autoencoders-in-keras.html

  from keras.callbacks import TensorBoard
  from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
  from keras.models import Model
  from keras import backend as K

  USE_MNIST = False
  if USE_MNIST:
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
  else:
    X = extract_all_grids(flatten=False)  # (48000, 51, 51)
    _, nrows, ncols = X.shape
    nrows = nrows if nrows % 2 == 0 else nrows + 1
    ncols = ncols if ncols % 2 == 0 else ncols + 1
    #X = np.array([grid[:nrows,:ncols] for grid in X])
    X = np.array([np.pad(grid, ((0, 1), (0, 1)), mode='edge') for grid in X])
    shape = (nrows, ncols, 1) # shape = list(X[0].shape) + [1]
    logger.info('shape: %s' % str(shape))  # (50, 50, 1)
    input_img = Input(shape=shape)

  # Conv2D(<num_output_filters>, <kernel_size>)
  x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
  x2 = MaxPooling2D((2, 2), padding='same')(x1)
  x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
  x4 = MaxPooling2D((2, 2), padding='same')(x3)
  x5 = Conv2D(8, (3, 3), activation='relu', padding='same')(x4)
  encoded = MaxPooling2D((2, 2), padding='same')(x5)

  # at this point the representation is (4, 4, 8) i.e. 128-dimensional (MNIST)
  # at this point the representation is (7, 7, 8) i.e. 392-dimensional (MAGNETIC)

  x6 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
  x7 = UpSampling2D((2, 2))(x6)
  x8 = Conv2D(8, (3, 3), activation='relu', padding='same')(x7)
  x9 = UpSampling2D((2, 2))(x8)
  x10 = Conv2D(16, (3, 3), activation='relu')(x9)
  x11 = UpSampling2D((2, 2))(x10)
  decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x11)

  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

  if USE_MNIST:
    from keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    #(60000, 28, 28), (10000, 28, 28)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

  else:

    x_train, x_test = model_selection.train_test_split(X, test_size=0.15)

    x_train -= x_train.min()
    x_train /= x_train.max()
    x_test -= x_test.min()
    x_test /= x_test.max()

    x_train = np.reshape(x_train, [len(x_train)] + list(shape))
    x_test = np.reshape(x_test, [len(x_test)] + list(shape))


  logger.info('Fitting autoencoder...')
  try:
    autoencoder.fit(
        x_train, x_train,
        epochs=50,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
  except Exception as e:
    import ipdb; ipdb.set_trace()

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
      '-p',
      help='Train PCA and save to disk',
      action='store_true')
  parser.add_argument(
      '-k',
      help='Train KernelPCA and save to disk',
      action='store_true')
  parser.add_argument(
      '-d',
      help='Display magnetic grids',
      action='store_true')
  parser.add_argument(
      '-f',
      help='Display full magnetic grid',
      action='store_true')
  parser.add_argument(
      '-a',
      help='Train Autoencoder and save to disk',
      action='store_true')

  args = parser.parse_args()

  if args.s:
    do_simple_regression()

  if args.m:
    do_magnetic_regression()

  if args.p:
    do_pca()

  if args.k:
    do_kpca()

  if args.d:
    display_magnetic_grids()

  if args.f:
    display_full_magnetic_grid()

  if args.a:
    do_autoencoder()

if __name__ == '__main__':
  main()
