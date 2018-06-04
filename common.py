from __future__ import print_function

# .gxf documentation:
# http://www.geosoft.com/media/uploads/resources/technical-notes/gxfr3d9_1.pdf

'''
This script performs a regression in columns imported from an excel workbook,
and the associated aeromagnetic data.

It performs the following tasks:
  - Import data from first worksheet in a workbook
  - Determine which columns are targets based on their names (first row)
  - Remove data columns that are highly correlated with target columns
  - Determine column types (one of Numerical, Textual, Categorical and
    Multi-categorical)
  - Convert Categorical and Multi-Categorical to One-Hot representations
  - Regress one target column at a time (ignoring other targets) via linear
    regression with k-fold cross-validation
  - Print Pearson's Correlation Coefficient and the standard deviation across
    folds for each target
  - TODO: And lots more...

TODO:
  - ANN features -> targets
  - CNN images -> targets
  - additional data dimensions
  - autoencoder grid search over hyperparameters
'''

# This ratio is the number of empty values divided by the total number of rows
# in a column. Columns which have too many empty values are ignored.
MAX_EMPTY_RATIO = 0.85

# Any column whose name contains at least one of these words (case-insensitive)
# is treated as a target for the regression.
#TARGET_COL_TERMS = ['capitalization', 'value']

# csvcut -c 101,102,103,104,105,106 snlworkbook_Combined_fixed.csv | csvstat
TARGET_COL_TERMS = ['reserves', 'resources']
# TODO: ignore 'total deal value (completion)'

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
#EXCEL_WORKBOOK_PATH = 'snlworkbook_Combined.xls'
EXCEL_WORKBOOK_PATH = 'data/snlworkbook_Combined_fixed.xls'

# Path to magnetic grid file (text)
MAGNETIC_GRID_PATH = 'data/200 m mag deg.dat'

# Path to magnetic grid file (binary)
# NAD83
# http://gdr.agg.nrcan.gc.ca/gdrdap/dap/search-eng.php?tree-0=Magnetic-Radiometric-EM+-+Magn%C3%A9tiques-Radioactivit%C3%A9-%C3%89M&tree-1=Compilations+-+Compilations&tree-2=Canada+-+200m+-+MAG&tree-3=Click+here+for+more+options&datatype-ddl=&layer_name=&submit_search=Submit+Search#results
BINARY_MAGNETIC_GRID_PATH = 'data/d950396/Canada - 200m - MAG - Residual Total Field - Composante residuelle.grd.gxf'

# GRS80
# http://gdr.agg.nrcan.gc.ca/gdrdap/pdf/Gravity_Anomaly_description.pdf
BINARY_GRAVITY_GRID_PATH = 'data/d250292/Canada 2 km - GRAV - Gravity Anomalies - Anomalies gravimetriques.grd.gxf'

# Size of patch of magnetic data in metres
MAG_PATCH_SIZE_M = 10000

# Size of magnetic data stride in metres
MAG_STRIDE_SIZE_M = 5000

# Number of PCA components to use as features
NUM_PCA_COMPONENTS = 10

# Set to True for verbose logging
DEBUG = True

# Latitude and longitude of mining sites to ignore (outliers)
IGNORE_LAT_LON_TUPS = [
    (50.4386, -90.5323),
    #(50.47099, -95.54623),  # this one results in no mask feats
    #(48.82997, -94.00014),
    #(48.6333, -90.0666),
    #(54.92991, -98.6328),
    #(51.015, -92.81),
    #(55.70038, -105.0701),
    #(59.49393, -91.46868),
    #(54.81535, -96.23095),
    #(50.4725, -95.0691),
    #(49.79008, -87.81143)
]

# File to store total high magnetic values of mine sites
MINE_MAG_VAL_PATH = 'out/mine-mag-vals.csv'

# Property ID column
ID_COL = 'Property ID'

# Name of column to use to join mines
# (only the one with the largest number of unique values will be used)
OWNER_COLS = [
    'Operator Name',
    'Owner Name Owner 1',
    'Owner Name Owner 1 2016'
]

MAG_ARRAY_PATH = 'out/mag-array.npy'

import logging
log_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

from gdal_grid import GDALGrid

from collections import Counter
from functools import partial
from matplotlib import cm
from pprint import pprint, pformat
from sklearn import preprocessing, metrics, linear_model, model_selection
from sklearn.decomposition import PCA, KernelPCA
from skimage import filters
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy, regionprops
from skimage.morphology import label
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
import sys
import time

from io import IOBase

def myrepr(arg):
  if isinstance(arg, IOBase):
    return "%s:%s" % (arg.name, os.fstat(arg.fileno())[8])
  elif isinstance(arg, dict):
    items = ["%r: %r" % (k, self[k]) for k in sorted(self)]
    return "{%s}" % ", ".join(items)
  else:
    return repr(arg)

try:
  cache = percache.Cache(".cache", livesync=True, repr=myrepr)
except:
  def cache(func):
    def wrapper(*args, **kwargs):
      read_cache = kwargs.get('read_cache', True)
      write_cache = kwargs.get('write_cache', True)
      kwargs.pop('read_cache', None)
      kwargs.pop('write_cache', None)
      return func(*args, **kwargs)
    return wrapper

PY3 = sys.version_info >= (3, 0)
EPS = np.finfo(np.float32).eps

@cache
def load_dataframe(data_fname=EXCEL_WORKBOOK_PATH, max_rows=None):
  logger.info('Reading workbook: %s' % data_fname)
  wb = open_workbook(data_fname)

  sheet = wb.sheets()[0]
  col_names = []
  values = []
  for i_row, row in enumerate(range(sheet.nrows)):
    if max_rows and i_row >= max_rows:
      break
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
                          max_unique_vals_to_rows=MAX_UNIQUE_VALS_TO_ROWS,
                          ignore_cols=None):
  ignore_cols = ignore_cols or []
  num_cols_to_add = 0
  for col in sorted(cols_by_type[ColType.CAT]):
    if col in ignore_cols:
      continue
    vals = df[col]
    unique_vals = set(vals) - set([None])
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

def remove_target_correlations(
    df,
    corr_thresh=CORR_THRESH,
    target_col_terms=TARGET_COL_TERMS
):
  logger.debug('remove_target_correlations() target_col_terms: %s' % target_col_terms)
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
    col1_is_target = any([word in col1.lower() for word in target_col_terms])
    col2_is_target = any([word in col2.lower() for word in target_col_terms])
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

def get_target_cols(cols_by_type, col_terms=TARGET_COL_TERMS):
  target_cols = [col for col in cols_by_type[ColType.NUM]
                 if any(word in col.lower() for word in col_terms)]
  logger.info('get_target_cols() target_cols:\n\t%s' % '\n\t'.join(target_cols))
  return target_cols

#@cache
def get_mine_data(
    keep_owner_cols=False,
    target_col_terms=TARGET_COL_TERMS,
    remove_empty_cols=True,
    remove_target_correlations=True,
    convert_to_onehot=True,
    remove_text_cols=True
):
  logger.info('Preparing data...')

  logger.debug('get_mine_data() target_col_terms: %s' % target_col_terms)
  start_time = time.time()

  df = load_dataframe()

  if remove_empty_cols:
    remove_empty_cols(df)

  if remove_target_correlations:
    remove_target_correlations(df, target_col_terms=target_col_terms)

  cols_by_type = parse_cols(df)

  target_cols = get_target_cols(cols_by_type, target_col_terms)
  logger.debug('target_cols: %s' % target_cols)

  if convert_to_onehot:
    logger.info('Converting multis...')
    convert_multi_to_onehot(df, cols_by_type)
    logger.info('Converting categories...')
    ignore_cols = OWNER_COLS if keep_owner_cols else []
    df = convert_cat_to_onehot(df, cols_by_type, ignore_cols=ignore_cols)

  if remove_text_cols:
    logger.info('Removing text columns...')
    remove_columns(df, cols_by_type[ColType.TEXT])

  for col, dtype in zip(df.columns, df.dtypes):
    logger.debug('dtype: %s, col: %s' % (dtype, col))

  logger.info('Data preparation took: %.2fs' % (time.time() - start_time))

  return df, cols_by_type, target_cols

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

regs = [
    #SVR(kernel='rbf', C=1e3, gamma=0.1),  # slow and innaccurate
    RandomForestRegressor(),
    LinearRegression()
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
      pipeline = Pipeline([
        ('transformer', scaler),
        ('estimator', reg)
      ])
      cv = model_selection.KFold(n_splits=5)
      scores = model_selection.cross_val_score(pipeline, X, y, cv=cv)
      score, std = scores.mean(), scores.std()
      logger.info('\t\tR2: %.4f, std: %.4f' % (score, std))

def do_simple_regression():
  df, cols_by_type, target_cols = get_mine_data()

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
    # TODO XXX: are x and y reversed?
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
    x = np.array(sorted(list(set(df.x)))[::stride])
    y = np.array(sorted(list(set(df.y)))[::stride])
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
    patch = extract_patch_from_pivot(pivot, cx, cy, box_size_m)
  except Exception as e:
    logger.warn('e: %s' % e)
    import ipdb; ipdb.set_trace()

  logger.info('patch.shape: %s' % str(patch.shape))
  return patch

#@cache
def get_patch_from_grid(
    grid,
    lon,
    lat,
    mask_val=None,
    fill_val=0,
    patch_size_m=MAG_PATCH_SIZE_M,
    return_idxs=False,
    square_only=False
):
  logger.debug('get_patch_from_grid() lon: %s, lat: %s, mask_val: %s' % (lon, lat, mask_val))

  if np.isnan(lon) or np.isnan(lat):
    loggder.debug('lon/lat was nan')
    return None

  try:
    cx, cy = grid.lonlat2pixel(lon, lat)
  except IndexError as e:
    logger.debug('No patch at (lon, lat): (%s, %s)' % (lon, lat))
    logger.debug('e: %s' % e)
    return None

  cols = int(patch_size_m / abs(grid.dx))
  rows = int(patch_size_m / abs(grid.dy))
  arr = grid.arr

  y0 = max(cy-rows, 0)
  y1 = min(cy+rows, arr.shape[0])
  x0 = max(cx-cols, 0)
  x1 = min(cx+cols, arr.shape[1])

  patch = arr[x0:x1, y0:y1]
  if square_only and patch.shape[0] != patch.shape[1]:
    logger.debug('Patch was not square')
    return None
  #logger.debug('patch.size: %s' % str(patch.size))

  if mask_val is not None:
    patch = np.ma.masked_equal(patch, mask_val).filled(fill_val)

  if return_idxs:
    rval = (patch, x0, x1, y0, y1)
  else:
    rval = patch

  return rval

@cache
def get_full_data(
    max_rows=None,
    keep_owner_cols=False,
    target_col_terms=TARGET_COL_TERMS,
    remove_empty_cols=False,
    remove_target_correlations=False,
    convert_to_onehot=False,
    from_binary_grid=True,
    patch_size_m=MAG_PATCH_SIZE_M
):
  df, cols_by_type, target_cols = get_mine_data(
      keep_owner_cols=keep_owner_cols,
      target_col_terms=target_col_terms,
      remove_empty_cols=remove_empty_cols,
      remove_target_correlations=remove_target_correlations,
      convert_to_onehot=convert_to_onehot
  )
  lat_col, lon_col = get_lat_lon_cols(df.columns)

  if from_binary_grid:
    grid = GDALGrid(BINARY_MAGNETIC_GRID_PATH)
    #import ipdb; ipdb.set_trace()

    # compare with lon/lat values generated by Surfer
    lonlat_by_xy = {
        (-733200,-1532800): (-101.93015422684,48.448804001006),
        (-733000,-1532800): (-101.92748929237,48.449083011348),
        (-732800,-1532800): (-101.92482432276,48.449361947708)
    }
    for (x, y), (lon, lat) in lonlat_by_xy.items():
      col, row = grid.coord2pixel(x, y)
      _lon, _lat = grid.pixel2lonlat(col, row)
      assert abs(lon - _lon) < 10**-5
      assert abs(lat - _lat) < 10**-5
    min_lon, max_lat = grid.pixel2lonlat(0, 0)
    max_lon, min_lat = grid.pixel2lonlat(grid.x_size-1, grid.y_size-1)
  else:
    mag_df = read_magnetic_data()
    min_lat = min(mag_df.lat)
    max_lat = max(mag_df.lat)
    min_lon = min(mag_df.lon)
    max_lon = max(mag_df.lon)
  logger.debug('lon: (%.2f, %.2f)' % (min_lon, max_lon))
  logger.debug('lat: (%.2f, %.2f)' % (min_lat, max_lat))

  try:
    logger.info('Removing rows without lat/lon...')
    n0 = len(df)
    df = df[pd.notnull(df[lat_col]) &
            pd.notnull(df[lon_col])]
    n1 = len(df)
    logger.info('Removed %d rows without lat/lon' % (n0 - n1))
    logger.info('Removing out of bounds...')
    df = df[(df[lat_col] >= min_lat) &
            (df[lat_col] <= max_lat) &
            (df[lon_col] >= min_lon) &
            (df[lon_col] <= max_lon)]
    n2 = len(df)
    logger.info('Removed %d rows out of bounds' % (n1 - n2))
    logger.info('Resetting index...')
    df = df.reset_index()
  except Exception as e:
    logger.warn('e:', e)
    import ipdb; ipdb.set_trace()

  if from_binary_grid:
    @cache
    def _get_patches(df, grid, lon_col, lat_col, show_patches=False):
      patches = []
      keep_idxs = []
      min_val = grid.arr.min()
      for i, row in df.iterrows():
        logger.debug('extracting patch %d of %d' % (i, df.shape[0]))
        patch = get_patch_from_grid(grid, row[lon_col], row[lat_col], min_val,
            patch_size_m=MAG_PATCH_SIZE_M)
        if patch is not None:
          try:
            print('patch.min():', patch.min())
          except Exception as exc:
            print('exc: %s' % exc)
        if patch is not None:
          if show_patches:
            plt.imshow(patch)
            plt.show()
          try:
            if len(set(patch.flatten())) > 1:
              patches.append(patch)
              keep_idxs.append(i)
          except Exception as exc:
            logger.error('Error: %s' % exc)
            import ipdb; ipdb.set_trace()
      return patches, keep_idxs
    logger.info('Extracting %d patches from grid...' % df.shape[0])
    patches, keep_idxs = _get_patches(df, grid, lon_col, lat_col, read_cache=False)
    df = df.iloc[keep_idxs]
    assert len(df) == len(patches)
  else:
    #df['mag'] = None  # set dtype to Object
    patches = []
    pivot = get_pivot(mag_df)
    for i, row in df.iterrows():
      if max_rows and i > max_rows:
        break
      logger.info('extracting magnetic region %d of %d' % (i, df.shape[0]))
      try:
        mag_region = get_mag_region(mag_df, pivot, row, lat_col, lon_col)
        #df.at[0, 'mag'] = mag_region
        patches.append(mag_region)
      except Exception as e:
        logger.warn('e:', e)
        import ipdb; ipdb.set_trace()

  return patches, df, cols_by_type, target_cols

@cache
def get_magnetic_features(df, patches, num_pca_components=NUM_PCA_COMPONENTS):
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

  logger.info('transforming %d patches of shape %s...' % (len(patches), str(patches[0].shape)))
  for i_patch, patch in enumerate(patches):
    if i_patch and i_patch % 100 == 0:
      progress_bar(i_patch, len(patches))
    mn = patch.min()
    mx = patch.max()
    std = patch.std()
    patch = ((patch - mn) / (mx - mn) * 256).astype('uint8')
    glcm = greycomatrix(patch, [5], [0], symmetric=True, normed=True)
    correlation = greycoprops(glcm, 'correlation')[0][0]
    variance = patch.var()
    contrast = greycoprops(glcm, 'contrast')[0][0]
    entropy = shannon_entropy(patch)
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    asm = greycoprops(glcm, 'ASM')[0][0]
    _sum = patch.sum()

    try:
      patch_pca = pca.transform([patch.flatten()])[0]
      patch_kpca = kpca.transform([patch.flatten()])[0]
      for i in range(num_pca_components):
        feats['pca_%d' % i].append(patch_pca[i])
        feats['kpca_%d' % i].append(patch_kpca[i])
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
  patches, df, cols_by_type, target_cols = get_full_data()
  feats = get_magnetic_features(df, patches)
  for name, vals in feats.items():
    logger.info('name: %s val: %s' % (name, vals[0]))
    df[name] = vals
  mag_feat_names = feats.keys()
  regress(df, target_cols, mag_feat_names)

@cache
def get_pivot(mag_df=None):
  mag_df = mag_df if mag_df is not None else read_magnetic_data()
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

  _display(_intensity, False, 'Magnetic Intensity',       1)
  _display(_contours,  False, 'Magnetic Contours',        2)
  _display(_intensity, True,  'Magnetic Intensity (log)', 3)
  _display(_contours,  True,  'Magnetic Contours (log)',  4)

  if save:
    filename = 'out/grid.png'
    logger.info('Saving to file %s...' % filename)
    plt.savefig(filename, dpi=dpi[0])
  if show:
    logger.info('plt.show()...')
    plt.show()

def display_magnetic_patches(save=True, show=False):
  if not (save or show):
    return

  patches, df, cols_by_type, target_cols = get_full_data()
  lat_col, lon_col = get_lat_lon_cols(df.columns)

  mn = min([patch.min() for patch in patches])
  mx = max([patch.max() for patch in patches])
  logger.info('mn: %.2f, mx: %.2f' % (mn, mx))

  for i, patch in enumerate(patches):
    patch_min = patch.min()
    patch_max = patch.max()
    logger.info('patch_min: %s, patch_max: %s' % (patch_min, patch_max))
    row = df.iloc[i]
    lat = row[lat_col]
    lon = row[lon_col]

    ax = plt.subplot(1,2,1)
    plt.imshow(patch)
    ax.set_title('Relative [%.2f, %.2f]' % (patch_min, patch_max))

    ax = plt.subplot(1,2,2)
    plt.imshow(patch, vmin=mn, vmax=mx)
    ax.set_title('Absolute [%.2f, %.2f]' % (mn, mx))

    plt.suptitle('Magnetic Intensities at lat: %s, lon: %s' % (lat, lon))

    if save:
      filename = 'out/patch_%04d_%s_%s.png' % (i, lat, lon)
      logger.info("Saving to file %s" % filename)
      plt.savefig(filename)
    if show:
      plt.show()

def extract_patch_from_pivot(pivot, cx, cy, box_size_m=MAG_PATCH_SIZE_M):
  half_size = box_size_m / 2.0
  x0 = int(cx - half_size)
  x1 = int(cx + half_size)
  y0 = int(cy - half_size)
  y1 = int(cy + half_size)
  patch = pivot.loc[x0:x1,y0:y1].values
  return patch

def progress_bar(value, endvalue, suffix='', bar_length=40):
  percent = float(value) / endvalue
  arrow = '-' * int(round(percent * bar_length)-1) + '>'
  spaces = ' ' * (bar_length - len(arrow))
  s = '\rProgress: [%s] %.2f%%' % (arrow + spaces, percent * 100)
  if suffix:
    s += ' (%s)' % suffix
  if percent == 1:
    s += '\n'
  sys.stdout.write(s)
  sys.stdout.flush()

@cache
def extract_all_patches(
    flatten=True,
    patch_size_m=MAG_PATCH_SIZE_M,
    stride_size_m=MAG_STRIDE_SIZE_M,
    func=None
):
  mag_df = read_magnetic_data()
  pivot = get_pivot(mag_df)

  min_y = min(mag_df.y)
  max_y = max(mag_df.y)
  min_x = min(mag_df.x)
  max_x = max(mag_df.x)

  # loop over top left corners
  X = []
  half_patch_size = patch_size_m / 2
  logger.debug('half_patch_size: %s' % half_patch_size)
  x_range = range(min_x, max_x - patch_size_m, stride_size_m)
  y_range = range(min_y, max_y - patch_size_m, stride_size_m)
  nx = len(x_range)
  ny = len(y_range)
  n_total = nx * ny
  n = 0
  logger.info('Extracting patches...')
  centroids = []
  for ix, x in enumerate(x_range):
    for iy, y in enumerate(y_range):
      cx = x + half_patch_size
      cy = y + half_patch_size
      centroids.append((cx, cy))
      # TODO: patches are not the same size (e.g. for patch size 1000)
      patch = extract_patch_from_pivot(pivot, cx, cy, patch_size_m)
      if n % 10 == 0:
        progress_bar(n, n_total, 'shape: %s' % str(patch.shape))
      vals = patch.flatten() if flatten else patch
      vals = func(vals) if func else vals
      X.append(vals)
      n += 1
  X = np.array(X)
  logger.info("X.shape: %s" % str(X.shape))

  return X, centroids

@cache
def _do_mag_pca(Klass, pct_rows=None):
  X, _ = extract_all_patches()

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
    patch_size = int(np.sqrt(component.shape[0]))
    patch = np.resize(component, (patch_size, patch_size))
    plt.imshow(patch)

    exp_var_ratio = pca.explained_variance_ratio_[i]
    plt.title('%dth principal component\nexplains %.2f variance' % (
      i, exp_var_ratio))

    if show:
      plt.show()
    if save:
      filename = 'out/%s_%d.png' % (name, i)
      logger.info('Saving to %s...' % filename)
      plt.savefig(filename, dpi=min(patch.shape))
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
    X, _ = extract_all_patches(flatten=False)  # (48000, 51, 51)
    _, nrows, ncols = X.shape
    nrows = nrows if nrows % 2 == 0 else nrows + 1
    ncols = ncols if ncols % 2 == 0 else ncols + 1
    #X = np.array([patch[:nrows,:ncols] for patch in X])
    X = np.array([np.pad(patch, ((0, 1), (0, 1)), mode='edge') for patch in X])
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

@cache
def _get_patch_masks(patches, show=False, majority_only=False):
  rval = []
  for i, patch in enumerate(patches):
    logger.debug('_get_patch_masks() patch %d of %d' % (i, len(patches)))
    im = patch
    im -= im.min()
    im /= im.max()
    im *= 255
    im = im.astype('uint8')

    vals = []
    filts = [f for f in dir(filters) if f.startswith('threshold_')
        and not np.any([f.endswith(w) for w in ['adaptive', 'local']])]
    for f in filts:
      filt = getattr(filters, f)
      try:
        val = filt(im)
      except Exception as e:
        val = np.zeros(im.shape)
        logger.warn('exception while running filt %s: %s' % (f, e))
      vals.append(val)

    masks = [im > val for val in vals]

    # add majority vote
    pixel_counts = np.zeros(patch.shape)
    for mask in masks:
      pixel_counts += mask
    majority_mask = pixel_counts > (len(masks) / 2)
    masks.append(majority_mask)
    filts.append('majority')

    mask_sizes = [m.sum() for m in masks]
    mask_tups = [(mask, size, name.split('_')[-1])
        for mask, size, name in zip(masks, mask_sizes, filts)]
    mask_tups.sort(key=lambda tup: tup[1])
    if majority_only:
      mask_tups = [(mask, size, name) for mask, size, name in mask_tups if name == 'majority']
    rval.append(mask_tups)
    if show:
      N = len(masks) + 1
      nr = np.ceil(np.sqrt(N))
      nc = np.ceil((N/nr))
      ax = plt.subplot(nr,nc,1)
      plt.imshow(patch)
      plt.title('patch')
      for i, (mask, mask_size, mask_name) in enumerate(mask_tups):
        ax = plt.subplot(nr,nc,i+2)
        plt.imshow(mask)
        plt.title(mask_name)
      plt.suptitle('Grid %d' % i)
      plt.show()
  return rval

def _get_mask_feats(patches, masks, feat_names=None):
  feats = {}
  for patch, patch_masks in zip(patches, masks):
    for mask, size, name in patch_masks:
      # density
      if not feat_names or 'density' in feat_names:
        vals = patch[mask]
        density = vals.sum() / mask.size
        feat_name = '%s_density' % name
        feats.setdefault(feat_name, [])
        feats[feat_name].append(density)
      # area
      if not feat_names or 'area' in feat_names:
        area = mask.sum()
        feat_name = '%s_area' % name
        feats.setdefault(feat_name, [])
        feats[feat_name].append(area)
      # sum
      if not feat_names or 'sum' in feat_names:
        vals = patch[mask]
        vals_sum = vals.sum()
        feat_name = '%s_sum' % name
        feats.setdefault(feat_name, [])
        feats[feat_name].append(vals_sum)
      # TODO: more (e.g. convexity, num holes, num connected components...)
  return feats

def do_blob_regression():
  patches, df, cols_by_type, target_cols = get_full_data()
  logger.info('Getting masks...')
  masks = _get_patch_masks(patches, majority_only=True)
  mask_feats = _get_mask_feats(patches, masks, ['sum'])
  for name, vals in mask_feats.items():
    logger.info('name: %s val: %s' % (name, vals[0]))
    df[name] = vals
  mask_feat_names = mask_feats.keys()
  regress(df, target_cols, mask_feat_names)
  import ipdb; ipdb.set_trace()

def _ignore_lat_lon(patches, df, cols_by_type, target_cols):
  logger.info('removing %s lat/lon rows...' % len(IGNORE_LAT_LON_TUPS))
  lat_col, lon_col = get_lat_lon_cols(df)
  for lat, lon in IGNORE_LAT_LON_TUPS:
    ignore_mask = (df[lat_col] == lat) & (df[lon_col] == lon)
    df = df[np.logical_not(ignore_mask)]
    ignore_idxs = np.nonzero(ignore_mask)
    logger.debug('ignore_idxs: %s' % ignore_idxs)
    for ignore_idx in ignore_idxs:  # should be only one
      patches = [g for i, g in enumerate(patches) if i not in ignore_idx]
  return patches, df, cols_by_type, target_cols

def _get_owner_col(df):
  num_uniques = [df[col].unique().size for col in OWNER_COLS]
  col_unique_tups = [(col, num_unique)
      for col, num_unique in zip(OWNER_COLS, num_uniques)]
  col_unique_tups.sort(key=lambda tup: tup[1])
  owner_col = col_unique_tups[-1][0]
  for col in OWNER_COLS:
    if col != owner_col:
      del df[col]
  return owner_col

def _remove_na(vals):
  isna = pd.isna(vals)
  vals = [v for v, na in zip(vals, isna) if not na]
  return np.array(vals)

def plot_mask_sum_vs_target(
    annotate=False,
    show=False,
    save=True,
    # TODO: plot all four combinations
    do_log_target=False,
    do_log_sum=False,
    save_vals=True,
    group_by_owner=False,
    max_total_high_magnetism=None,  #0.5*10**7
    min_by_targ={
      #'Total Deal Value (Announcement) Deal 1 (Reported)': 250
    },
    #bin_split_ratios=[0.0001, 0.001, 0.01, 0.1],
    bin_split_ratios=[1],
    get_full_data_fn=get_full_data
):
  # TODO: refactor
  patches, df, cols_by_type, target_cols = get_full_data_fn(
      keep_owner_cols=group_by_owner,
      read_cache=False,
      return_grid=False
  )
  logger.debug('Got %d patches' % len(patches))
  lat_col, lon_col = get_lat_lon_cols(df)
  logger.info('Getting masks...')
  masks = _get_patch_masks(patches, majority_only=True)
  logger.debug('len(masks): %s' % len(masks))
  mask_feats = _get_mask_feats(patches, masks)
  logger.debug('len(mask_feats): %s' % len(mask_feats))
  sums = mask_feats['majority_sum']
  #TODO: get rid of area
  areas = mask_feats['majority_area']
  if group_by_owner:
    logger.info('Grouping sums and areas by owner...')
    owner_col = _get_owner_col(df)
    owners = df[owner_col].sort_values()
    unique_owners = owners.unique()
    sum_area_tups_by_owner = {}
    for owner, _sum, area in zip(owners, sums, areas):
      sum_area_tups_by_owner.setdefault(owner, [])
      sum_area_tups_by_owner[owner].append((_sum, area))
    sum_sums = []
    area_sums = []
    for owner in unique_owners:
      sum_area_tups = sum_area_tups_by_owner[owner]
      N = len(sum_area_tups)
      sum_sum = sum([_sum for _sum, area in sum_area_tups])
      area_sum = sum([area for _sum, area in sum_area_tups])
      sum_sums.append(sum_sum / N)
      area_sums.append(area_sum / N)
    sums = sum_sums
    areas = area_sums
  df_orig = df

  # binning
  bin_split_ratios = sorted(bin_split_ratios)
  assert bin_split_ratios[-1] <= 1
  if bin_split_ratios[-1] != 1:
    bin_split_ratios.append(1)
  logger.debug('bin_split_ratios: %s' % bin_split_ratios)

  logger.debug('target_cols: %s' % target_cols)
  for i, target_col in enumerate(target_cols):
    df = df_orig
    logger.info('target_col: %s' % target_col)
    df = df[~df[target_col].isna()]
    target_vals = df[target_col]
    max_target_val = max(target_vals)
    min_target_val = min(target_vals)
    logger.info('min: %s, max: %s' % (min_target_val, max_target_val))
    if group_by_owner:
      owners = df[owner_col]
      targ_vals_by_owner = {}
      for target_val, owner in zip(target_vals, owners):
        targ_vals_by_owner.setdefault(owner, [])
        if not pd.isna(target_val):
          targ_vals_by_owner[owner].append(target_val)
      target_vals = []
      unique_owners = owners.unique()
      for owner in unique_owners:
        targ_vals = targ_vals_by_owner[owner]
        N = len(targ_vals)
        val = sum(targ_vals) / N if N is not 0 else 0
        target_vals.append(val)
      target_vals = np.array(target_vals)
      '''
      import ipdb; ipdb.set_trace()
      target_vals = np.array([
        sum(v)/len(targ_vals_by_owner[owner])
        for v in targ_vals_by_owner[owner]
        for owner in unique_owners])
      '''
    isna = pd.isna(target_vals)
    iszero = target_vals <= target_vals.min()
    for _target_col in min_by_targ:
      targ_min = min_by_targ[_target_col]
      logger.info('_target_col: %s, targ_min: %s' % (_target_col, targ_min))
      #df = df[df[_target_col] >= targ_min]
      iszero |= (target_vals <= targ_min)
    target_vals = [v for v, na, zero in zip(target_vals, isna, iszero)
                   if (not na) and (not zero)]
    if not target_vals:
      logger.info('No valid target values, skipping target')
      continue
    target_vals = np.array(target_vals)
    if do_log_target:
      if any([t <= 0 for t in target_vals]):
        target_vals -= target_vals.min()
        target_vals += EPS
      target_vals = np.log(target_vals)
    this_sums = np.array([s for s, na, zero in zip(sums, isna, iszero)
                         if (not na) and (not zero)])
    if do_log_sum:
      this_sums -= this_sums.min()
      this_sums += EPS
      this_sums = np.log(this_sums)
    if max_total_high_magnetism is not None:
      keep_idxs = np.argwhere(this_sums > max_total_high_magnetism)
      this_sums = [this_sums[i] for i in keep_idxs]
      target_vals = [target_vals[i] for i in keep_idxs]

    # split into bins and regress each separately
    X = np.array(this_sums)
    y = np.array(target_vals)
    X_bins = []
    y_bins = []
    y_min = y.min()
    y_max = y.max()
    y_range = y_max - y_min
    prev_bin_split = y_min
    for bin_split_ratio in bin_split_ratios:
      bin_split = y_range * bin_split_ratio

      bin_idxs = (y >= prev_bin_split) & (y < bin_split)
      if bin_split_ratio == 1:
        bin_idxs |= y == bin_split

      X_bin = X[bin_idxs]
      y_bin = y[bin_idxs]
      X_bins.append(X_bin)
      y_bins.append(y_bin)

      prev_bin_split = bin_split

    n_bins = len(bin_split_ratios)
    fontsize = 5
    plt.figure()
    plt.xticks(fontsize=fontsize)
    for i_bin, (X_bin, y_bin) in enumerate(zip(X_bins[::-1], y_bins[::-1])):
      plt.subplot(n_bins, 1, i_bin+1)

      plt.scatter(X_bin, y_bin)
      ylabel = '%s%s%s' % (target_col,
          ' (log)' if do_log_target else '',
          ' (grouped)' if group_by_owner else '')
      xlabel = '%s%s' % ('Total High Magnetism',
          ' (log)' if do_log_sum else '')
      title = '%s vs. %s' % (ylabel, xlabel)
      plt.ylabel(ylabel, fontsize=fontsize)
      plt.xlabel(xlabel, fontsize=fontsize)
      if annotate:
        assert 0, 'not implemented with binning'
        for target_val, magsum, lat, lon in zip(
            target_vals, this_sums, df[lat_col], df[lon_col]):
          plt.annotate('%s x %s' % (lat, lon), (magsum, target_val))

      reg = LinearRegression()
      X_bin = np.array(X_bin).reshape(-1, 1)
      try:
        reg.fit(X_bin, y_bin)
      except Exception as e:
        #import ipdb; ipdb.set_trace()
        continue

      score = reg.score(X_bin, y_bin)

      cv = model_selection.KFold(n_splits=5)
      try:
        cross_val_scores = model_selection.cross_val_score(
            reg, X_bin, y_bin, cv=cv)
        cross_val_score = cross_val_scores.mean()

        title = '%s\nR2=%.4f\nR2_cv=%.4f' % (title, score, cross_val_score)
        plt.title(title, fontsize=fontsize)
        plt.plot(X_bin, reg.predict(X_bin), color='k')
      except Exception as e:
        print('exception: %s' % e)

    figure = plt.gcf()
    figure.set_size_inches(8, 6)

    if show:
      plt.show()
    if save:
      filename = 'out/%s%starget_vs_%smagnetism_%d.png' % (
          'grouped_' if group_by_owner else '',
          'log_' if do_log_target else '',
          'log_' if do_log_sum else '',
          i)
      logger.info('Saving to %s' % filename)
      plt.savefig(filename, dpi=100)

  if save_vals:
    try:
      logger.info('Saving mine magnetic values to %s' % MINE_MAG_VAL_PATH)

      # XXX does not work with binning
      assert bin_split_ratios == [1], "Can't write mag vals with binning"

      with open(MINE_MAG_VAL_PATH, 'w') as csvfile:
        fieldnames = 'id latitude longitude magsum'.split()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        logger.info('len(df_orig): %s' % len(df_orig))
        logger.info('len(sums): %s' % len(sums))
        for _id, lat, lon, mag in zip(
            df_orig[ID_COL],
            df_orig[lat_col],
            df_orig[lon_col],
            sums
        ):
          writer.writerow({
            'id': int(_id),
            'latitude': lat,
            'longitude': lon,
            'magsum': mag
          })
    except Exception as e:
      logger.error('Exception: %s' % e)

  import ipdb; ipdb.set_trace()

def plot_mask_sum_vs_target_grouped():
  plot_mask_sum_vs_target(group_by_owner=True)

def print_stats():
  mag_df = read_magnetic_data()
  min_lat = min(mag_df.lat)
  max_lat = max(mag_df.lat)
  min_lon = min(mag_df.lon)
  max_lon = max(mag_df.lon)
  logger.info('min_lat: %s' % min_lat)
  logger.info('max_lat: %s' % max_lat)
  logger.info('min_lon: %s' % min_lon)
  logger.info('max_lon: %s' % max_lon)

def compare_grids():

  import affine

  def pixel_from_coord(lon, lat, data_source):
    return px, py

  from osgeo import gdal, gdalconst
  grids = [gdal.Open(path, gdalconst.GA_ReadOnly) for path in [
    BINARY_GRAVITY_GRID_PATH,
    BINARY_MAGNETIC_GRID_PATH,
  ]]

  import ipdb; ipdb.set_trace()

  lon, lat = -78.1271, 48.1366
  m = 2000000
  for grid in grids:

    def xy2lonlat(x, y, grid):
      gt = grid.GetGeoTransform()
      x0, pw, ry, y0, rx, ph = gt
      lon = x0 + x*dx + y*ry
      lat = y0 + x*rx + y*ph
      return lon, lat

    forward_transform = affine.Affine.from_gdal(*grid.GetGeoTransform())
    reverse_transform = ~forward_transform
    col, row = reverse_transform * (lon, lat)
    col, row = int(col + 0.5), int(row + 0.5)
    print('%s, %s' % (col, row))

    _, dx, _, _, _, dy = grid.GetGeoTransform()
    dx = abs(dx)
    dy = abs(dy)
    cols = m / dx
    rows = m / dy

    x0 = int(col - cols / 2)
    x1 = int(col + cols / 2)
    y0 = int(row - rows / 2)
    y1 = int(row + rows / 2)

    x0 = 476
    x1 = 645
    y0 = 743
    y1 = 928

    print('%s:%s, %s:%s' % (x0, x1, y0, y1))
    arr = grid.ReadAsArray()

    if abs(dx) == 200:
      print('striding')
      arr = arr[::10,::10]

    print('arr.shape: %s' % str(arr.shape))
    arr = np.ma.masked_values(arr, arr.min())
    #arr[arr == arr.min()] = 0
    arr -= arr.min()
    arr /= arr.max()
    arr[y0:y1, x0:x1] = arr.min()
    plt.figure()
    plt.imshow(arr)
  plt.show()

  return


  from mpl_toolkits.basemap import Basemap
  '''
  magnetic grid top    left  (lon, lat, x, y):(-179.49,61.12 -3265200.0 2479200.0)
  magnetic grid top    right (lon, lat, x, y):(-4.86,  61.95 3174200.0  2479200.0)
  magnetic grid bottom left  (lon, lat, x, y):(-126.18,34.05 -3265200.0 -2353600.0)
  magnetic grid bottom right (lon, lat, x, y):(-58.60, 34.43 3174200.0  -2353600.0)
  gravity grid top     left  (lon, lat, x, y):(173.11, 64.53 -2936000.0 4282000.0)
  gravity grid top     right (lon, lat, x, y):(-0.69,  52.84 4214000.0  4282000.0)
  gravity grid bottom  left  (lon, lat, x, y):(-123.65,31.04 -2936000.0 -1418000.0)
  gravity grid bottom  right (lon, lat, x, y):(-56.41, 25.90 4214000.0  -1418000.0)
  gravity csv min_lon: -186.89  max_lon: -0.69
  gravity csv min_lat: 25.90,   max_lat: 87.41
  gravity csv min_x: -2936000.00        max_x: 4214000.00
  gravity csv min_y: -1418000.00,       max_y: 4282000.00
  '''
  map_by_name = {
    'grav': {
      #'fname': BINARY_GRAVITY_GRID_PATH,
      'llcrnrlon': -123.65,
      'llcrnrlat': 31.04,
      'urcrnrlon': -0.69,
      'urcrnrlat': 52.84,
      'lat_0': 49,
      'lon_0': -95,
    },
    'grav_csv': {
      'llcrnrlon': -186.89,
      'llcrnrlat': 25.90,
      'urcrnrlon': -0.69,
      'urcrnrlat': 87.41,
      #'lat_0': 63,
      #'lon_0': -92,
    },
    'mag': {
      #'fname': BINARY_MAGNETIC_GRID_PATH,
      'llcrnrlon': -126.18,
      'llcrnrlat': 34.05,
      'urcrnrlon': -4.86,
      'urcrnrlat': 61.95,
      'lat_0': 63,
      'lon_0': -92,
    },
    'mag_xml': {
      #'fname': BINARY_MAGNETIC_GRID_PATH,
      'llcrnrlon': -179.49,
      'llcrnrlat': 34.05,
      'urcrnrlon': -4.86,
      'urcrnrlat': 85.22,
      'lat_0': 63,
      'lon_0': -92,
    },
    'test': {
      'lat_0': 10,
      'lon_0': 0,
      'width': 10000000,
      'height': 10000000
    }
  }
  for i, (map_name, map_args) in enumerate(map_by_name.items()):
    #plt.subplot(len(map_by_name),1,i+1)
    plt.figure()
    map_args['projection'] = 'lcc'
    m = Basemap(**map_args)
    # draw coastlines, meridians and parallels.
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='#99ffff')
    m.fillcontinents(color='#cc9966',lake_color='#99ffff')
    if 'llcrnrlat' in map_args:
      m.drawparallels(
          np.arange(
            map_args['llcrnrlat'],
            map_args['urcrnrlat'],
            20
          ),
          labels=[1,0,0,1]
      )
      m.drawmeridians(
          np.arange(
            map_args['llcrnrlon'],
            map_args['urcrnrlon'],
            20
          ),
          labels=[1,0,0,1]
      )
    plt.title(map_name)
  plt.show()

  import ipdb; ipdb.set_trace()

  grid_mag = GDALGrid(BINARY_MAGNETIC_GRID_PATH)
  grid_grv = GDALGrid(BINARY_GRAVITY_GRID_PATH)

  for grid_name, grid in (('magnetic', grid_mag), ('gravity', grid_grv)):
    colrow_by_name = {
      'top \tleft': (0, 0),
      'top \tright': (grid.x_size-1, 0),
      'bottom \tleft': (0, grid.y_size-1),
      'bottom \tright': (grid.x_size-1, grid.y_size-1)
    }
    for name, (col, row) in colrow_by_name.items():
      lon, lat = grid.pixel2lonlat(col, row)
      x, y = grid.pixel2coord(col, row)
      logger.info('%s grid %s \t(lon, lat, x, y): \t(%.2f, \t%.2f\t%s\t%s)' % (
        grid_name, name, lon, lat, x, y))

  _df = pd.read_csv(
    'data/2km gravity anomalies_converted.csv',
    sep=',',
    header=None,
    names='x y mag lon lat'.split(),
    dtype={
      'x': np.int64,
      'y': np.int64,
      'mag': np.float64,
      'lat': np.float64,
      'lon': np.float64
    }
  )
  min_lat = _df.lat.min()
  max_lat = _df.lat.max()
  min_lon = _df.lon.min()
  max_lon = _df.lon.max()
  min_x = _df.x.min()
  min_y = _df.y.min()
  max_x = _df.x.max()
  max_y = _df.y.max()
  logger.debug('gravity csv min_lon: %.2f \tmax_lon: %.2f' % (min_lon, max_lon))
  logger.debug('gravity csv min_lat: %.2f,\tmax_lat: %.2f' % (min_lat, max_lat))
  logger.debug('gravity csv min_x: %.2f \tmax_x: %.2f' % (min_x, max_x))
  logger.debug('gravity csv min_y: %.2f,\tmax_y: %.2f' % (min_y, max_y))

if __name__ == '__main__':
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
      help='Display magnetic patches',
      action='store_true')
  parser.add_argument(
      '-f',
      help='Display full magnetic grid',
      action='store_true')
  parser.add_argument(
      '-a',
      help='Train Autoencoder and save to disk',
      action='store_true')
  parser.add_argument(
      '-b',
      help='Do blob feature regression',
      action='store_true')
  parser.add_argument(
      '-t',
      help='Do blob density test',
      action='store_true')
  parser.add_argument(
      '-l',
      help='Plot mask sum vs. market cap',
      action='store_true')
  parser.add_argument(
      '-g',
      help='Plot mask sum vs. market cap (grouped by owner)',
      action='store_true')
  parser.add_argument(
      '--stats',
      help='Print stats on data',
      action='store_true')
  parser.add_argument(
      '--magnetic',
      help='Read geomagnetic data',
      action='store_true')
  parser.add_argument(
      '--compare',
      help='Compare binary grids for gravity and magnetism',
      action='store_true')
  parser.add_argument(
      '--cmd',
      help='Compare magnetic data implementations',
      action='store_true')

  args = parser.parse_args()

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

  if args.s:
    do_simple_regression()

  if args.m:
    do_magnetic_regression()

  if args.p:
    do_pca()

  if args.k:
    do_kpca()

  if args.d:
    display_magnetic_patches()

  if args.f:
    display_full_magnetic_grid()

  if args.a:
    do_autoencoder()

  if args.b:
    do_blob_regression()

  if args.t:
    do_blob_test()

  if args.l:
    plot_mask_sum_vs_target()

  if args.g:
    plot_mask_sum_vs_target_grouped()

  if args.stats:
    print_stats()

  if args.magnetic:
    data = read_magnetic_data()

  if args.compare:
    compare_grids()

  if args.cmd:
    compare_magnetic_data()
