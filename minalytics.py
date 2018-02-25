from __future__ import print_function

import logging
log_format = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from cachier import cachier
from collections import Counter
from pprint import pprint, pformat
from sklearn import preprocessing, metrics, linear_model
from xlrd import open_workbook
import cPickle as pickle
import datetime
import pandas as pd
import numpy as np
import scipy as sci
import statsmodels.api as sm
import time

@cachier(stale_after=datetime.timedelta(days=1))
def load_dataframe():
  data_fname = 'snlworkbook_Combined.xls'
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
    if dtype == np.float64:
      num_cols.append(col)
    elif dtype == np.object:
      nonempty_vals = [unicode(v) for v in vals if v not in ['', None]]
      unique_vals = list(set(nonempty_vals))
      if len(unique_vals) <= 1:
        empty_cols.append(col)
        continue

      logger.info('%s: [%s]' % (i_col, col))
      logger.info('first 4 unique_vals: %s' % '\n\t'.join(list([''] + [truncate(s) for s in unique_vals])[:5]))

      logger.debug('len(unique_vals): %s, len(nonempty_vals): %s' % (len(unique_vals), len(nonempty_vals)))
      max_unique_val_len = max(len(v) for v in unique_vals)
      longest_unique_val = [v for v in unique_vals if len(v) == max_unique_val_len][0]
      logger.debug('max_unique_val_len: %s, longest_unique_val: %s' % (max_unique_val_len, longest_unique_val))
      if (len(unique_vals) == len(nonempty_vals) or max_unique_val_len > 500):
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

        dup_part_counts = {part: count for part, count in part_counts.items() if count > 2}
        dup_part_counts_by_col[col] = dup_part_counts
        logger.debug('dup_part_counts:\n%s' % pformat(dup_part_counts))
        if dup_part_counts:
          cols_target = multi_cat_cols
        else:
          cols_target = cat_cols
        
      val_counts = Counter(nonempty_vals)
      dup_val_counts = {val: count for val, count in val_counts.items() if count > 1}
      logger.debug('dup_val_counts:\n%s' % pformat(dup_val_counts))

      if cols_target is cat_cols:
        logger.info('cat_col')
      elif cols_target is text_cols:
        logger.info('text_col')
      else:
        logger.info('multi_cat_col')
      cols_target.append(col)

      logger.info('*' * 40)

    else:
      raise 'Unknown column type'

  def truncate_keys(d):
    rval = {}
    for key, val in d.items():
      rval[truncate(key)] = val
    return rval

  logger.info('num_cols:\n%s' % pformat(num_cols))
  logger.info('text_cols:\n%s' % pformat(text_cols))
  logger.info('cat_cols:\n%s' % pformat({col: truncate_keys(dup_part_counts_by_col[col]) for col in cat_cols}))
  logger.info('multi_cat_cols:\n%s' % pformat({col: truncate_keys(dup_part_counts_by_col[col]) for col in multi_cat_cols}))

  return {
    ColType.NUM: num_cols,
    ColType.TEXT: text_cols,
    ColType.CAT: cat_cols,
    ColType.MULTI: multi_cat_cols,
    ColType.EMPTY: empty_cols
  }

  #return num_cols, text_cols, cat_cols, multi_cat_cols, empty_cols

def remove_empty_cols(df, ignore_cols=None, min_ratio=0.3):
  ignore_cols = ignore_cols or []
  col_ratios = []
  for col in df.columns:
    vals = df[col]
    num_vals = len(vals)
    if any(pd.notnull(vals)):
      nonempty = vals[pd.notnull(vals)]
    else:
      nonempty = []
    num_nonempty = len(nonempty)
    ratio = round(1.0 * num_nonempty / num_vals, 2)
    col_ratios.append((ratio, col))
  col_ratios.sort(key=lambda x: x[0])
  logger.info('col_ratios: %s' % pformat(col_ratios))

  cols_to_remove = set([col for ratio, col in col_ratios
                        if ratio < min_ratio or col in ignore_cols])
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
    num_to_add = len([_ for _, count in part_counts.items() if count > min_count])
    part_counts_to_add = {p: c for p, c in part_counts.items() if c > min_count}
    for i, (part, count) in enumerate(part_counts_to_add.items()):
      col_name = '%s_%s' % (col, part)
      new_cols.append(col_name)
      logger.info('Adding column %d of %d: %s' % (i, num_to_add, col_name))
      num_added += 1
      df[col_name] = df[col].map(lambda x: 1 if x and part in x else 0)
    num_added -= 1
    del df[col]
  assert len(df.columns) == prev_num_cols + num_added, (len(df.columns), prev_num_cols, num_added)
  cols_by_type[ColType.MULTI].extend(new_cols)

def expand_categories(df, cols_by_type, max_unique_vals_to_rows=0.01):
  num_cols_to_add = 0
  #unique_vals_by_col = {}
  for col in sorted(cols_by_type[ColType.CAT]):
    logger.info('col: %s' % col)
    vals = df[col]
    unique_vals = set(vals) - set([None])
    #unique_vals_by_col[col] = unique_vals
    unique_vals_to_rows = 1.0 * len(unique_vals) / df.shape[0]
    logger.info('\tunique_vals_to_rows: %.2f, len(unique_vals): %s' % (unique_vals_to_rows, len(unique_vals)))
    if 1.0 * len(unique_vals) / df.shape[0] > max_unique_vals_to_rows:
      logger.info('\t(removing)')
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
      assert len(new_cols) == len(unique_vals), (len(new_cols), len(unique_vals))
  return df

@cachier()
def load_preproc_data():

  df = load_dataframe()

  remove_empty_cols(df)

  cols_by_type = parse_cols(df)

  target_cols = [col for col in cols_by_type[ColType.NUM]
                 if any(word in col.lower() for word in ['capitalization', 'value'])]
  logger.info('target_cols: %s' % pformat(target_cols))

  logger.info('df.shape: %s' % str(df.shape))
  logger.info('Converting multis...')
  convert_multi_to_onehot(df, cols_by_type)

  logger.info('df.shape: %s' % str(df.shape))
  logger.info('Expanding categories...')
  df = expand_categories(df, cols_by_type)

  logger.info('df.shape: %s' % str(df.shape))
  logger.info('Removing text columns...')
  remove_columns(df, cols_by_type[ColType.TEXT])

  for col, dtype in zip(df.columns, df.dtypes):
    logger.debug('dtype: %s, col: %s' % (dtype, col))

  logger.info('df.shape: %s' % str(df.shape))

  return df, cols_by_type, target_cols

def main():

  logger.info('Preparing data...')
  start_time = time.time()
  df, cols_by_type, target_cols = load_preproc_data(ignore_cache=True)
  logger.info('Done, took: %.2fs' % (time.time() - start_time))

  for target_col in target_cols:
    logger.info('target_col: %s' % target_col)
    this_df = df[pd.notnull(df[target_col].values)]
    this_df = this_df.fillna(0)
    logger.debug('this_df.shape: %s' % str(this_df.shape))
    y = this_df[target_col].values
    del this_df[target_col]
    X = this_df.values
    logger.debug('X.shape: %s, y.shape: %s' % ((X.shape, y.shape)))

    if 0:
      X = sm.add_constant(X)
      logger.info('fitting...')
      model = sm.OLS(y, X).fit()
      model.summary()

    # TODO: scipy.sparse.csr_matrix

    # normalize
    # http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization
    logger.debug('Normalizing...')
    X_norm = preprocessing.normalize(X)

    reg = linear_model.LinearRegression()
    logger.debug('Fitting...')
    reg.fit(X_norm, y)

    # TODO: cross val
    logger.debug('Predicting...')
    y_pred = reg.predict(X_norm)

    r2 = metrics.r2_score(y, y_pred)
    logger.info('r2: %s' % r2)
    

if __name__ == '__main__':
  main()
