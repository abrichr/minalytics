from gdal_grid import GDALGrid

from minalytics import *

DEBUG = True

# Size of patch of magnetic data in metres
MAG_PATCH_SIZE_M = 10000

import logging
log_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

def get_patch_from_grid(grid, lon, lat, patch_size_m=MAG_PATCH_SIZE_M):
  logger.debug('get_patch_from_grid() lon: %s, lat: %s' % (lon, lat))

  if np.isnan(lon) or np.isnan(lat):
    loggder.debug('lon/lat was nan')
    return None

  try:
    cx, cy = grid.lonlat2pixel(lon, lat)
  except IndexError:
    logger.debug('No patch at (lon, lat): (%s, %s)' % (lon, lat))
    return None

  dx = int(patch_size_m / abs(grid.dx))
  dy = int(patch_size_m / abs(grid.dy))
  arr = grid.arr

  y0 = max(cy-dy, 0)
  y1 = min(cy+dy, arr.shape[0])
  x0 = max(cx-dx, 0)
  x1 = min(cx+dx, arr.shape[1])

  patch = arr[x0:x1, y0:y1]
  logger.debug('patch.size: %s' % str(patch.size))

  return patch

def get_full_data(max_rows=None, keep_owner_cols=False, show_patches=False):
  df, cols_by_type, target_cols = get_mine_data(keep_owner_cols=keep_owner_cols)
  lat_col, lon_col = get_lat_lon_cols(df.columns)

  grid = GDALGrid(BINARY_MAGNETIC_GRID_PATH)

  # compare with lon/lat values generated by Surfer
  lonlat_by_yx = {
      (-733200,-1532800): (-101.93015422684,48.448804001006),
      (-733000,-1532800): (-101.92748929237,48.449083011348),
      (-732800,-1532800): (-101.92482432276,48.449361947708)
  }
  for (x, y), (lon, lat) in lonlat_by_yx.items():
    col, row = grid.coord2pixel(x, y)
    _lon, _lat = grid.pixel2lonlat(col, row)
    assert abs(lon - _lon) < 10**-5
    assert abs(lat - _lat) < 10**-5

  min_lon, max_lat = grid.pixel2lonlat(0, 0)
  max_lon, min_lat = grid.pixel2lonlat(grid.x_size-1, grid.y_size-1)
  logger.debug('min_lon: %s, max_lon: %s' % (min_lon, max_lon))
  logger.debug('min_lat: %s, max_lat: %s' % (min_lat, max_lat))

  try:
    logger.debug('Removing rows without lat/lon...')
    n0 = len(df)
    df = df[pd.notnull(df[lat_col]) &
            pd.notnull(df[lon_col])]
    n1 = len(df)
    logger.debug('Removed %d rows without lat/lon' % (n0 - n1))
    logger.debug('Removing out of bounds...')
    df = df[(df[lat_col] >= min_lat) &
            (df[lat_col] <= max_lat) &
            (df[lon_col] >= min_lon) &
            (df[lon_col] <= max_lon)]
    n2 = len(df)
    logger.debug('Removed %d rows out of bounds' % (n1 - n2))
    logger.debug('Resetting index...')
    df = df.reset_index()
  except Exception as e:
    logger.warn('e:', e)
    import ipdb; ipdb.set_trace()

  patches = []
  for i, row in df.iterrows():
    logger.debug('extracting patch %d of %d' % (i, df.shape[0]))
    patch = get_patch_from_grid(grid, row[lon_col], row[lat_col])
    if patch is not None:
      if show_patches:
        plt.imshow(patch)
        plt.show()
      if len(set(patch.flatten())) > 1:
        patches.append(patch)

  return patches, df, cols_by_type, target_cols

if __name__ == '__main__':
  plot_mask_sum_vs_mkt_cap(get_full_data_fn=get_full_data)
  #jpatches, df, cols_by_type, target_cols = get_full_data()
  import ipdb; ipdb.set_trace()
