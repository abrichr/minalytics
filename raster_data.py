from pprint import pprint

from osgeo import gdal, osr

DEBUG = True

# TODO: move this elsewhere
BINARY_MAGNETIC_GRID_PATH = 'data/d950396/Canada - 200m - MAG - Residual Total Field - Composante residuelle.grd.gxf'

import logging
log_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

class RasterData(object):
  def __init__(self, raster_path=BINARY_MAGNETIC_GRID_PATH):
    logger.debug('gdal.Open()')
    self.raster = gdal.Open(raster_path)

    logger.debug('raster.ReadAsArray()')
    self.arr = self.raster.ReadAsArray()

    self.n_rows = self.raster.RasterYSize
    self.n_cols = self.raster.RasterXSize

    self.gt = self.raster.GetGeoTransform()
    self.gt_inv = gdal.InvGeoTransform(self.gt)

    # origin
    self.ox = self.gt[0]
    self.oy = self.gt[3]

    # pixel size
    self.dx = self.gt[1]
    self.dy = self.gt[5]

    # rotation
    self.ry = self.gt[2]
    self.rx = self.gt[4]

    # coordinate transformation to lat/long
    # https://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/tolonlat.py
    self.srs = osr.SpatialReference()
    self.projection = self.raster.GetProjection()
    assert self.srs.ImportFromWkt(self.projection) == 0, (
        "ERROR: Cannot import projection: %s" % self.projection)
    self.srsLatLong = self.srs.CloneGeogCS()
    self.ct = osr.CoordinateTransformation(self.srs, self.srsLatLong)

    '''
    self.band = self.raster.GetRasterBand(1)
    self.min, self.max = band.GetMinimum(), band.GetMaximum()
    if self.min is None or self.max is None:
      band.ComputeStatistics(0)
      self.min, self.max = band.GetMinimum(), band.GetMaximum()
    '''

  # TODO: are these in degree coordinates or metres?
  '''
  def get_x_from_pixel(self, col, row, centre=True):
    assert 0 <= row <= self.n_rows, 'Invalid row: %s' % row
    assert 0 <= col <= self.n_cols, 'Invalid col: %s' % col
    x = self.ox + col*self.dx + row*self.ry + (self.dx / 2.0 if centre else 0)
    return x

  def get_y_from_pixel(self, col, row, centre=True):
    assert 0 <= row <= self.n_rows, 'Invalid row: %s' % row
    assert 0 <= col <= self.n_cols, 'Invalid col: %s' % col
    y = self.oy + col*self.rx + row*self.dy + (self.dy / 2.0 if centre else 0)
    return y

  def get_xy_from_pixel(self, col, row, centre=True):
    return self.get_x_from_pixel(col, row, centre), \
           self.get_y_from_pixel(col, row, centre)

  def get_lat_from_pixel(self, x, y):
    # TODO: bounds check
    (lon, lat, height) = self.ct.TransformPoint(x, y)
    return lat

  def get_lon_from_pixel(self, x, y):
    # TODO: bounds check
    (lon, lat, height) = self.ct.TransformPoint(x, y)
    return lon

  def get_lonlat_from_pixel(self, x, y):
    # TODO: bounds check
    (lon, lat, height) = self.ct.TransformPoint(x, y)
    return lat, lon
  '''

  def get_pixel_from_lonlat(self, lon, lat):
    # TODO: bounds check
    ''' 
    # Applies the following computation, converting a (pixel, line) coordinate into a georeferenced (geo_x, geo_y) location.
    x, y = gdal.ApplyGeoTransform(self.gt_inv, lon, lat) 
    col = x / self.dx
    row = y / self.dy
    return col, row
    '''
    '''
    x, y, z = self.ct.TransformPoint(lon, lat)
    import ipdb; ipdb.set_trace()
    '''

    # http://gazar.readthedocs.io/en/latest/_modules/gazar/grid.html#GDALGrid.coord2pixel
    #self.grid = GDALGrid(self.raster)

    #self.affine = Affine.from_gdal(*self.raster.GetGeoTransform())

    from gdal_grid import GDALGrid
    self.grid = GDALGrid(self.raster)
    import ipdb; ipdb.set_trace()


    logger.debug('lon,lat: %s -> x,y: %s' % ((lon, lat), (x, y)))
    return x, y

  def get_patch(self, lon, lat, box_size_m):
    '''
    try:
      d_lat = abs(mag_df.lat - row[lat_col])
      d_lon = abs(mag_df.lon - row[lon_col])
      d_tot = d_lat + d_lon
      argmin = d_tot.idxmin()
      row = mag_df.loc[argmin]
      cx = row.x
      cy = row.y
      grid = extract_grid_from_pivot(pivot, cx, cy, box_size_m)
    except Exception as e:
      logger.warn('e: %s' % e)
      import ipdb; ipdb.set_trace()
    '''
    col, row = self.get_pixel_from_lonlat(lon, lat)
    box_size_px_x = box_size_m // self.dx
    box_size_px_y = box_size_m // self.dy
    dx = box_size_px_x // 2
    dy = box_size_px_y // 2
    logger.debug('row: %s, col: %s, dx: %s, dy: %s' % (row, col, dx, dy))
    patch = self.arr[row-dy:row+dy, col-dx:row+dx]
    logger.debug('patch.shape: %s' % str(patch.shape))
    return patch

def get_magnetic_data(
    magnetic_grid_path=BINARY_MAGNETIC_GRID_PATH
):
  rd = RasterData(magnetic_grid_path)
  import ipdb; ipdb.set_trace()

def main():
  mag = get_magnetic_data()


if __name__ == '__main__':
    main()
