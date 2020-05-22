from osgeo import gdal, osr

DEBUG = True

BINARY_MAGNETIC_GRID_PATH = 'data/d950396/Canada - 200m - MAG - Residual Total Field - Composante residuelle.grd.gxf'

import logging
log_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

class RasterData(object):
  def __init__(self, raster_path):
    logger.debug('gdal.Open()')
    self.raster = gdal.Open(raster_path)

    logger.debug('raster.ReadAsArray()')
    self.arr = self.raster.ReadAsArray()

    self.n_rows = self.raster.RasterYSize
    self.n_cols = self.raster.RasterXSize

    self.gt = self.raster.GetGeoTransform()
    # origin
    self.ox = self.gt[0]
    self.oy = self.gt[3]
    # pixel size
    self.dx = self.gt[1]
    self.dy = self.gt[5]
    # rotation
    self.ry = self.gt[2]
    self.rx = self.gt[4]

    # https://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/tolatlong.py
    self.srs = osr.SpatialReference()
    assert self.srs.ImportFromWkt(self.raster.GetProjection()) == 0
    self.srsLatLong = self.srs.CloneGeogCS()
    self.ct = osr.CoordinateTransformation(self.srs, self.srsLatLong)

  def get_x(self, col, row, centre=True):
    assert 0 <= row <= self.n_rows
    assert 0 <= col <= self.n_cols
    x = self.ox + col*self.dx + row*self.ry + (self.dx / 2.0 if centre else 0)
    return x

  def get_y(self, col, row, centre=True):
    assert 0 <= row <= self.n_rows
    assert 0 <= col <= self.n_cols
    y = self.oy + col*self.rx + row*self.dy + (self.dy / 2.0 if centre else 0)
    return y

  def get_x_y(self, col, row, centre=True):
    return self.get_x(col, row, centre), self.get_y(col, row, centre)

  def get_lat(self, x, y):
    (lon, lat, height) = self.ct.TransformPoint(x, y)
    return lat

  def get_lon(self, x, y):
    (lon, lat, height) = self.ct.TransformPoint(x, y)
    return lon

  def get_lat_lon(self, x, y):
    (lon, lat, height) = self.ct.TransformPoint(x, y)
    return lat, lon

def get_magnetic_data(
    magnetic_grid_path=BINARY_MAGNETIC_GRID_PATH
):
  rd = RasterData(magnetic_grid_path)
  import ipdb; ipdb.set_trace()

def main():
  mag = get_magnetic_data()

if __name__ == '__main__':
    main()
