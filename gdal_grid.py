# Modified from http://gazar.readthedocs.io/

# -*- coding: utf-8 -*-
#
#  gazar.grid
#
#  Author : Alan D Snow, 2017.
#  License: BSD 3-Clause

"""gazar.grid docstring
This module is a collection of GDAL functions and tools for grids.
Documentation can be found at `_gazar Documentation HOWTO`_.

.. _gazar Documentation HOWTO:
   https://github.com/snowman2/gazar

"""
# default modules
from csv import writer as csv_writer
import itertools
import os

# external modules
from affine import Affine
from mapkit import lookupSpatialReferenceID
import numpy as np
from osgeo import gdal, gdalconst, ogr, osr
from pyproj import Proj, transform
import utm

gdal.UseExceptions()


def utm_proj_from_latlon(latitude, longitude, as_wkt=False, as_osr=False):
    """
    Returns UTM projection information from a latitude,
    longitude corrdinate pair.

    Parameters
    ----------
    latitude : float
        The center latitude.
    longitude:  float
        The center longitude.
    as_wkt:  bool, optional
        If True, will return the WKT projection string.
    as_osr: bool, optional
        If True, will return the :func:`osr.SpatialReference` object.

    Returns
    -------
    :obj:`str` or :func:`osr.SpatialReference`
        Defaults to the proj.4 string.
    """
    # get utm coordinates
    utm_centroid_info = utm.from_latlon(latitude, longitude)
    zone_number, zone_letter = utm_centroid_info[2:]

    # METHOD USING SetUTM. Not sure if better/worse
    sp_ref = osr.SpatialReference()

    south_string = ''
    if zone_letter < 'N':
        south_string = ', +south'
    proj4_utm_string = ('+proj=utm +zone={zone_number}{zone_letter}'
                        '{south_string} +ellps=WGS84 +datum=WGS84 '
                        '+units=m +no_defs')\
        .format(zone_number=abs(zone_number),
                zone_letter=zone_letter,
                south_string=south_string)
    ret_val = sp_ref.ImportFromProj4(proj4_utm_string)
    if ret_val == 0:
        north_zone = True
        if zone_letter < 'N':
            north_zone = False
        sp_ref.SetUTM(abs(zone_number), north_zone)

    sp_ref.AutoIdentifyEPSG()

    if as_osr:
        return sp_ref
    elif as_wkt:
        return sp_ref.ExportToWkt()
    return sp_ref.ExportToProj4()



def project_to_geographic(x_coord, y_coord, osr_projetion):
    """Project point to EPSG:4326

    Parameters
    ----------
    x_coord : float
        The point x-coordinate.
    y_coord:  float
        The point y-coordinate.
    osr_projetion:  :func:`osr.SpatialReference`
        The projection for the point.

    Returns
    -------
    :obj:`tuple`
        The projected point coordinates.
        (x_coord, y_coord)
    """
    # Make sure projected into global projection
    sp_ref = osr.SpatialReference()
    sp_ref.ImportFromEPSG(4326)
    trans = osr.CoordinateTransformation(osr_projetion, sp_ref)
    return trans.TransformPoint(x_coord, y_coord)[:2]


class GDALGrid(object):
    """
    Wrapper for :func:`gdal.Dataset` with
    :func:`osr.SpatialReference` object.

    Parameters
    ----------
    grid_file : :obj: str` or :func:`gdal.Dataset`
        The grid file to be wrapped.
    prj_file : :obj:`str`, optional
        Path to projection file.

    """
    def __init__(self, grid_file, prj_file=None,
                 min_val_replacement=None, shift_to_zero=True):
        if isinstance(grid_file, gdal.Dataset):
            self.dataset = grid_file
        else:
            self.dataset = gdal.Open(grid_file, gdalconst.GA_ReadOnly)

        self.arr = self.dataset.ReadAsArray()

        # XXX HACK
        # the .gxf file contains the following lines:
        #   #ZMINIMUM
        #   -6586.06543
        #   #ZMAXIMUM
        #   728998
        # but these values do not appear to be accessible from GDAL Python
        # TODO: read these values directly from file
        if min_val_replacement is not None:
            self.arr[self.arr == self.arr.min()] = min_val_replacement

        if shift_to_zero:
            self.arr -= self.arr.min()

        # set projection object
        if prj_file is not None:
            self.projection = osr.SpatialReference()
            with open(prj_file) as pro_file:
                self.projection.ImportFromWkt(pro_file.read())
        else:
            self.projection = osr.SpatialReference()
            self.projection.ImportFromWkt(self.dataset.GetProjection())

        # identify EPSG code where applicable
        self.projection.AutoIdentifyEPSG()
        # set affine from geotransform
        self.affine = Affine.from_gdal(*self.dataset.GetGeoTransform())

    def __repr__(self):
        # hack to cache results of functions whose arguments include a GDALGrid
        # TODO: compute file hash from filename
        return '%s\n%s' % (self.dataset.GetFileList(), repr(self.arr))

    @property
    def geotransform(self):
        """:obj:`tuple`: The geotransform for the dataset."""
        return self.dataset.GetGeoTransform()

    @property
    def dx(self):
        return self.geotransform[1]

    @property
    def dy(self):
        return self.geotransform[5]

    @property
    def x_size(self):
        """int: size of x dimensions"""
        return self.dataset.RasterXSize

    @property
    def y_size(self):
        """int: size of y dimensions"""
        return self.dataset.RasterYSize

    @property
    def num_bands(self):
        """int: number of bands in raster"""
        return self.dataset.RasterCount

    @property
    def wkt(self):
        """:obj:`str`:WKT projection string"""
        return self.projection.ExportToWkt()

    @property
    def proj4(self):
        """:obj:`str`:proj4 string"""
        return self.projection.ExportToProj4()

    @property
    def proj(self):
        """func:`pyproj.Proj`: Proj4 object"""
        return Proj(self.proj4)

    @property
    def epsg(self):
        """:obj:`str`: EPSG code"""
        epsg = self.projection.GetAuthorityCode(None)
        if epsg:
            return epsg
        # attempt online lookup
        return lookupSpatialReferenceID(self.wkt)

    def bounds(self, as_geographic=False, as_utm=False, as_projection=None):
        """Returns bounding coordinates for the dataset.

        Parameters
        ----------
        as_geographic : bool, optional
            If True, this will return the bounds in EPSG:4326.
            Default is False.
        as_utm:  bool, optional
            If True, it will attempt to find the UTM zone and
            will return bounds in that UTM zone.
        as_projection:  :func:`osr.SpatialReference`, optional
            Output projection for bounds.

        Returns
        -------
        :obj:`tuple`
            (x_min, x_max, y_min, y_max)
            Bounds for the grid in the format

        """
        new_proj = None
        x_min, y_min = self.affine * (0, self.dataset.RasterYSize)
        x_max, y_max = self.affine * (self.dataset.RasterXSize, 0)

        if as_geographic:
            new_proj = osr.SpatialReference()
            new_proj.ImportFromEPSG(4326)
        elif as_utm:
            lon_min, lat_max = project_to_geographic(x_min, y_max,
                                                     self.projection)
            lon_max, lat_min = project_to_geographic(x_max, y_min,
                                                     self.projection)
            # convert to UTM
            new_proj = utm_proj_from_latlon((lat_min + lat_max) / 2.0,
                                            (lon_min + lon_max) / 2.0,
                                            as_osr=True)
        elif as_projection:
            new_proj = as_projection

        if new_proj is not None:
            ggrid = self.to_projection(new_proj)
            return ggrid.bounds()

        return x_min, x_max, y_min, y_max


    def pixel2coord(self, col, row):
        """Returns global coordinates to pixel center using base-0 raster index.

        Parameters
        ----------
        col: int
            The 0-based column index.
        row:  int
            The 0-based row index.

        Returns
        -------
        :obj:`tuple`
            (x_coord, y_coord) - The x, y coordinate of the pixel
            center in the dataset's projection.

        """
        if col >= self.x_size:
            raise IndexError("Column index out of bounds...")
        if row >= self.y_size:
            raise IndexError("Row index is out of bounds ...")
        return self.affine * (col + 0.5, row + 0.5)


    def coord2pixel(self, x_coord, y_coord):
        """Returns base-0 raster index using global coordinates to pixel center

        Parameters
        ----------
        x_coord: float
            The projected x coordinate of the cell center.
        y_coord:  float
            The projected y coordinate of the cell center.

        Returns
        -------
        :obj:`tuple`
            (col, row) - The 0-based column and row index of the pixel.
        """
        col, row = ~self.affine * (x_coord, y_coord)
        if col > self.x_size or col < 0:
            raise IndexError("Longitude {0} is out of bounds ..."
                             .format(x_coord))
        if row > self.y_size or row < 0:
            raise IndexError("Latitude {0} is out of bounds ..."
                             .format(y_coord))

        return int(col), int(row)


    def pixel2lonlat(self, col, row):
        """Returns latitude and longitude to pixel center using base-0 raster index

        Parameters
        ----------
        col: int
            The 0-based column index.
        row:  int
            The 0-based row index.

        Returns
        -------
        :obj:`tuple`
            (longitude, latitude) - The lat, lon of the pixel
            center in the dataset's projection.
        """
        x_coord, y_coord = self.pixel2coord(col, row)
        longitude, latitude = project_to_geographic(x_coord, y_coord,
                                                    self.projection)
        return longitude, latitude


    def lonlat2pixel(self, longitude, latitude):
        """Returns base-0 raster index using longitude and latitude of pixel center

        Parameters
        ----------
        longitude: float
            The longitude of the cell center.
        latitude:  float
            The latitude of the cell center.

        Returns
        -------
        :obj:`tuple`
            (col, row) - The 0-based column and row index of the pixel.
        """
        sp_ref = osr.SpatialReference()
        sp_ref.ImportFromEPSG(4326)  # geographic
        transx = osr.CoordinateTransformation(sp_ref, self.projection)
        x_coord, y_coord = transx.TransformPoint(longitude, latitude)[:2]
        return self.coord2pixel(x_coord, y_coord)


    @property
    def coords(self):
        """Returns x and y coordinate arrays representing the grid.

        Returns
        -------
        y_coords: :func:`numpy.array`
            The Y coordinate array.
        x_coords: :func:`numpy.array`
            The X coordinate array.
        """
        y_coords = np.zeros((self.y_size, self.x_size))
        x_coords = np.zeros((self.y_size, self.x_size))
        for xii, yii in itertools.product(
                range(self.x_size), range(self.y_size)):
            x_coords[yii, xii], y_coords[yii, xii] = \
                self.pixel2coord(xii, yii)

        return y_coords, x_coords

    @property
    def latlon(self):
        """Returns latitude and longitude arrays representing the grid.

        Returns
        -------
        proj_lats: :func:`numpy.array`
            The latitude array.
        proj_lons: :func:`numpy.array`
            The longitude array.
        """
        y_coords, x_coords = self.coords

        proj_lons, proj_lats = transform(self.proj,
                                         Proj(init='epsg:4326'),
                                         x_coords,
                                         y_coords)
        return proj_lats, proj_lons

    def np_array(self, band=1, masked=True):
        """Returns the raster band as a numpy array.

        Parameters
        ----------
        band: obj:`int`, optional
            Band number (1-based). Default is 1. If 'all',
            it will return all of the data as a 3D array.
        masked: bool, optional
            If True, will return the array masked with the NoData
            value. Default is True.

        Returns
        -------
        :func:`numpy.array` or :func:`numpy.ma.array`
        """
        if band == 'all':
            #grid_data = self.dataset.ReadAsArray()
            grid_data = self.arr
        else:
            raster_band = self.dataset.GetRasterBand(band)
            #grid_data = raster_band.ReadAsArray()
            grid_data = self.arr
            nodata_value = raster_band.GetNoDataValue()
            if nodata_value is not None and masked:
                return np.ma.array(data=grid_data,
                                   mask=(grid_data == nodata_value))
        return np.array(grid_data)


    def get_val(self, x_pixel, y_pixel, band=1):
        """Returns value of raster

        Parameters
        ----------
        x_pixel: int
            X pixel location (0-based).
        y_pixel: int
            Y pixel location (0-based).
        band: int, optional
            Band number (1-based). Default is 1.

        Returns
        -------
        object dtype
        """
        return self.dataset.GetRasterBand(band)\
                   .ReadAsArray(x_pixel, y_pixel, 1, 1)[0][0]


    def get_val_latlon(self, longitude, latitude, band=1):
        """Returns value of raster from a latitude and longitude point.

        Parameters
        ----------
        longitude: float
            The longitude of the cell center.
        latitude:  float
            The latitude of the cell center.
        band: int, optional
            Band number (1-based). Default is 1.

        Returns
        -------
        object dtype
        """
        x_pixel, y_pixel = self.lonlat2pixel(longitude, latitude)
        return self.get_val(x_pixel, y_pixel, band)


    def get_val_coord(self, x_coord, y_coord, band=1):
        """Returns value of raster from a projected coordinate point.

        Parameters
        ----------
        x_coord: float
            The projected x coordinate of the cell center.
        y_coord:  float
            The projected y coordinate of the cell center.
        band: int, optional
            Band number (1-based). Default is 1.

        Returns
        -------
        object dtype
        """
        x_pixel, y_pixel = self.coord2pixel(x_coord, y_coord)
        return self.get_val(x_pixel, y_pixel, band)


    def write_prj(self, out_projection_file, esri_format=False):
        """Writes projection file.

        Parameters
        ----------
        out_projection_file:  :obj:`str`
            Output path for file.
        esri_format: bool, optional
            If True, it will convert the projection string to
            the Esri format. Default is False.
        """
        if esri_format:
            self.projection.MorphToESRI()
        with open(out_projection_file, 'w') as prj_file:
            prj_file.write(self.wkt)
            prj_file.close()


    def to_polygon(self,
                   out_shapefile,
                   band=1,
                   fieldname='DN',
                   self_mask=None):
        """Converts the raster to a polygon.

        Based on:
        ---------
        https://svn.osgeo.org/gdal/trunk/gdal/swig/python/scripts
            /gdal_polygonize.py

        https://stackoverflow.com/questions/25039565
            /create-shapefile-from-tif-file-using-gdal

        Parameters
        ----------
        out_shapefile:  :obj:`str`
            Output path for shapefile.
        band: int, optional
            Band number (1-based). Default is 1.
        fieldname: str, optional
            Name of the output field. Defailt is 'DN'.
        self_mask: bool, optional
            If True, will use self as mask. Default is None.
        """

        raster_band = self.dataset.GetRasterBand(band)
        if self_mask:
            self_mask = raster_band
        else:
            self_mask = None

        drv = ogr.GetDriverByName("ESRI Shapefile")
        dst_ds = drv.CreateDataSource(out_shapefile)
        dst_layername = os.path.splitext(os.path.basename(out_shapefile))[0]
        dst_layer = dst_ds.CreateLayer(dst_layername, srs=self.projection)

        # mapping between gdal type and ogr field type
        type_mapping = {gdal.GDT_Byte: ogr.OFTInteger,
                        gdal.GDT_UInt16: ogr.OFTInteger,
                        gdal.GDT_Int16: ogr.OFTInteger,
                        gdal.GDT_UInt32: ogr.OFTInteger,
                        gdal.GDT_Int32: ogr.OFTInteger,
                        gdal.GDT_Float32: ogr.OFTReal,
                        gdal.GDT_Float64: ogr.OFTReal,
                        gdal.GDT_CInt16: ogr.OFTInteger,
                        gdal.GDT_CInt32: ogr.OFTInteger,
                        gdal.GDT_CFloat32: ogr.OFTReal,
                        gdal.GDT_CFloat64: ogr.OFTReal}

        fld = ogr.FieldDefn(fieldname, type_mapping[raster_band.DataType])
        dst_layer.CreateField(fld)

        gdal.Polygonize(raster_band,
                        self_mask,
                        dst_layer,
                        0,
                        [],
                        callback=None)


    def to_projection(self, dst_proj,
                      resampling=gdalconst.GRA_NearestNeighbour):
        """Reproject dataset to new projection.

        Parameters
        ----------
        dst_proj:  :func:`osr.SpatialReference`
            Output projection.

        Returns
        -------
        :func:`~GDALGrid`
        """
        return gdal_reproject(self.dataset,
                              src_srs=self.projection,
                              dst_srs=dst_proj,
                              resampling=resampling,
                              as_gdal_grid=True)


    def to_tif(self, file_path):
        """Write out as geotiff.

        Parameters
        ----------
        file_path:  :obj:`str`
            Output path for file.
        """
        drv = gdal.GetDriverByName('GTiff')
        drv.CreateCopy(file_path, self.dataset)


    def _to_ascii(self, header_string, file_path, band, print_nodata=True):
        """Writes data to ascii file"""
        if print_nodata:
            nodata_value = self.dataset.GetRasterBand(band).GetNoDataValue()
            if nodata_value is not None:
                header_string += "NODATA_value {0}\n".format(nodata_value)

        with open(file_path, 'w') as out_ascii_grid:
            out_ascii_grid.write(header_string)
            grid_writer = csv_writer(out_ascii_grid,
                                     delimiter=" ")
            grid_writer.writerows(self.np_array(band, masked=False))

    def to_grass_ascii(self, file_path, band=1, print_nodata=True):
        """Writes data to GRASS ASCII file format.

        Parameters
        ----------
            file_path: :obj:`str`
                Path to output ascii file.
            band: obj:`int`, optional
                Band number (1-based). Default is 1.
            print_nodata: bool, optional
                If True, it will write out the NoData value
                for the raster band. Default is False.
        """
        # PART 1: HEADER
        # get data extremes
        west_bound, east_bound, south_bound, north_bound = self.bounds()
        header_string = u"north: {0:.9f}\n".format(north_bound)
        header_string += "south: {0:.9f}\n".format(south_bound)
        header_string += "east: {0:.9f}\n".format(east_bound)
        header_string += "west: {0:.9f}\n".format(west_bound)
        header_string += "rows: {0}\n".format(self.y_size)
        header_string += "cols: {0}\n".format(self.x_size)

        # PART 2: WRITE DATA
        self._to_ascii(header_string, file_path, band, print_nodata)


    def to_arc_ascii(self, file_path, band=1, print_nodata=True):
        """Writes data to Arc ASCII file format.

        Parameters
        ----------
            file_path: :obj:`str`
                Path to output ascii file.
            band: obj:`int`, optional
                Band number (1-based). Default is 1.
            print_nodata: bool, optional
                If True, it will write out the NoData value
                for the raster band. Default is False.
        """
        # PART 1: HEADER
        # get data extremes
        bounds = self.bounds()
        west_bound = bounds[0]
        south_bound = bounds[2]
        cellsize = (self.geotransform[1] - self.geotransform[-1]) / 2.0
        header_string = u"ncols {0}\n".format(self.x_size)
        header_string += "nrows {0}\n".format(self.y_size)
        header_string += "xllcorner {0}\n".format(west_bound)
        header_string += "yllcorner {0}\n".format(south_bound)
        header_string += "cellsize {0}\n".format(cellsize)

        # PART 2: WRITE DATA
        self._to_ascii(header_string, file_path, band, print_nodata)



class ArrayGrid(GDALGrid):
    """
    Loads :func:`numpy.array` into a :func:`~GDALGrid`.

    Parameters
    ----------
    in_array : :func:`numpy.array`
        2D or 3D array of data.
    wkt_projection : :obj:`str`
        WKT projection string.
    geotransform: :obj:`tuple`
        Geotransform for array.
    gdal_dtype: :func:`gdalconst`, optional
        The data type of the `in_array` for GDAL.
        Default is `gdalconst.GDT_Float32`.
    nodata_value: int or float, optional
        The value used in the grid for NoData. Default is None.

    """
    def __init__(self,
                 in_array,
                 wkt_projection,
                 geotransform,
                 gdal_dtype=gdalconst.GDT_Float32,
                 nodata_value=None):

        num_bands = 1
        if in_array.ndim == 3:
            num_bands, y_size, x_size = in_array.shape
        else:
            y_size, x_size = in_array.shape

        dataset = gdal.GetDriverByName('MEM').Create("tmp_ras",
                                                     x_size,
                                                     y_size,
                                                     num_bands,
                                                     gdal_dtype)

        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(wkt_projection)

        if in_array.ndim == 3:
            for band in range(1, num_bands + 1):
                rband = dataset.GetRasterBand(band)
                rband.WriteArray(in_array[band - 1])
                if nodata_value is not None:
                    rband.SetNoDataValue(nodata_value)
        else:
            rband = dataset.GetRasterBand(1)
            rband.WriteArray(in_array)
            if nodata_value is not None:
                rband.SetNoDataValue(nodata_value)

        super(ArrayGrid, self).__init__(dataset)



def geotransform_from_yx(y_arr, x_arr, y_cell_size=None, x_cell_size=None):
    """
    Calculates geotransform from arrays of y and x coords.
    Assumes Y max and X min are at [0,0].

    Parameters
    ----------
        y_arr: :func:`numpy.array`
            Array of latitudes or y coordinates.
        x_arr: :func:`numpy.array`
            Array of longitudes or x coordinates.
        y_cell_size: :obj:`float`, optional
            Y cell size in projected coordinates.
        x_cell_size: :obj:`float`, optional
            X cell size from projected coordinates.

    Returns
    -------
    :obj:`tuple`
        geotransform: (x_min, x_cell_size, x_skew, y_max, y_skew, -y_cell_size)
    """
    if y_arr.ndim < 2:
        x_2d, y_2d = np.meshgrid(x_arr, y_arr)
    else:
        x_2d = x_arr
        y_2d = y_arr
    # get cell size
    if x_cell_size is None:
        x_cell_size = np.nanmean(np.absolute(np.diff(x_2d, axis=1)))
    if y_cell_size is None:
        y_cell_size = np.nanmean(np.absolute(np.diff(y_2d, axis=0)))
    # get top left corner
    min_x_tl = x_2d[0, 0] - x_cell_size / 2.0
    max_y_tl = y_2d[0, 0] + y_cell_size / 2.0
    return min_x_tl, x_cell_size, 0, max_y_tl, 0, -y_cell_size



def load_raster(grid):
    """
    Load in a raster as a :func:`~GDALGrid`.

    Parameters
    ----------
        grid: :obj:`str` or :func:`gdal.Dataset` or :func:`~GDALGrid`
            The raster to be loaded in.
    Returns
    -------
    :func:`~GDALGrid`
    """
    if isinstance(grid, gdal.Dataset):
        src = grid
        src_proj = src.GetProjection()
    elif isinstance(grid, GDALGrid):
        src = grid.dataset
        src_proj = grid.wkt
    else:
        src = gdal.Open(grid, gdalconst.GA_ReadOnly)
        src_proj = src.GetProjection()

    return src, src_proj


def resample_grid(original_grid,
                  match_grid,
                  to_file=False,
                  output_datatype=None,
                  resample_method=gdalconst.GRA_Average,
                  as_gdal_grid=False):
    """
    This function resamples a grid and outputs the result to a file.

    Based on: http://stackoverflow.com/questions/10454316/how-to-project-and-
        resample-a-grid-to-match-another-grid-with-gdal-python

    Parameters
    ----------
        original_grid: :obj:`str` or :func:`gdal.Dataset` or :func:`~GDALGrid`
            The original grid dataset.
        match_grid: :obj:`str` or :func:`gdal.Dataset` or :func:`~GDALGrid`
            The grid to match.
        to_file: :obj:`str` or bool, optional
            Default is False, which returns an in memory grid.
            If :obj:`str`, it writes to file.
        output_datatype: :func:`osgeo.gdalconst`, optional
            A valid datatype from gdalconst (Ex. gdalconst.GDT_Float32).
        resample_method: :func:`osgeo.gdalconst`, optional
            A valid resample method from gdalconst.
            Default is gdalconst.GRA_Average.
        as_gdal_grid: bool, optional
            Return as :func:`~GDALGrid`. Default is False.

    Returns
    -------
    None or :func:`gdal.Dataset` or :func:`~GDALGrid`
        If `to_file` is a :obj:`str`, then it returns None.
        Otherwise, if `to_file` is False then it returns a
        :func:`gdal.Dataset` unless `as_gdal_grid` is True.
        Then, it returns :func:`~GDALGrid`.

    """
    # Source of the data
    src, src_proj = load_raster(original_grid)

    # ensure output datatype is set
    if output_datatype is None:
        output_datatype = src.GetRasterBand(1).DataType

    # Grid to use to extract subset and match
    match_ds, match_proj = load_raster(match_grid)
    match_geotrans = match_ds.GetGeoTransform()

    if not to_file:
        # in memory raster
        dst_driver = gdal.GetDriverByName('MEM')
        dst_path = ""
    else:
        # geotiff
        dst_driver = gdal.GetDriverByName('GTiff')
        dst_path = to_file

    dst = dst_driver.Create(dst_path,
                            match_ds.RasterXSize,
                            match_ds.RasterYSize,
                            src.RasterCount,
                            output_datatype)

    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    for band_i in range(1, dst.RasterCount + 1):
        nodata_value = src.GetRasterBand(band_i).GetNoDataValue()
        if not nodata_value:
            nodata_value = -9999
        dst.GetRasterBand(band_i).SetNoDataValue(nodata_value)

    # extract subset and resample grid
    gdal.ReprojectImage(src, dst,
                        src_proj,
                        match_proj,
                        resample_method)

    if not to_file:
        if as_gdal_grid:
            return GDALGrid(dst)
        return dst
    del dst



def gdal_reproject(src,
                   dst=None,
                   src_srs=None,
                   dst_srs=None,
                   epsg=None,
                   error_threshold=0.125,
                   resampling=gdalconst.GRA_NearestNeighbour,
                   as_gdal_grid=False):
    """
    Reproject a raster image.

    Based on: https://github.com/OpenDataAnalytics/
            gaia/blob/master/gaia/geo/gdal_functions.py

    Parameters
    ----------
        src: :obj:`str` or :func:`gdal.Dataset` or :func:`~GDALGrid`
            The source image.
        dst: :obj:`str`, optional
            The filepath of the output image to write to.
        src_srs: :func:`osr.SpatialReference`, optional
            The source image projection.
        dst_srs: :func:`osr.SpatialReference`, optional
            The destination projection. If not provided,
            the code will use `epsg`.
        epsg: int, optional
            The EPSG code to reproject to. If not provided,
            the code will use `dst_srs`.
        error_threshold: float, optional
            Default is 0.125 (same as gdalwarp commandline).
        resampling: :func:`osgeo.gdalconst`
            Method to use for resampling. Default method is
            `gdalconst.GRA_NearestNeighbour`.
        as_gdal_grid: bool, optional
            Return as :func:`~GDALGrid`. Default is False.

    Returns
    -------
    :func:`gdal.Dataset` or :func:`~GDALGrid`
        By default, it returns `gdal.Dataset`.
        It will return :func:`~GDALGrid` if `as_gdal_grid` is True.
    """
    # Open source dataset
    src_ds = load_raster(src)[0]

    # Define target SRS
    if dst_srs is None:
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(int(epsg))

    dst_wkt = dst_srs.ExportToWkt()

    # Resampling might be passed as a string
    if not isinstance(resampling, int):
        resampling = getattr(gdal, resampling)

    src_wkt = None
    if src_srs is not None:
        src_wkt = src_srs.ExportToWkt()

    # Call AutoCreateWarpedVRT() to fetch default values
    # for target raster dimensions and geotransform
    reprojected_ds = gdal.AutoCreateWarpedVRT(src_ds,
                                              src_wkt,
                                              dst_wkt,
                                              resampling,
                                              error_threshold)

    # Create the final warped raster
    if dst:
        gdal.GetDriverByName('GTiff').CreateCopy(dst, reprojected_ds)
    if as_gdal_grid:
        return GDALGrid(reprojected_ds)
    return reprojected_ds

