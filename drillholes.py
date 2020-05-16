from pprint import pprint
import os.path

import numpy as np
import pandas as pd
import pyproj
import rasterio
import utm

#log = rasterio.logging.getLogger()
#log.setLevel(rasterio.logging.DEBUG)
import ipdb; ipdb.set_trace()
rasterio.rio.main.configure_logging(99999)

DIRPATH_CSV = './drillholes/ODHD_Sept2018/Access_Database/'

FNAME_DRILLHOLE_CSV = 'ODHD_Sept2018 - Drill_hole.csv'
FNAME_ELEMENT_CSV = 'ODHD_Sept2018 - Element.csv'
FNAME_DRILLHOLE_ELEMENT_CSV = 'ODHD_Sept2018 - Drill_hole_element.csv'

FPATH_DRILLHOLE_CSV = os.path.join(DIRPATH_CSV, FNAME_DRILLHOLE_CSV)
FPATH_ELEMENT_CSV = os.path.join(DIRPATH_CSV, FNAME_ELEMENT_CSV)
FPATH_DRILLHOLE_ELEMENT_CSV = os.path.join(DIRPATH_CSV, FNAME_DRILLHOLE_ELEMENT_CSV)

# Path to binary magnetic grid file (NAD83)
# http://gdr.agg.nrcan.gc.ca/gdrdap/dap/search-eng.php?tree-0=Magnetic-Radiometric-EM+-+Magn%C3%A9tiques-Radioactivit%C3%A9-%C3%89M&tree-1=Compilations+-+Compilations&tree-2=Canada+-+200m+-+MAG&tree-3=Click+here+for+more+options&datatype-ddl=&layer_name=&submit_search=Submit+Search#results
FPATH_MAGNETIC_GRID = './data/d950396/Canada - 200m - MAG - Residual Total Field - Composante residuelle.grd.gxf'

TYPE_BY_COLUMN = {
    # identifiers
    'AFRI_ID': str,
    'Drill Hole ID': int,
    'Element ID': str,
    'ID__drillhole_element': int,

    # numeric
    'Azimuth': float,
    'Dip': float,
    'Length': float,
    'Overburden': float,

    # to be converted
    'Description': str,  # ore density -> element (categorical), min, max (float)
    'Type': str,  # drill type -> categorical

    # geographical
    'UTM Datum': str,
    'UTM Zone': float,
    'UTM Easting': float,
    'UTM Northing': float,
    'Lat Dec': float,
    'Lng Dec': float,

    # source info 
    'Company Hole ID': str,
    'Company Name': str,
    'Comments': str,
    'Map Source': str,
    'Property Name': str,
    'RGP District': str,
    'Township or Area': str,
    'Year Drilled': int,
}


def parse_ore_density_description(df):
    column = 'Description'
    values = df[column]

    def parse_density_of_element(x):
        if isinstance(x, str):
            return x.split()[2]
        else:
            return None

    def parse_min_density_pct(x):
        return {
            np.nan: None,
            'Presence Of Gold: At Least 3000 ppb': \
                    3000 / 10**9 * 100,
            'Presence Of Gold: Between 500 and 3000 ppb': \
                    500 / 10**9 * 100,
            'Presence Of Zinc: At Least 0.25%': \
                    0.25,
            'Presence Of Copper: At Least 0.1%': \
                    0.1,
            'Presence Of Silver: At Least 35 grams per ton': \
                    35 / 10**6 * 100,
            'Presence Of Lead: At Least 1.0%': \
                    1.0,
            'Presence Of Nickel: At Least 0.1%': \
                    0.1,
            'Presence Of Platinum Group Elements: At Least 500 ppb': \
                    500 / 10**9 * 100
        }[x]

    df['Density of Element'] = df['Description'].apply(parse_density_of_element)
    df['Min Density Percent'] = df['Description'].apply(parse_min_density_pct)


def get_df_drillhole(print_unique_values_by_column=False):
    df_drillhole = pd.read_csv(FPATH_DRILLHOLE_CSV)
    df_element = pd.read_csv(FPATH_ELEMENT_CSV)
    df_drillhole_element = pd.read_csv(FPATH_DRILLHOLE_ELEMENT_CSV)

    df = df_drillhole.merge(
        df_drillhole_element,
        left_on='ID',
        right_on='Drill Hole ID',
        how='outer',
        suffixes=('__drillhole', '__drillhole_element')
    ).merge(
        df_element,
        left_on='Element ID',
        right_on='ID',
        how='outer',
        suffixes=('__drillhole_element', '__element')
    )

    if print_unique_values_by_column:
        unique_values_by_column = {}
        for column in df.columns:
            values = df[column]
            unique = values.unique()
            unique_values_by_column[column] = unique[:20]
        pprint(unique_values_by_column)

    parse_ore_density_description(df)

    return df

def latlon2pixel(lat, lon, ds_feature):
    # XXX

    # from https://github.com/mapbox/rasterio/issues/1446
    from rasterio import crs, warp
    print('-' * 40)
    print('rasterio')
    print('-' * 40)
    src_crs = crs.CRS.from_epsg(4326)
    dst_crs_from_ds = ds_feature.crs
    dst_crs_from_string = crs.CRS.from_string((
        '+ellps=GRS80 '
        '+lat_0=63 +lat_1=49 +lat_2=77 +lon_0=-92 '
        '+no_defs '
        '+proj=lcc '
        # this is what differentiates this from dst_crs_from_ds
        #'+towgs84=0,0,0,0,0,0,0 '
        '+units=m '
        '+x_0=0 +y_0=0 '
    ))
    print('src_crs: {}'.format(src_crs))
    for dst_crs in (dst_crs_from_ds, dst_crs_from_string):
        xs = [lat]
        ys = [lon]
        x, y = warp.transform(
            src_crs, dst_crs, xs, ys
        )
        print(' -' * 40)
        print('dst_crs: {}'.format(dst_crs))
        print('lon, lat: {}, {}'.format(lon, lat))
        print('x, y: {}, {}'.format(x, y))
    '''
    lon, lat: -89.90664875288734, 52.447658919586274
    x, y: [5178339726.087983], [4366321884.802004]
    '''

    # from gdal_grid
    from affine import Affine
    from osgeo import gdal, gdalconst, ogr, osr
    print('-'* 40)
    print('gdalgrid')
    print('-'* 40)

    def coord2pixel(x_coord, y_coord, dataset):
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
        affine = Affine.from_gdal(*dataset.GetGeoTransform())
        col, row = ~affine * (x_coord, y_coord)
        if col > dataset.RasterXSize or col < 0:
            raise IndexError("Longitude {0} is out of bounds ..."
                             .format(x_coord))
        if row > dataset.RasterYSize or row < 0:
            raise IndexError("Latitude {0} is out of bounds ..."
                             .format(y_coord))

        return int(col), int(row)

    dataset = gdal.Open(FPATH_MAGNETIC_GRID, gdalconst.GA_ReadOnly)
    print('dataset: {}'.format(dataset))
    projection = osr.SpatialReference()
    projection.ImportFromWkt(dataset.GetProjection())
    print('projection: {}'.format(projection))

    sp_ref = osr.SpatialReference()
    sp_ref.ImportFromEPSG(4326)
    print('sp_ref: {}'.format(sp_ref))
    transx = osr.CoordinateTransformation(sp_ref, projection)
    print('transx: {}'.format(transx))
    x_coord, y_coord = transx.TransformPoint(lon, lat)[:2]
    print('x_coord, y_coord: {}, {}'.format(x_coord, y_coord))
    col, row = coord2pixel(x_coord, y_coord, dataset)
    print('col, row: {}, {}'.format(col, row))

    import ipdb; ipdb.set_trace()

    return col, row 


def get_feature_patches(ds_feature, df_drillhole):

    feature_crs = ds_feature.crs
    feature_proj = feature_crs['proj']
    feature_lat_0 = feature_crs['lat_0']
    feature_lat_1 = feature_crs['lat_1']
    feature_lat_2 = feature_crs['lat_2']
    feature_lon_0 = feature_crs['lon_0']
    feature_x_0 = feature_crs['x_0']
    feature_y_0 = feature_crs['y_0']
    feature_ellps = feature_crs['ellps']
    feature_to_wgs84 = feature_crs['towgs84']
    feature_units = feature_crs['units']
    feature_no_defs = feature_crs['no_defs']
    feature_transform = ds_feature.transform

    print((
        'Magnetic Dataset:\n'
        '*****************\n'
        'feature_proj:     {}\n'
        'feature_lat_0:    {}\n'
        'feature_lat_1:    {}\n'
        'feature_lat_2:    {}\n'
        'feature_lon_0:    {}\n'
        'feature_x_0:      {}\n'
        'feature_y_0:      {}\n'
        'feature_ellps:    {}\n'
        'feature_to_wgs84: {}\n'
        'feature_units:    {}\n'
        'feature_no_defs:  {}\n'
        'feature_transform:\n{}\n'
    ).format(
        feature_proj,
        feature_lat_0,
        feature_lat_1,
        feature_lat_2,
        feature_lon_0,
        feature_x_0,
        feature_y_0,
        feature_ellps,
        feature_to_wgs84,
        feature_units,
        feature_no_defs,
        feature_transform
    ))


    def get_feature_patch_for_drillhole(row, ds_feature, lat_lon_delta_tups, print_drillholes=False):
        #print('row: {}'.format(row))

        drillhole_utm_datum = row['UTM Datum']
        drillhole_utm_zone = row['UTM Zone']
        drillhole_utm_easting = row['UTM Easting']
        drillhole_utm_northing = row['UTM Northing']
        drillhole_lat_dec = row['Lat Dec']
        drillhole_lon_dec = row['Lng Dec']

        if print_drillholes:
            print((
                'Drillhole:\n'
                '**********\n'
                'drillhole_utm_datum:    {}\n'
                'drillhole_utm_zone:     {}\n'
                'drillhole_utm_easting:  {}\n'
                'drillhole_utm_northing: {}\n'
                'drillhole_lat_dec:      {}\n'
                'drillhole_lon_dec:      {}'
            ).format(
                drillhole_utm_datum,
                drillhole_utm_zone,
                drillhole_utm_easting,
                drillhole_utm_northing,
                drillhole_lat_dec,
                drillhole_lon_dec
            ))

        # conversion with utm library
        if np.isnan(drillhole_utm_easting) or np.isnan(drillhole_utm_northing):
            return

        lat, lon = utm.to_latlon(drillhole_utm_easting, drillhole_utm_northing, drillhole_utm_zone, northern=True)

        # compare against known latitude/longitude values
        if (not np.isnan(drillhole_lat_dec) and
                not np.isnan(drillhole_lon_dec) and
                drillhole_lat_dec != 0 and
                drillhole_lon_dec != 0):

            d_lat = abs(drillhole_lat_dec - lat)
            d_lon = abs(drillhole_lon_dec - lon)
            distance = deg2m(drillhole_lat_dec, drillhole_lon_dec, lat, lon)
            lat_lon_delta_tups.append((
                distance,
                (d_lat, d_lon),
                (drillhole_lat_dec, lat),
                (drillhole_lon_dec, lon)
            ))

        # convert latitude/longitude to pixel
        col, row = latlon2pixel(lat, lon, ds_feature)

        

        '''
        from_easting = drillhole_utm_easting
        from_northing = drillhole_utm_northing
        from_proj = 'utm'
        from_datum = drillhole_utm_datum
        from_ellps = drillhole_utm_datum
        from_zone = int(drillhole_utm_zone)

        to_proj = feature_proj
        to_lat_0 = feature_lat_0
        to_lat_1 = feature_lat_1
        to_lat_2 = feature_lat_2
        to_lon_0 = feature_lon_0
        to_x_0 = feature_x_0
        to_y_0 = feature_y_0
        to_ellps = feature_ellps
        to_no_defs = feature_no_defs

        from_proj_string = (
            '+proj={from_proj} '
            '+datum={from_datum} '
            '+zone={from_zone} '
        ).format(
            from_proj=from_proj,
            from_datum=from_datum,
            from_zone=from_zone,
        )
        to_proj_string = (
            '+ellps={to_ellps} '
            '+lat_0={to_lat_0} '
            '+lat_1={to_lat_1} '
            '+lat_2={to_lat_2} '
            '+lon_0={to_lon_0} '
            '{no_defs} '
            '+proj={to_proj} '
            '+x_0={to_x_0} '
            '+y_0={to_y_0} '
        ).format(
            to_ellps=to_ellps,
            to_lat_0=to_lat_0,
            to_lat_1=to_lat_1,
            to_lat_2=to_lat_2,
            to_lon_0=to_lon_0,
            no_defs='+no_defs' if to_no_defs else '',
            to_proj=to_proj,
            to_x_0=to_x_0,
            to_y_0=to_y_0,
        )

        to_proj_strings = [
            to_proj_string,
            feature_crs.to_string(),
            to_proj_string + '+towgs84=0,0,0,0,0,0,0'
        ]

        proj_from = pyproj.Proj(from_proj_string)
        from_pt = (from_easting, from_northing)
        print('from_proj_string: {}'.format(from_proj_string))
        print('from_pt: {}'.format(from_pt))
        for to_proj_string in to_proj_strings:
            print('*' * 40)
            print('to_proj_string: {}'.format(to_proj_string))
            proj_to = pyproj.Proj(to_proj_string)
            to_pt = pyproj.transform(proj_from, proj_to, *from_pt)
            print('to_pt: {}'.format(to_pt))
            to_easting, to_northing = to_pt
            dx = feature_transform[0]
            dy = feature_transform[4]
            ix = to_easting / dx
            iy = to_northing / dy
            print('ix: {}, iy: {}'.format(ix, iy))
        '''

    lat_lon_delta_tups = []
    feature_patches = []
    for _, row in df_drillhole.iterrows():
        feature_patch = get_feature_patch_for_drillhole(row, ds_feature, lat_lon_delta_tups)
        feature_patches.append(feature_patch)
    lat_lon_delta_tups.sort()
    largest_delta_tup = lat_lon_delta_tups[-1]
    largest_distance = largest_delta_tup[0]
    print('largest discrepancy: {.5f}km'.format(largest_distance))
    import ipdb; ipdb.set_trace()
    return feature_patches


def deg2m(lat1, lon1, lat2, lon2):
    from math import sin, cos, sqrt, atan2, radians

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def main():

    ds_feature = rasterio.open(FPATH_MAGNETIC_GRID)
    df_drillhole = get_df_drillhole()
    feature_patches = get_feature_patches(ds_feature, df_drillhole)


if __name__ == '__main__':
    main()
