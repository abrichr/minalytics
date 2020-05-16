import sys
import pandas as pd
from osgeo import gdal


def rat_to_df(in_rat):
    """
    Given a GDAL raster attribute table, convert to a pandas DataFrame

    Parameters
    ----------
    in_rat : gdal.RasterAttributeTable
        The input raster attribute table

    Returns
    -------
    df : pd.DataFrame
        The output data frame
    """
    # Read in each column from the RAT and convert it to a series infering
    # data type automatically
    s = [pd.Series(in_rat.ReadAsArray(i), name=in_rat.GetNameOfCol(i))
         for i in range(in_rat.GetColumnCount())]

    # Concatenate all series together into a dataframe and return
    return pd.concat(s, axis=1)


def main(argv):
    # Open the raster and get a handle on the raster attribute table
    # Assume that we want the first band's RAT
    ds = gdal.Open(argv[0])
    rb = ds.GetRasterBand(1)
    rat = rb.GetDefaultRAT()

    # Convert the RAT to a pandas dataframe
    df = rat_to_df(rat)
    print(df)

    # Close the dataset
    ds = None


if __name__ == '__main__':
    main(sys.argv[1:])
