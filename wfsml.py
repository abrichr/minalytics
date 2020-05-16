import os
import requests
from collections import defaultdict
from owslib.wfs import WebFeatureService
from pprint import pprint, pformat

from minalytics import get_mine_data, get_lat_lon_cols

# http://www.nrcan.gc.ca/earth-sciences/geomatics/canadas-spatial-data-infrastructure/19359

wfs_urls = [
    'http://maps-cartes.ec.gc.ca/arcgis/services/NPRI_FGP_All_Layers/MapServer/WFSServer?SERVICE=WFS&REQUEST=GetCapabilities'
]

for wfs_url in wfs_urls:
  print('*' * 40)
  print('wfs_url: %s' % wfs_url)
  try:
    wfs = WebFeatureService(url=wfs_url)
  except Exception as e:
    print('exception: %s' % e)
    continue

  print('title: %s' % wfs.identification.title)
  ops = list(wfs.operations)
  for op in ops:
    print('op: %s' % op)
    pprint(vars(op))
  for feat_name, feat in wfs.contents.items():
    print('feat_name: %s' % feat_name)
    pprint(vars(feat))
  import ipdb; ipdb.set_trace()
