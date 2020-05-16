# example result of wms request:
# http://wms.agg.nrcan.gc.ca/wms2/wms2.aspx?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&SRS=EPSG:4326&BBOX=-134.655104452868%2C54.8175179381999%2C-112.153810679302%2C77.3194048763998&LAYERS=138&STYLES=&FORMAT=image%2Fpng&WIDTH=1024&HEIGHT=1024&TRANSPARENT=True

import os
import requests
from collections import defaultdict
from owslib.wms import WebMapService
from pprint import pprint, pformat

from minalytics import get_mine_data, get_lat_lon_cols

wms_url_versions = {
    # takes forever to parse
    # TODO: let this run overnight
    #'http://wms.agg.nrcan.gc.ca/wms/wms.aspx?request=GetCapabilities': '1.3.0',
    'http://wms.agg.nrcan.gc.ca/wms2/wms2.aspx?request=GetCapabilities': '1.1.1',
}

layer_titles = [
  'Canada Grid_ 1km - DTM for Canada - Grille_ MNT du Canada',
  'Densities and Magnetic Susceptibilities – Densités et susceptibilités magnétiques - GDB',
  'Canada - 200m - MAG - Residual Total Field - Composante residuelle',
  'Canada - 200m - MAG - 1st Vertical Derivative - Dérivée 1ère verticale',
  'Gravity-Magnetics-Bathymetry GDB – Gravité-magnétiques-bathymétrie GDB',
  'Gravity - GDB Point Data - GDB Données ponctuelles',
  'Canada 2 km - GRAV - Gravity Anomalies - Anomalies gravimétriques',
  'Canada 2 km - GRAV - Horizontal Gradient - Gradient horizontal',
  'Canada 2 km - GRAV - 1st Vertical Derivative - Dérivée 1ère verticale',
  'Canada 2 km - GRAV - Bouguer - Bouguer',
  'Canada 2 km - GRAV - Isostatic Residual - Isostatiques résiduelles',
  'Canada 2 km - GRAV - Free Air - Air libre',
  'Canada 2 km - GRAV - Observed - Observée',
  'Acoustic properties (Vp, Vs, density) - GDB - Propriétés acoustiques (Vp, Vs, densité) - GDB',
]

def get_wms(wms_url, wms_version):
  xml_fname = 'xml/%s.xml' % (
      wms_url.replace("://", "_").replace(".", "_").replace("/", "_"))
  print('wms_url: \t%s' % wms_url)
  print('xml_fname: \t%s' % xml_fname)
  try:
    xml = open(xml_fname, 'rb').read()
    #from lxml import etree
    #xml = etree.parse(xml_fname)
    print('Reading xml from disk...')
  except Exception as e:
    print('Exception while reading xml from disk: %s' % e)
    print('Downloading xml...')
    xml = requests.get(wms_url).text
    open(xml_fname, 'w').write(xml)
    print('saved xml to disk')

  if wms_version == '1.3.0':
    # workaround for case sensitivity bug in owslib/map/wms130.py
    xml = xml.replace(str.encode('crs='), str.encode('CRS='))
    # workaround for huge_tree bug in owslib/map/common.py
    def readString(self, st):
      from owslib.util import strip_bom
      from owslib.etree import etree
      import sys
      sys.setrecursionlimit(2**31-1)
      raw_text = strip_bom(st)
      p = etree.XMLParser(huge_tree=True)
      return etree.fromstring(raw_text, parser=p)
    from owslib.map.common import WMSCapabilitiesReader
    WMSCapabilitiesReader.readString = readString

  wms = WebMapService(wms_url, wms_version, xml=xml)
  return wms

for wms_url, wms_version in wms_url_versions.items():
  wms = get_wms(wms_url, wms_version)

contents = list(wms.contents)

SUMMARIZE = False
if SUMMARIZE:
  op_names = [op.name for op in wms.operations]
  for op_name in op_names:
    print('op_name: %s' % op_name)
    op = wms.getOperationByName(op_name)
    pprint(vars(op))

  fmts = wms.getOperationByName('GetMap').formatOptions

  layer_by_id = {}
  vals_by_attr = defaultdict(set)
  area_tups = []

  for layer_id in contents:
    layer = wms[layer_id]

    for key, val in vars(layer).items():
      if type(val) is list:
        val = tuple(sorted(val))
      elif type(val) is dict:
        val = str(val)
      vals_by_attr[key].add(val)

    title = layer.title
    bbox = layer.boundingBoxWGS84
    index = layer.index
    _id = layer.id
    name = layer.name

    min_lon, min_lat, max_lon, max_lat = bbox
    width = max_lon - min_lon
    height = max_lat - min_lat
    area = width * height
    area_tups.append((area, title, bbox))

    layer_by_id[layer_id] = {
      'title': title,
      'bbox': bbox,
      'index': index,
      'id': _id,
      'name': name,
      'area': area
    }

  valcount_by_attr = {k: len(v) for k, v in vals_by_attr.items()}
  print('valcount_by_attr: \n%s' % pformat(valcount_by_attr))

  area_tups.sort(key=lambda tup: tup[0])
  print('area_tups: \n%s' % pformat(area_tups))
  for area, title, bbox in area_tups:
    if area > 8000 and area < 64000:
      print(title)

#import ipdb; ipdb.set_trace()

SAVE = True
if SAVE:
  deg_pad = 0.1
  df, cols_by_type, target_cols = get_mine_data()
  lat_col, lon_col = get_lat_lon_cols(df.columns)
  for i, row in df.iterrows():
    lat = row[lat_col]
    lon = row[lon_col]
    for layer_id in contents:
      layer = wms[layer_id]
      title = layer.title
      if title not in layer_titles:
        continue
      bbox = layer.boundingBoxWGS84
      min_lon, min_lat, max_lon, max_lat = bbox
      if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
        img_box = (lon - deg_pad, lat - deg_pad, lon + deg_pad, lat + deg_pad)
        for d in (256,):
          fname_bylonlat = 'out/wms/bylonlat/%s_%s_%s_%s_%s.png' % (
              lon, lat, deg_pad, d, layer.title.replace(' ', '-'))
          fname_bylayer = 'out/wms/bylayer/%s_%s_%s_%s_%s.png' % (
              layer.title.replace(' ', '-'), lon, lat, deg_pad, d)
          print('fname_bylonlat: %s' % fname_bylonlat)
          print('fname_bylayer: %s' % fname_bylayer)
          if os.path.exists(fname_bylonlat) and os.path.exists(fname_bylayer):
            print('file exists')
            continue
          img = wms.getmap(
              layers=[str(layer_id)],
              srs='EPSG:4269',
              bbox=img_box,
              size=(d, d),
              #format='image/png'
          )
          img_bytes = img.read()
          for fname in (fname_bylonlat, fname_bylayer):
            out = open(fname, 'wb')
            out.write(img_bytes)
            out.close()
