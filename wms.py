from owslib.wms import WebMapService

layer_names = [
  'Canada - 200m - MAG - Residual Total Field - Composante residuelle',
  'Canada 2 km - GRAV - Gravity Anomalies - Anomalies gravimétriques',
]

s_by_layer = {
  'Canada - 200m - MAG - Residual Total Field - Composante residuelle': 2048,
  'Canada 2 km - GRAV - Gravity Anomalies - Anomalies gravimétriques': 4096 
}

url = 'http://wms.agg.nrcan.gc.ca/wms2/wms2.aspx?request=GetCapabilities'
'''
http://wms.agg.nrcan.gc.ca/wms2/wms2.aspx?SERVICE=WMS&
REQUEST=GetMap&
VERSION=1.1.1&
SRS=EPSG:4326&
BBOX=-263.411205233539%2c62.6930840872226%2c-252.161020467078%2c73.9436146090283&
LAYERS=21&
STYLES=&
FORMAT=image%2fpng&
WIDTH=1024&
HEIGHT=1024&
TRANSPARENT=True

http://wms.agg.nrcan.gc.ca/wms2/wms2.aspx?SERVICE=WMS&
REQUEST=GetMap&
VERSION=1.1.1&
SRS=EPSG:4326&
BBOX=-263.411205233539%2c51.442553565417%2c-252.161020467078%2c62.6930840872226&
LAYERS=21&
STYLES=&
FORMAT=image%2fpng&
WIDTH=1024&
HEIGHT=1024&
TRANSPARENT=True

http://wms.agg.nrcan.gc.ca/wms2/wms2.aspx?SERVICE=WMS&
REQUEST=GetMap&
VERSION=1.1.1&
SRS=EPSG:4326&
BBOX=-263.411205233539%2c73.9436146090283%2c-252.161020467078%2c85.194145130834&
LAYERS=21&
STYLES=&
FORMAT=image%2fpng&
WIDTH=1024&
HEIGHT=1024&
TRANSPARENT=True

http://wms.agg.nrcan.gc.ca/wms2/wms2.aspx?SERVICE=WMS&
REQUEST=GetMap&
VERSION=1.1.1&
SRS=EPSG:4326&
BBOX=-252.161020467078%2c17.690962%2c-240.910835700617%2c28.9414925218057&
LAYERS=21&
STYLES=&
FORMAT=image%2fpng&
WIDTH=1024&
HEIGHT=1024&
TRANSPARENT=True


surfer:

SERVICE=WMS&
REQUEST=GetMap&
VERSION=1.1.1&
SRS=EPSG:4326&
BBOX=-157.156398226434%2C32.315631%2C-134.655104452868%2C54.8175179381999&
LAYERS=138&
STYLES=&
FORMAT=image%2Fpng&
WIDTH=1024&
HEIGHT=1024&
TRANSPARENT=True

python:

service=WMS&
version=1.1.1&
request=GetMap&
layers=1%2C3%2C8&
styles=&
width=8020&
height=2411&
srs=EPSG%3A4269&
bbox=-179.657692%2C32.315631%2C-3.426856%2C85.296148&
format=None&
transparent=FALSE&
bgcolor=0xFFFFFF&
exceptions=application%2Fvnd.ogc.se_xml
'''
wms = WebMapService(url)
contents = list(wms.contents)
for content in contents:
    layer = wms[content]
    if layer.title in layer_names:
      bbox = layer.boundingBox
      bbox84 = layer.boundingBoxWGS84
      fmts = wms.getOperationByName('GetMap').formatOptions
      print('content: %s' % content)
      print('title: %s' % layer.title)
      print('bbox: %s' % str(bbox))
      print('bbox84: %s' % str(bbox84))
      print('fmts: %s' % str(fmts))
      bbox = bbox[:-1]
      # from surfer
      bbox = (-263.411205233539,62.6930840872226,-252.161020467078,73.9436146090283)
      # original
      bbox = (-179.657692, 32.315631, -3.426856, 85.296148)
      # malartic
      # 48.1366° N, 78.1271° W
      bbox = (-80, 45, -75, 50)
      s = s_by_layer[layer.title]
      s = 4096 
      for s in (128, 256, 512, 1024, 2048, 4096):
        print('s: %s' % s)
        img = wms.getmap(
            layers=[str(content)],
            srs='EPSG:4269',
            bbox=bbox,
            size=(s,s),
            #format='image/png'
        )
        fname = 'out/wms_%s_%s_%s.png' % (bbox, s, layer.title.replace(' ', '-'))
        print('fname: %s' % fname)
        out = open(fname, 'wb')
        out.write(img.read())
        out.close()

