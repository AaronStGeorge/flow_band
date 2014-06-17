from tifffile import TiffFile
from dolfin import *
import inspect
import os
import sys

src_directory = '../../VarGlaS/src/'
sys.path.append(src_directory)

import utilities

def get_study_region():
  """
  return:
    vara - ditionary - contains projection information as well as bed data
    
  """
  filename = inspect.getframeinfo(inspect.currentframe()).filename
  home     = os.path.dirname(os.path.abspath(filename))
  
  sys.path.append(home + '/external_import_scripts')
  from tifffile import TiffFile
  
  direc    = home + '/study_region/elevation/ASTGTM2_S78E161_dem' 
  vara     = dict()
   
  # extents of domain :
  nx    =  1049
  ny    =  1031
  dx    =  17.994319205518387
  west  =  423863.131
  east  =  west  + nx*dx
  south =  -1304473.006
  north =  south + ny*dx
  
  # projection info :
  proj   = 'stere'
  lat_0  = '-90'
  lon_0  = '0'
  lat_ts = '-71'
  
  # retrieve data :
  data    = TiffFile(direc + '.tif')
  vara['b'] = {'map_data'          : data.asarray()[::-1, :],
             'map_western_edge'  : west,
             'map_eastern_edge'  : east,  
             'map_southern_edge' : south,
             'map_northern_edge' : north,
             'projection'        : proj,
             'standard lat'      : lat_0,
             'standard lon'      : lon_0,
             'lat true scale'    : lat_ts}
  return vara

# cread expression for bed
mesh = Mesh('study_region_mesh/study_region.xml')
sr   = utilities.DataInput(None,get_study_region(),mesh=mesh)
bed  = sr.get_spline_expression("b")

if __name__ == '__main__':

  V = FunctionSpace(mesh, 'Lagrange', 1)
  
  plot(project(bed,V))
  interactive()
