from dolfin import Mesh
import inspect
import os
import sys

src_directory = '../../VarGlaS/src/'
sys.path.append(src_directory)

import utilities


class SrData(utilities.DataInput):
  """
  container class for data associated with Antarctic dry valleys study region
  """

  def __init__(self,mesh_flag='normal'):

    # get absolute directory of sr_data.py (this file)
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    self.home     = os.path.dirname(os.path.abspath(filename))
    
    # get mesh at specified resolution
    if mesh_flag == 'super_fine':
      self.mesh = Mesh(self.home + 
                      '/meshes/study_region_super_fine.xml')
    elif mesh_flag == 'medium':
      self.mesh = Mesh(self.home + 
                      '/meshes/study_region_medium.xml')
    else:
      self.mesh = Mesh(self.home + '/meshes/study_region.xml')

    # initialise all fields of parent class as members of this class
    super(SrData,self).__init__(None,self.get_study_region(), mesh=self.mesh)

    # expression for bed
    self.B = self.get_spline_expression('b')

  def get_study_region(self):
    """
    return:
      vara - dictionary - contains projection information as well as bed data
                         intended to be an input to utilites.DataInput
    """

    sys.path.append(self.home + '/external_import_scripts')
    from tifffile import TiffFile
    
    direc    = self.home + '/elevation_data/elevation/ASTGTM2_S78E161_dem' 
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
    vara['b'] = {'map_data'          : data.asarray(),
                 'map_western_edge'  : west,
                 'map_eastern_edge'  : east,  
                 'map_southern_edge' : south,
                 'map_northern_edge' : north,
                 'projection'        : proj,
                 'standard lat'      : lat_0,
                 'standard lon'      : lon_0,
                 'lat true scale'    : lat_ts}
    return vara

if __name__ == '__main__':
  from dolfin import *

  sr = SrData(mesh_flag="medium")

  V = FunctionSpace(sr.mesh, 'Lagrange', 1)
  
  plot(project(sr.B,V))
  interactive()
