from matplotlib import pyplot
from numpy import ma
from dolfin import *
import sys

src_directory = '../flow_line_study_region'
sys.path.append(src_directory)

from tifffile import TiffFile
from sr_data import mesh, utilities


# get data
data = TiffFile(
"../flow_line_study_region/study_region/elevation/ASTGTM2_S78E161_dem.tif")

# extents of domain :
nx    =  1049
ny    =  1031
dx    =  17.994319205518387
west  =  423863.131
east  =  west  + nx*dx
south =  -1304473.006
north =  south + ny*dx

vara['b'] = {'map_data'          : data.asarray(),
             'map_western_edge'  : west,
             'map_eastern_edge'  : east,  
             'map_southern_edge' : south,
             'map_northern_edge' : north,
             'projection'        : proj,
             'standard lat'      : lat_0,
             'standard lon'      : lon_0,
             'lat true scale'    : lat_ts}

# create expression for bed
sr   = utilities.DataInput(None,vara[b],mesh=mesh)
bed

#plot
pyplot.imshow(vara['b']['map_data'], extent=[west,east,south,north])
plot(project(bed,FunctionSpace(mesh, 'Lagrange', 1)))
pyplot.show()
Interactive()
