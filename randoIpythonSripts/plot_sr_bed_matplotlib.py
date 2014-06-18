from matplotlib import pyplot
import sys

src_directory = '../data'
sys.path.append(src_directory)

from sr_data import SrData

# starting points for flow line ode solver
y0s = [(437634,-1.29813e6),\
       (433390,-1.29595e6),\
       (435283,-1.29619e6),\
       (433532,-1.29695e6)]

sr = SrData()

# extents of domain :
nx    =  1049
ny    =  1031
dx    =  17.994319205518387
west  =  423863.131
east  =  west  + nx*dx
south =  -1304473.006
north =  south + ny*dx

#plot
fig = pyplot.gcf()
pyplot.imshow(sr.data['b'], 
              extent=[sr.x_min,sr.x_max,sr.y_min,sr.y_max], 
              origin='lower')
for y0 in y0s:
  fig.gca().add_artist(pyplot.Circle(y0,100,color='k'))
pyplot.show()
