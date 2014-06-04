# coding: utf-8
from dolfin import *
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
mesh = IntervalMesh(100,0,100)
v = FunctionSpace(mesh, 'Lagrange', 1)
x = Expression("x[0]")

bed = Expression('sin(x[0]/10)')
c = conditional(le(bed,0),0,1)

class MyExpression1(Expression):
  def eval(self, value, x):
      if bed(x[0]) < 0:
          value[0] = 1
      else:
          value[0] = 0

u = project(MyExpression1(),v)

mc = mesh.coordinates()
ma = u.vector().array()

plt.figure(1)
plt.subplot(211)
plt.plot(mc,ma)

plt.subplot(212)
plt.plot(mc,gaussian_filter(ma,2))
plt.show()

#plot(u)
#plot(project(bed,v))
#interactive()
