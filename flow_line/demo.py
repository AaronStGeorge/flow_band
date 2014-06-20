from sr_flow_line import *
from dolfin import *

# Initial conditions
y0s = [(437634,-1.29813e6),\
       (433390,-1.29595e6),\
       (435283,-1.29619e6),\
       (433532,-1.29695e6),\
       (437081,-1.29799e6)]

"""
for i in range(len(y0s)):
  fl = FlowLineSr(y0s[i])
  fl.plot_fl()

  mesh = IntervalMesh(100, fl.minimum, fl.maximum)
  Q    = FunctionSpace(mesh, 'Lagrange', 1)

  plot(project(fl.spline_expression_sr_bed(element=Q.ufl_element()),Q))
  interactive()
"""
