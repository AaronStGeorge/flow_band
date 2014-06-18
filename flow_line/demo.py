from sr_flow_line import spline_expression_sr_bed
from dolfin import *

# Initial conditions
y0s = [(437634,-1.29813e6),\
       (433390,-1.29595e6),\
       (435283,-1.29619e6),\
       (433532,-1.29695e6)]

for i in range(len(y0s)):
  fl = spline_expression_sr_bed(y0s[i])
  fl.plot_fl()

  mesh = IntervalMesh(100, min(fl.s), max(fl.s))
  Q    = FunctionSpace(mesh, 'Lagrange', 1)

  plot(project(fl,Q))
  interactive()
