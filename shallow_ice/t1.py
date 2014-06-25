# Shallow Ice approximation flow band model try 1
from dolfin import *
from matplotlib import pyplot as plt


#==== Parameters ===============================================================

L   = 100      # initial length - (meters)
h0  = 10      # h(0) initial height at 0 - (meters)
hL  = 0        # h(L) initial height at L - (meters)
nfe = 100      # number of finite elements - (int) 

#==== Helper functions =========================================================

# Smoothly join a parabola and a line at xi.
def initial_h(h0,hL,xi,L):  
  if xi <= 0:
    return Expression('m*x[0]+b', m=(hL-h0)/L, b=h0, cell=interval)
  A = (h0-hL)/(xi-2*L)/xi
  m = 2*A*xi
  b = hL-m*L
  return Expression('x[0] < xi ? A*x[0]*x[0]+C : m*x[0]+b',
                     xi=xi, A=A, C=h0, m=m, b=b, cell=interval)

# Plot glacier profile at current t
class Ploting:

  def __init__(self, L):
    self.full_mesh = IntervalMesh(nfe, 0, 2*L)
    self.full_V    = FunctionSpace(self.full_mesh, 'Lagrange', 1)
    self.full_x    = self.full_mesh.coordinates()

  def plot_profile(self, L, B, h, mesh, V):
    x   = mesh.coordinates()
    bed = map(project(B, self.full_V), self.full_x)
    ice = map(project(h, V), x)

    plt.plot(self.full_x, bed)
    plt.plot(x, ice)
    plt.show()

ploting = Ploting(L)

#==== Mesh =====================================================================

mesh = IntervalMesh(nfe, 0, L)
V    = FunctionSpace(mesh, 'Lagrange', 1)

#==== Initial & boundary conditions ============================================

# Bed elevation
B = Constant(5)

# Initial profile
h = initial_h(B(0)+h0 ,B(L)+hL, .25*L, L)

ploting.plot_profile(L, B, h, mesh, V)
