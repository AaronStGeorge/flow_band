from dolfin import *
from matplotlib import pyplot
import numpy as np
from constants import *

def smooth_step_conditional(c, V, mesh):
  """
  diffuses conditional (or any other function for that matter) to create a 
  smoothed step function.

  :param c     : Expression conditional to be smoothed
  :param V     : Function space 
  :param mesh  : mesh that V is defined on
  """

  # set boundary condition to conditional
  class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
      return on_boundary
              
  boundary = Boundary()
  bc = DirichletBC(V, c, boundary)

  # set diffusivity parameter to average cell size
  cs = CellSize(mesh)
  cs = project(cs,V)
  m = cs.vector()*cs.vector()
  D = m.sum()/m.size()

  # solve the heat equation with dt=1 (therefore neglected)
  u = TrialFunction(V)
  v = TestFunction(V)
  a = u*v*dx + D*inner(nabla_grad(u), nabla_grad(v))*dx
  L = c*v*dx

  u = Function(V)
  solve(a == L, u, bc)

  return u

# Smoothly join a parabola and a line at xi.
def initial_h(h0,hL,xi,L):  
  if xi <= 0:
    return Expression('m*x[0]+b', m=(hL-h0)/L, b=h0, cell=interval)
  A = (h0-hL)/(xi-2*L)/xi
  m = 2*A*xi
  b = hL-m*L
  return Expression('x[0] < xi ? A*x[0]*x[0]+C : m*x[0]+b',
                     xi=xi, A=A, C=h0, m=m, b=b, cell=interval)

# Set up plotting window
def plot_details():
  pyplot.xlim(0,500)
  labels = '0, ,100, ,200, ,300, ,400, ,500'.split(',')
  pyplot.xticks(range(0,501,50), labels)
  pyplot.grid(True)

def update_mesh(n, L, mesh, h_u_dhdx, dt, maximum_L):
  
  new_L = min(L + float(dt)*h_u_dhdx(L)[1], maximum_L)
  h_ = np.array(map(h_u_dhdx, mesh.coordinates()))[:,0]
  new_mesh_coordinates = np.linspace(0, new_L, N+1)
  h_on_new_mesh = np.empty(N+1)

  for i in range(N+1):
    if new_mesh_coordinates[i] < L:
      h_on_new_mesh[indices[i]] = h_u_dhdx(new_mesh_coordinates[i])[0]
    else:
      h_on_new_mesh[indices[i]] = h_[-1]

  h_old.vector()[:] = h_on_new_mesh
  mesh.coordinates()[:,0] = new_mesh_coordinates

  if (dolfin.__version__[:3] == '1.3') or (dolfin.__version__[:3] == '1.4'):
    mesh.bounding_box_tree().build(mesh)
  else:
    mesh.intersection_operator().clear()
  
  return new_L
