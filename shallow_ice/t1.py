# Shallow Ice approximation flow band model try 1
from dolfin import *
from matplotlib import pyplot as plt
from numpy import mod

#==== Parameters ===============================================================

# Physical constants
A    = 1.0e-16      # Flow law parameter (Pa^-3 a^-1)
g    = 9.81         # Acceleration due to gravity (m/s**2)
rhoi = 910          # Density of ice kg/m^3
n    = 3.           # Glen's flow law exponent
spy  = 31556926     # Seconds per year
smb  = Constant(.3) # Surface mass balance
smb  = Expression('(-.5/1e6)*x[0] + .5')

# Model parameters
dt     = 500.      # time step (seconds)
T      = 50000      # Final time
lm     = 1500e3     # Length mesh - (meters)
ie     = 100 # Extent of ice - (meters) 
h0     = 10         # h(0) initial height at x=0 - (meters)
hie    = 0          # h(ie) initial height at x=ie - (meters)
nfe    = 100        # number of finite elements - (int) 

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

#smb = initial_h(.5,0,750000,1e6)

class Ploting:

  def __init__(self, L):
    self.full_mesh = IntervalMesh(nfe, 0, 2*L)
    self.full_V    = FunctionSpace(self.full_mesh, 'Lagrange', 1)
    self.full_x    = self.full_mesh.coordinates()

  def plot_profile(self, L, B, h, mesh, V):
    """
    Plot glacier profile at current t
    """
    pyplot.ion()
    pyplot.figure()
    pyplot.clf()

    x   = mesh.coordinates()
    bed = map(project(B, self.full_V), self.full_x)
    ice = map(project(h, V), x)

    plt.plot(self.full_x, bed)
    plt.plot(x, ice)
    pyplot.draw()

def plot_profile(mesh, V, H):
  """
  Plot glacier profile at current t
  """
  plt.ion()
  plt.clf()
  plt.ylim(-100,5500)

  x   = mesh.coordinates()
  #bed = map(project(B, V), x)
  ice = map(project(H, V), x)

  #plt.plot(x, bed)
  plt.plot(x, ice)
  plt.draw()

#==== Model ====================================================================

# Diffusivity non-linear taken from Huybrecht's equation 3
def D_nl(H):
    return 2.*A*(rhoi*g)**n/(n+2.) * H**(n+2)  \
           * inner(nabla_grad(H),nabla_grad(H))**((n-1.)/2.)

# Create mesh and define function spaces
mesh = IntervalMesh(nfe, 0, lm)
V    = FunctionSpace(mesh, "Lagrange", 1)

# Boundary conditions
def terminus(x, on_boundary):
    return on_boundary and near(x[0], lm)

bcs = DirichletBC(V, Constant(0), terminus)

# Define trial and test functions
H   = Function(V)
H_  = Function(V)  # previous solution
v   = TestFunction(V)
dH  = TrialFunction(V)

# Weak statement of the equations
F =     ( (H - H_) / dt * v \
          + D_nl(H) * dot(grad(H),grad(v)) \
          - smb * v) * dx

J = derivative(F,H,dH)


problem = NonlinearVariationalProblem(F,H,bcs=bcs,J=J)
solver  = NonlinearVariationalSolver(problem)

solver.parameters['nonlinear_solver'] = 'snes'
solver.parameters['snes_solver']['method']='vinewtonrsls'
solver.parameters['snes_solver']['maximum_iterations']=20
solver.parameters['snes_solver']['linear_solver']='mumps'
solver.parameters['snes_solver']['preconditioner'] = 'ilu'

# Step in time
t = 0.0

# bounds
lb = interpolate(Expression('0'),V)
ub = interpolate(Expression('1e4'),V)

while (t < T):
    t += dt
    H_.vector()[:] = H.vector()
    solver.solve(lb,ub)
    plot_profile(mesh, V, H)
