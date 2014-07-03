# Shallow Ice approximation flow band model try 1
from dolfin import *
from matplotlib import pyplot as plt
from numpy import mod

#==== Parameters ===============================================================

# Physical constants
A    = 1.0e-16    # Flow law parameter (Pa^-3 a^-1)
g    = 9.81       # Acceleration due to gravity (m/s**2)
rhoi = 910        # Density of ice kg/m^3
n    = 3.         # Glen's flow law exponent
spy  = 31556926   # Seconds per year

smb  = Expression('(-.5/1e6)*x[0] + .5')  # Surface mass balance

# Model parameters
dt   = 500.    # Time step (seconds)
T    = 50000   # Final time
ilm  = 1500e3  # Initial length of mesh - (meters)
nfe  = 100     # Number of finite elements - (int) 

#==== Helper functions =========================================================

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
mesh = IntervalMesh(nfe, 0, ilm)
V    = FunctionSpace(mesh, "Lagrange", 1)

# Boundary conditions
def terminus(x, on_boundary):
    return on_boundary and near(x[0], ilm)

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
