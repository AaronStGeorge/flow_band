"""
This program solves the shallow ice equation.
Written by: Jesse Johnson
"""

from dolfin import *
from numpy import mod

# Physical constants
A    = 1.0e-16    # Flow law parameter (Pa^-3 a^-1)
g    = 9.81       # Acceleration due to gravity (m/s**2)
rhoi = 910        # Density of ice kg/m^3
n    = 3.         # Glen's flow law exponent
spy  = 31556926   # Seconds per year
smb  = Expression('fmin(.5,s * (Rel - sqrt(pow(x[0]-xs,2) + pow(x[1] - ys,2))))'\
                  , s = 1e-5, Rel = 450e3, xs = 750e3,ys=750e3)
#smb  = Constant(.3)

#
# Model parameters
dt     = 5000.        # time step (seconds)
T      = 50000       # Final time

# Diffusivity non-linear taken from Huybrecht's equation 3
def D_nl(H):
    return 2.*A*(rhoi*g)**n/(n+2.) * H**(n+2)  \
           * inner(nabla_grad(H),nabla_grad(H))**((n-1.)/2.)


# Create mesh and define function spaces
mesh = RectangleMesh(0,0,1500e3,1500e3,31, 31)
V = FunctionSpace(mesh, "Lagrange", 1)
# Boundary conditions
def boundary(x,on_boundary):
    return on_boundary
bcs = DirichletBC(V,Constant(0),boundary)

# Define trial and test functions
H   = Function(V)
H_  = Function(V)  # previous solution
v   = TestFunction(V)
dH  = TrialFunction(V)

# Weak statement of the equations
F =     ( (H - H_) / dt * v \
          + D_nl(H) * dot(grad(H),grad(v))\
          - smb * v) * dx

J = derivative(F,H,dH)


problem = NonlinearVariationalProblem(F,H,bcs=bcs,J=J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'snes'
solver.parameters['snes_solver']['method']='vinewtonrsls'
solver.parameters['snes_solver']['maximum_iterations']=20
solver.parameters['snes_solver']['linear_solver']='mumps'
solver.parameters['snes_solver']['preconditioner'] = 'ilu'

# Output file
file = File("results/H.pvd")

# Step in time
t = 0.0

# bounds
lb = interpolate(Expression('0'),V)
ub = interpolate(Expression('1e4'),V)

while (t < T):
    t += dt
    H_.vector()[:] = H.vector()
    solver.solve(lb,ub)
    file << H
