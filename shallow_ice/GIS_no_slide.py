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

# Model parameters
dt     = 50.            # time step (seconds)
T      = 1000.          # Final time

def D_nl(H,B):
    return 2.*A*(rhoi*g)**n/(n+2.) * H**(n+2)  \
           * dot(grad(B+H),grad(B+H))**((n-1.)/2.)

# Create mesh and define function spaces
mesh = Mesh("Greenland_10km/GIS_mesh_10km.xml")

V = FunctionSpace(mesh, "Lagrange", 1)

# Load data fields:
smb = Function(V) # Surface mass balance
S_o = Function(V) # Surface elevation
H_o = Function(V) # Thickness
u_o = Function(V) # Speed observed

File("Greenland_10km/adot.xml") >> smb
File("Greenland_10km/S.xml")    >> S_o
File("Greenland_10km/H.xml")    >> H_o
File("Greenland_10km/vmag.xml") >> u_o

B = project(S_o - H_o,V)

# Boundary conditions
def boundary(x,on_boundary):
    return on_boundary
bcs = DirichletBC(V,Constant(0),boundary)

# Define trial and test functions
S  = Function(V)
H  = Function(V)
H_  = Function(V)  # previous solution
v = TestFunction(V)
dH = TrialFunction(V)

# Weak statement of the equations
F =     ( (H - H_) / dt * v \
          + D_nl(H,B) * dot(grad(S),grad(v))\
          - smb * v) * dx

J = derivative(F,H,dH)

problem = NonlinearVariationalProblem(F,H,bcs=bcs,J=J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'snes'
solver.parameters['snes_solver']['method']='virs'
solver.parameters['snes_solver']['maximum_iterations']=30
solver.parameters['snes_solver']['linear_solver']='mumps'
solver.parameters['snes_solver']['preconditioner'] = 'ilu'

# Output file
file_S = File("GIS_no_slide/S.pvd")
file_u = File("GIS_no_slide/U.pvd")
file_H = File("GIS_no_slide/H.pvd")
file_dS= File("GIS_no_slide/deltaS.pvd")

Sout = Function(V)
dSout = Function(V)
Uout = Function(V)
Hout = Function(V)

Sout.vector()[:] = S_o.vector()
Hout.vector()[:] = H_o.vector()
Uout.vector()[:] = u_o.vector()

file_S << Sout
file_H << Hout
file_u << Uout

# Step in time
t = 0.0

# bounds
lb = interpolate(Expression('0'),V)
ub = interpolate(Expression('1e4'),V)
H_.vector()[:] = H_o.vector()
S.vector()[:] = S_o.vector()

while (t < T):
    solver.solve(lb,ub)
    t += dt
    H_.vector()[:] = H.vector()
    S = project(B + H,V)
    dS = project(S-S_o,V)
    U = project(D_nl(H,B) * dot(grad(B+H),grad(B+H))**.5 / (H+.1),V)

    Sout.vector()[:]  = S.vector()
    dSout.vector()[:] = dS.vector()
    Hout.vector()[:]  = H.vector()
    Uout.vector()[:]  = U.vector()
    file_H << Hout
    file_S << Sout
    file_u << Uout
    file_dS << dSout
