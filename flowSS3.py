# This script produces the steady state surface profiles depicted on
# page 6 of "Beyond Back Stress: Model experiments on the stability of
# marine-terminating outlet glaciers" by Kees van der Veen.

# This version solves a system of three equations; dh/dx is solved for
# explicitly in order to impose dh/dx=0 at the divide.
# A longitudinal stress term (see equation 9.65 in "Fundamentals of Glacier
# Dynamics", Second Edition by C. J. van der Veen) is included in the force
# balance but it is modified from that in equation 9.65 to avoid singularities.

# The system of three 1-D ordinary differential equations is:
# 1. Conservation of mass (see equation 1 in "Beyond Back Stress: ..."):
#    d(HUW)/dx = MW
# 2. Force Balance (see equation 6 in "Beyond Back Stress: ..."):
#    - rho g H h' = tau_bx
#      - 2 B_s d/dx (H [((n+2)U/(n+1)W)^2 + (dU/dx)^2]^((1-n)/2n) dU/dx)
#      + B_s H/W ((n+2)U/2W)^(1/n)
#    where tau_bx is basal drag, given by:
#      tau_bx = mu A_s(H)^p at locations where h_b > 0
#      tau_bx = mu A_s(H + rho_w/rho h_b)^p U^(1/n)
#                 where h_b < 0 and the ice is grounded, h > |h_b|*(rho_w/rho-1)
#      tau_bx = 0 where h_b < 0 and the ice is floating, h < |h_b|*(rho_w/rho-1)
# 3. h' = dh/dx

# These equations are solved for surface elevation h (and the related ice 
# thickness H) and velocity u.  The velocity being an average over both depth
# and width.

# Boundary conditions at the divide are u(0)=0 and h'(0)=0.

# If 'i' is not in the command line arguments, only the grounded portion of the
# ice is modeled.

# The boundary condition enforced at the terminus is determined by command line
# arguments; 'd' is for Dirichlet, 'a' attempts to use a stress balance.
# Dirichlet conditions are h(L)=c, where L and c were taken from solutions
# obtained using Kees's FORTRAN program Flow3.FOR.

# If you want the ice to float, you must include the word 'conditional' or
# 'expression' as command line arguments.

# Examples:  python flowSS3.py d
#            python flowSS3.py i a conditional

# You can specify that the basal drag goes to zero near the terminus without
# causing the bottom of the ice to raise above the base topography with:

#            python flowSS3.py i a conditional grounded

# Last (?) modified by Glen Granzow on January 15, 2014.

from dolfin import *
from matplotlib import pyplot
from numpy import array
import sys

command_line_arguments = sys.argv

##############
# User input #
##############

try:
  curve = int(raw_input('Which curve is to be produced? (1 or 2): '))
except:
  curve = 1

###################
# Model constants #
###################

# Using the notation in the paper (not the book)

n = 3.0       # Flow law exponent
g = 9.8       # Gravitational constant
rho   = 0.917 # Density of ice
rho_w = 1.025 # Density of sea water

mu  = 1.0     # "a variable friction parameter"
A_s = 0.01    # "a sliding constant"  (B_s in the book)
p   = 1.0     # Exponent on effective pressure term (q in the book)
B_s = 540.0   # Flow law constant (B in the book)

half_width = "25000/(1 + 200*exp(0.05e-3*(x[0]-200e3))) + 5000"
W = Expression(half_width, cell=interval) # Half width

bed = "1000/(1 + 200*exp(0.10e-3*(x[0]-250e3))) - 950"
h_b = Expression(bed, cell=interval) # Bed elevation

# A useful expression for the effective pressure term:

x0 = 250.0e3 + 10.0e3*ln(1.0/3800.0) # x-coordinate where h_b = 0
reduction = Expression('x[0] <= x0? 0.0 : ratio*('+bed+')', x0=x0, ratio=rho_w/rho)

# Parameters specific to the two curves in "Beyond back stress: ..."

if curve == 1:
  M = Constant(0.3)   # Accumulation
  if 'i' in command_line_arguments:
    L = 498.00e3      # Domain size
    c =  78.46        # Surface elevation at the calving front
  else:
    L = 491.36e3      # Domain size
    c = 121.56        # Surface elevation at the grounding line
else:
#  M = Expression("5.3/(1 + 200*exp(0.05e-3*(x[0]-500e3))) - 5") # Accumulation
  M = Expression("5.3/(1 + 200*exp(0.05e-3*(x[0]-504e3))) - 5")
  if 'i' in command_line_arguments:
    L = 467.87        # Domain size
    c = 31.67         # Surface elevation at the calving front
  else:
    L = 446.04e3      # Domain size
    c = 115.90        # Surface elevation at the grounding line

if 'zb' in sys.argv: # set basal drag to zero near the terminus
  step_function = Expression('x[0] < L-10e3? 1.0 : 0.0', L=L)

################################
# Finite Element Approximation #
################################

order = 1  # Linear finite elements
try:       # Number of finite elements
  N = int(command_line_arguments[-1])
except:
  N = 100

# Define the function space

mesh = IntervalMesh(N,0,L)
V3   = VectorFunctionSpace(mesh, 'Lagrange', order, 3)

# Boundary conditions

class Divide(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], 0)

class Terminus(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], L)

boundary = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
ds = Measure('ds')[boundary]

INTERIOR, DIVIDE, TERMINUS = 99, 0, 1
boundary.set_all(INTERIOR)
Divide().mark(boundary, DIVIDE)
Terminus().mark(boundary, TERMINUS)

h,u,dhdx = V3.sub(0), V3.sub(1), V3.sub(2)

boundary_conditions = [DirichletBC(u,    Constant(0), boundary, DIVIDE),
                       DirichletBC(dhdx, Constant(0), boundary, DIVIDE)]

#if 'i' not in command_line_arguments:
if 'd' in command_line_arguments:
  boundary_conditions += [DirichletBC(h, Constant(c), boundary, TERMINUS)]
  
# Initial guess of the solution: h is the surface elevation

h    = Expression('a+b*x[0]', a=3000, b=-(3000-c)/L, cell=interval)
u    = Expression('x[0]/L', L=L/250)  # u is the velocity
dhdx = h.dx(0)                        # dhdx is the surface slope

hudhdx = project(as_vector((h,u,dhdx)), V3)
h,u,dhdx = split(hudhdx)

if 'g' in command_line_arguments:     # save the initial guess to plot later
  initial_h = zip(*map(hudhdx,mesh.coordinates()))[0]

if 'expression' in sys.argv:
  floating = Expression('x[0] > 491.36e3? 1.0 : 0.0', cell=interval)
  H = (1.0-floating)*(h-h_b) + floating*h/(1-rho/rho_w)
elif 'conditional' in sys.argv:
  floating = le(h,-h_b*(rho_w/rho-1))
  H = conditional(floating, h/(1-rho/rho_w), h-h_b) # H is the ice thickness
else:
  H = h-h_b

if 'grounded' in sys.argv:
  H = h - h_b

# Define the variational problem

testFunction  = TestFunction(V3)
trialFunction = TrialFunction(V3)
phi1, phi2, phi3 = split(testFunction)

# Streamline Upwind Petrov Galerkin method:

unorm = sqrt(dot(u, u) + 1e-10)
cellh = CellSize(mesh)

if 'u1' in sys.argv:
  phi1 = phi1 + cellh/(2*unorm)*dot(u, phi1.dx(0))
if 'u2' in sys.argv:
  phi2 = phi2 + cellh/(2*unorm)*dot(u, phi2.dx(0))
#  phi2 = phi2 + 0.5*cellh*phi2.dx(0) # * u/unorm
if 'u3' in sys.argv:
  phi3 = phi3 + cellh/(2*unorm)*dot(u, phi3.dx(0))
#  phi3 = phi3 + 0.5*cellh*phi3.dx(0) # * sign(u)

mass_conservation = ((H*u*W).dx(0) - M*W)*phi1
#mass_conservation = (H*u*W)*phi1.dx(0) + M*W*phi1

driving_stress = rho*g*H*dhdx
basal_drag     = mu*A_s*(H+reduction)**p*u**(1/n)

if 'expression' in sys.argv:
  basal_drag = (1.0-floating) * basal_drag
elif 'conditional' in sys.argv:
  basal_drag = conditional(floating, 0.0, basal_drag)

if 'zb' in sys.argv:
  basal_drag *= step_function # set basal drag to zero near the terminus

lateral_drag   = (B_s*H/W)*((n+2)*u/(2*W))**(1/n)
force_balance  = (driving_stress + basal_drag + lateral_drag) * phi2 

F = (mass_conservation + force_balance)*dx

invariant_squared = ((n+2.)/(n+1.)*u/W)**2+u.dx(0)**2
longitudinal_stress = 2*B_s*H*invariant_squared**((1-n)/(2*n))*u.dx(0)

if 'o' not in command_line_arguments:
  F += longitudinal_stress * phi2.dx(0) * dx
  if 'a' in command_line_arguments:
    dudx = (0.25*rho*g*(1-rho/rho_w)*H/B_s)**n
    invariant_squared_L = ((n+2.)/(n+1.)*u/W)**2+dudx**2
    F -= 2*B_s*H*invariant_squared_L**((1-n)/(2*n))*dudx * phi2 * ds(TERMINUS)
  elif 'p' in command_line_arguments:
    pass
  else:
    F -= longitudinal_stress * phi2 * ds(TERMINUS)

F += (h.dx(0)-dhdx)*phi3*dx

J  = derivative(F, hudhdx, trialFunction)
problem = NonlinearVariationalProblem(F, hudhdx, boundary_conditions, J)
solver  = NonlinearVariationalSolver(problem)

# To see a list of all parameters: info(solver.parameters, True)
solver_parameters = solver.parameters['newton_solver']
solver_parameters['maximum_iterations'] = 20
#solver_parameters['absolute_tolerance'] = 0.002
#solver_parameters['relative_tolerance'] = 1e-6

#########
# Solve #
#########

try:
  solver.solve()
except RuntimeError as message:
  print message
  end()
  response = raw_input('Press ENTER to continue ("q" to quit) ')
  if response == 'q': import sys; sys.exit()

###########################################
# Plot the surface elevation and velocity #
###########################################

V = FunctionSpace(mesh, 'Lagrange', order)

surface      = project(h,V)
bed          = project(h_b, V)
base         = project(h-H, V)
width        = project(W,V)
velocity     = project(u,V)

def plot_details():
  pyplot.xlim(0,500)
  pyplot.xticks(range(0,501,50))
  pyplot.grid(True)

pyplot.ion()
pyplot.figure(1, figsize=(6,6)).subplots_adjust(left=0.175, right=0.95)
pyplot.clf()
pyplot.subplot(211)
x = 0.001*mesh.coordinates()
pyplot.plot(x,map(surface,mesh.coordinates()),'b')
pyplot.plot(x,map(bed,mesh.coordinates()),'g')
pyplot.plot(x,map(base,mesh.coordinates()),'-+r')
pyplot.ylabel('Height above sea level (m)')
plot_details()

if 'g' in command_line_arguments:
  pyplot.plot(x,initial_h,'m')

print '\nElevation at the divide is %f, thickness is %f\n' % (surface(0), surface(0)-bed(0))

pyplot.subplot(212)
pyplot.plot(x,map(velocity,mesh.coordinates()))
pyplot.plot(x,array(map(width,mesh.coordinates()))/1000)
pyplot.xlabel('Distance from ice divide (km)')
pyplot.ylabel('Velocity (m/a)')
plot_details()

# Plot output from Kees's Flow3.FOR program

if 'c' in command_line_arguments:
  sys.path.append('/home/glen/python/')
  from readKees import readFile
  directory = '/home/glen/glen/python/fenics/kees/fortran/curve%s' % curve
  xx, yy = readFile(directory,'PROFS.DAT',(1,3))
  pyplot.subplot(211)
  pyplot.plot(xx,yy,'--r')
  xx, yy = readFile(directory,'TERMS.DAT',(1,2))
  pyplot.subplot(212)
  pyplot.plot(xx,yy,'--r')

################################################
# Plot the terms in the force balance equation #
################################################

pyplot.figure(2, figsize=(6,6))
pyplot.clf()

tau_d   = project(-driving_stress,V)
tau_b   = project(basal_drag,V)
tau_lat = project(lateral_drag,V)
tau_lon = project(-longitudinal_stress.dx(0),V)

if 'z' in command_line_arguments:
# Solve for the gradient in longitudinal stress instead of projecting
  dtaudx = TrialFunction(V)
  testFunction = TestFunction(V)
  tau_lon = Function(V)
  solve(dtaudx * testFunction * dx == longitudinal_stress*testFunction.dx(0) * dx, tau_lon, DirichletBC(V, Constant(0), 'near(x[0],0)'))

if 'M' in command_line_arguments:
  accumulation = project(M,V)
  pyplot.gcf().subplots_adjust(left=0.175, right=0.95)
  pyplot.subplot(211)
  pyplot.plot(x,map(accumulation,mesh.coordinates()))
  pyplot.ylabel('Surface mass balance (m/yr)')
  plot_details()
  pyplot.subplot(212)

pyplot.plot(x,map(tau_d,mesh.coordinates()),label=r'$\tau_d$')
pyplot.plot(x,map(tau_b,mesh.coordinates()),label=r'$\tau_b$')
pyplot.plot(x,map(tau_lat,mesh.coordinates()),label=r'$\tau_\perp$')
pyplot.plot(x,map(tau_lon,mesh.coordinates()),label=r'$\tau_-$')
pyplot.legend(loc='best').get_frame().set_alpha(0.5)
pyplot.xlabel('Distance from ice divide (km)')
plot_details()

# Plot the residual

if 'r' in command_line_arguments:
  if 'o' in command_line_arguments:
    pyplot.plot(x,map(lambda z: tau_b(z)+tau_lat(z), mesh.coordinates()), '--k')
    pyplot.plot(x,map(lambda z: tau_d(z)-tau_b(z)-tau_lat(z), mesh.coordinates()), '--')
  else:
    pyplot.plot(x,map(lambda z: tau_b(z)+tau_lat(z)+tau_lon(z), mesh.coordinates()), '--k')
    pyplot.plot(x,map(lambda z: -(tau_d(z)-tau_b(z)-tau_lat(z)-tau_lon(z)), mesh.coordinates()), '--')

# Plot the surface slope

if 's' in command_line_arguments:
  slope = project(dhdx,V)

  pyplot.figure(3, figsize=(6,6))
  pyplot.clf()
  pyplot.plot(x,map(slope,mesh.coordinates()),label='dh/dx')
  pyplot.grid(True)
  pyplot.legend(loc='best').get_frame().set_alpha(0.5)

# Plot terms involved in the terminus boundary condition

if 'l' in command_line_arguments:
  """
      RXX(IMAX)=(0.5*ROIG)*(H(IMAX)-(ROWI*((HBASE(IMAX)*HBASE(IMAX))/
     $  H(IMAX))))
    	RXX(IMAX)=RXX(IMAX)-SIGB

      EXX(IMAX)=((RXX(IMAX)/(2.*BFL))**EXPN) """

  strain_rate = project(u.dx(0),V)
  bc  = project((0.25*rho*g/B_s *(1-rho/rho_w) * H)**n,V)
#  bc2 = project((0.25*rho*g/B_s *(H - rho_w/rho* h_b*h_b/H))**n,V)

  pyplot.figure(4, figsize=(6,6))
  pyplot.clf()
  pyplot.plot(x,map(strain_rate,mesh.coordinates()),label='du/dx')
  pyplot.plot(x,map(bc, mesh.coordinates()),label='1')
#  pyplot.plot(x,map(bc2,mesh.coordinates()),'--',label='2')
  pyplot.grid(True)
  pyplot.legend(loc='best').get_frame().set_alpha(0.5)

try:
  __IPYTHON__
except:
  raw_input('Press ENTER to end the program')

