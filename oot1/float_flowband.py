from dolfin import *
from matplotlib import pyplot
from numpy import array, linspace, empty
import sys
import numpy as np
from helper_functions import *

src_directory = '../flow_line'
sys.path.append(src_directory)

from demo import *


command_line_arguments = sys.argv

curve = 1 
dt = Constant(150)
nSteps = 50

# Using the notation in the paper (not the book)
n     = 3.0   # Flow law exponent
g     = 9.8   # Gravitational constant
rho   = 0.917 # Density of ice
rho_w = 1.025 # Density of sea water

mu  = 1.0     # "a variable friction parameter"
A_s = 0.01    # "a sliding constant"  (B_s in the book)
p   = 1.0     # Exponent on effective pressure term (q in the book)
B_s = 540.0   # Flow law constant (B in the book)


# Half width
W = Expression("25000/(1 + 200*exp(0.05e-3*(x[0]-200e3))) + 5000",cell=interval) 
bed = "1000/(1 + 200*exp(0.10e-3*(x[0]-250e3))) - 950"
bed = "40*sin(x[0]*(2*3.14159265358979323846/100e3)) + 50"

class Bed(Expression):
    def eval(self,value,x):
        if x < 300e3:
            value[0] = 40*sin(-x[0]*2*3.14159265358979323846/100e3) + 50
        else:
            value[0] = 1000/(1 + 200*exp(0.10e-3*(x[0]-400e3))) - 950

#h_b = Expression(bed, cell=interval)  # Bed elevation
#h_b = Bed(cell=interval)
flow_line = FlowLineSr(y0s[4])

# Parameters specific to the two curves in "Beyond back stress: ..."
if curve == 1:
  M = Constant(0.3)    # Accumulation
  if 'shelf' in command_line_arguments:
    maximum_L = 498.00e3 # Final length
    c =  78.46           # Surface elevation at the calving front
  else:
    maximum_L = 491.36e3 # Final length (distance to the grounding line)
    c = 121.56           # Surface elevation at the terminus
else:
  M = Expression("5.3/(1 + 200*exp(0.05e-3*(x[0]-500e3))) - 5") # Accumulation
  maximum_L = 1e10     # Maximum length
  if 'shelf' in command_line_arguments:
    c = 31.67          # Surface elevation at the calving front
  else:
    c = 115.90         # Surface elevation at the terminus

maximum_L = flow_line.maximum  # Maximum length
L = .2*maximum_L               # Initial length


###########################################
# Define the Mesh and Mark its Boundaries #
###########################################

N    = 100  # Number of finite elements
mesh = IntervalMesh(N,0,L)

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

#############################################################
# Set Up the Finite Element Approximation for the Time Loop #
#############################################################

order = 1  # Linear finite elements
V     = FunctionSpace(mesh, 'Lagrange', order)
V3    = VectorFunctionSpace(mesh, 'Lagrange', order, 3)

h_b = flow_line.spline_expression_sr_bed(V.ufl_element())

#####################
# Initial Condition #
#####################

h = initial_h(h_b(L)+200, h_b(L)+100, 0.4*L, L)

helper_functions = HelperFunctions(mesh, V, h_b, W, maximum_L)

class BC(Expression):
  def eval(self,value,x):
    value[0] = h_b(L) + 100

boundary_conditions = [
  DirichletBC(V3.sub(0), BC(), boundary, TERMINUS), # h(L) = c
  DirichletBC(V3.sub(1), Constant(0), boundary, DIVIDE),   # u(0) = 0
  DirichletBC(V3.sub(2), Constant(0), boundary, DIVIDE)]   # dh/dx(0) = 0

h_old = project(h, V)
u = project(Expression('x[0]/L', L=1000*L), V) # initial guess of velocity
h_u_dhdx = project(as_vector((h,u,h.dx(0))), V3)
h,u,dhdx = split(h_u_dhdx)

class Floating(Expression):
  def eval(self,value,x):
    if h_u_dhdx(x[0])[0] <= -h_b(x[0])*(rho_w/rho-1):
      value[0] = 1
    else:
      value[0] = 0

t_float = helper_functions.smooth_step_conditional(Floating())

# A useful expression for the effective pressure term:
class Reduction(Expression):
  def eval(self,value,x):
    if h_b(x) > 0:
        value[0] = 0
    else:
        value[0] = (rho_w/rho) * h_b(x)
reduction = Reduction()
reduction = Constant(0)

if 'floating' in command_line_arguments:
  # H is the ice thickness
  H = t_float * (h/(1-rho/rho_w)) + (1-t_float) * (h-h_b) 
    
else:
  H = h - h_b

testFunction  = TestFunction(V3)
trialFunction = TrialFunction(V3)
phi1, phi2, phi3 = split(testFunction)

mass_conservation = ((h-h_old)/dt + (H*u*W).dx(0)/W - M)*phi1

driving_stress = rho*g*H*dhdx
basal_drag     = mu*A_s*(H+reduction)**p*u**(1/n)
lateral_drag   = (B_s*H/W)*((n+2)*u/(2*W))**(1/n)

if 'floating' in command_line_arguments:
  basal_drag = (1-t_float) * basal_drag

force_balance = (driving_stress + basal_drag + lateral_drag) * phi2

F = (mass_conservation + force_balance)*dx

invariant_squared = ((n+2.)/(n+1.)*u/W)**2+u.dx(0)**2
longitudinal_stress = 2*B_s*H*invariant_squared**((1-n)/(2*n))*u.dx(0)

F += longitudinal_stress * phi2.dx(0) * dx
F -= longitudinal_stress * phi2 * ds(TERMINUS)
F += (h.dx(0)-dhdx)*phi3*dx

J = derivative(F, h_u_dhdx, trialFunction)
problem = NonlinearVariationalProblem(F, h_u_dhdx, boundary_conditions, J)
solver  = NonlinearVariationalSolver(problem)

# To see a list of all parameters: info(solver.parameters, True)
solver_parameters = solver.parameters['newton_solver']
solver_parameters['maximum_iterations'] = 100
solver_parameters['absolute_tolerance'] = 1e-8
solver_parameters['relative_tolerance'] = 1e-8

#############
# Time Loop #
#############

#for i in range(nSteps):
for i in range(10):
  try:

    #########
    # Solve #
    #########

    if 'i' not in command_line_arguments:
      try:
        solver.solve()
      except RuntimeError as message:
        print message
        end()
        response = raw_input('Press ENTER to continue ("q" to quit) ')
        if response == 'q': sys.exit()

    helper_functions.mtpl_plot(h, H, u, L, h_b, 
                               driving_stress, 
                               basal_drag, 
                               lateral_drag, 
                               longitudinal_stress,
                               t_float,
                               maximum_L)

    # Update the domain length and mesh coordinates
    L    = helper_functions.update_mesh(L,maximum_L,mesh,dt,h_u_dhdx,h_old)
    temp = helper_functions.smooth_step_conditional(Floating())
    t_float.vector().set_local(temp.vector().array())
    
  except KeyboardInterrupt:
    print '\n##############################################################'
    print   '### Control C was pressed (this can have some bad effects) ###'
    print   '##############################################################\n'
    command_line_arguments.append('p')

try:
  __IPYTHON__
except:
  raw_input('Press ENTER to end the program')
