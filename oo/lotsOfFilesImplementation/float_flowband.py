# This script solves a time dependent problem leading to the steady state
# surface profiles depicted on page 6 of "Beyond Back Stress: Model experiments
# on the stability of marine-terminating outlet glaciers" by Kees van der Veen.
# This version includes a moving terminus.

# A system of three equations is solved.
# (dh/dx is solved for explicitly in order to impose dh/dx=0 at the divide.)
# A longitudinal stress term (see equation 9.65 in "Fundamentals of Glacier
# Dynamics", Second Edition by C. J. van der Veen) is included in the force
# balance but it is modified from that in equation 9.65 to avoid singularities.
# (Including 'o' as a command line argument omits the longitudinal stress term.)

# Time stepping uses the (implicit) backward Euler method.

# The system of three 1-D ordinary differential equations is:
# 1. Conservation of mass (see equation 1 in "Beyond Back Stress: ..."):
#    dH/dt = -(1/W) d(HUW)/dx + M
# 2. Force Balance (see equation 6 in "Beyond Back Stress: ..."):
#    - rho g H h' = mu A_s(H + rho_w/rho h_b)^p U^(1/n)
#      - 2 B_s d/dx (H [((n+2)U/(n+1)W)^2 + (dU/dx)^2]^((1-n)/2n) dU/dx)
#      + B_s H/W ((n+2)U/2W)^(1/n) where h_b < 0
#    and
#    - rho g H h' = mu A_s H^p
#      - 2 B_s d/dx (H [((n+2)U/(n+1)W)^2 + (dU/dx)^2]^((1-n)/2n) dU/dx)
#      + B_s H/W ((n+2)U/2W)^(1/n) where h_b > 0.
# 3. h' = dh/dx

# Boundary conditions are u(0)=0, h'(0)=0 and h(L)=c where c was taken from
# solutions obtained using Kees's FORTRAN program Flow3.FOR.

# Written by Glen Granzow on January 17, 2014.
# Modified on April 24, 2014 to work with FEniCS versions 1.4, 1.3, or 1.2.
#   Modified by Jesse Johnson

from dolfin import *
from matplotlib import pyplot
from numpy import array, linspace, empty
import sys
import numpy as np
from helper_functions import *
from constants import *

command_line_arguments = sys.argv

# globals
dt = Constant(150)
nSteps = 50
L = 100e3     # Initial length

# Half width
W = Expression("25000/(1 + 200*exp(0.05e-3*(x[0]-200e3)))+5000", cell=interval) 
bed = "1000/(1 + 200*exp(0.10e-3*(x[0]-250e3))) - 950"
bed = "40*sin(x[0]*(2*3.14159265358979323846/100e3)) + 50"

class Bed(Expression):
  def eval(self,value,x):
    if x < 300e3:
      value[0] = 40*sin(-x[0]*2*3.14159265358979323846/100e3) + 50
    else:
      value[0] = 1000/(1 + 200*exp(0.10e-3*(x[0]-400e3))) - 950

h_b = Expression(bed, cell=interval)  # Bed elevation
h_b = Bed(cell=interval)

####################### This is going to need to be changed ####################
#####################
# Initial Condition #
#####################

h = initial_h(120+0.0003*L, c, 0.4*L, L)
################################################################################

###########################################
# Define the Mesh and Mark its Boundaries #
###########################################

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
V  = FunctionSpace(mesh, 'Lagrange', order)
V3 = VectorFunctionSpace(mesh, 'Lagrange', order, 3)

boundary_conditions = [
  DirichletBC(V3.sub(0), Constant(c), boundary, TERMINUS), # h(L) = c
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

t_float = smooth_step_conditional(Floating(), V, mesh)

# A useful expression for the effective pressure term:
class Reduction(Expression):
  def eval(self,value,x):
    if h_b(x) > 0:
        value[0] = 0
    else:
        value[0] = (rho_w/rho) * h_b(x)
reduction = Reduction()

if 'floating' in command_line_arguments:
  H = t_float * (h/(1-rho/rho_w)) + (1-t_float) * (h-h_b)#H is the ice thickness
  
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
F += (h.dx(0)-dhdx)*phi3 * dx

J = derivative(F, h_u_dhdx, trialFunction)
problem = NonlinearVariationalProblem(F, h_u_dhdx, boundary_conditions, J)
solver  = NonlinearVariationalSolver(problem)

# To see a list of all parameters: info(solver.parameters, True)
solver_parameters = solver.parameters['newton_solver']
solver_parameters['maximum_iterations'] = 100
solver_parameters['absolute_tolerance'] = 1e-8
solver_parameters['relative_tolerance'] = 1e-8

# Create variables for plotting the topography
full_mesh = IntervalMesh(100, 0, 500e3)
full_x    = full_mesh.coordinates()
full_V    = FunctionSpace(full_mesh, 'Lagrange', 1)
full_bed  = map(project(h_b, full_V), full_x)
full_W    = array(map(project(W, full_V), full_x))/1000.0
full_x    = full_x * 0.001 # convert from meters to kilometers

########################################
# A function to update the moving mesh #
########################################

mesh_coordinates = interpolate(Expression("x[0]"), V).vector().array()
indices = []
for i in range(N+1):
  for j in range(N+1):
    if mesh_coordinates[i] == mesh.coordinates()[j,0]:
      indices.append(j)
      break
del(mesh_coordinates)

if len(indices) != N+1:
  print '\nERROR: Correspondence of indices of h.vector.array to\
          mesh.coordinates failed.'
  sys.exit()

#############
# Time Loop #
#############

for i in range(nSteps):
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

    ########
    # Plot #
    ########

    ### Figure 1 ###

    pyplot.ion()
    pyplot.figure(1, figsize=(6,6)).subplots_adjust(left=0.175, right=0.95)
    pyplot.clf()
    pyplot.subplot(311)

    surface  = project(h,V)
    bottom   = project(h-H,V)
    velocity = project(u,V)

    surface_ = map(surface, mesh.coordinates())+[bottom(L)]
    bottom_  = map(bottom, mesh.coordinates())
    x = 0.001*mesh.coordinates().flatten()
    # plot the glacier top surface
    pyplot.plot(list(x)+[x[-1]],surface_,'b',label="top") 
    # plot the glacier bottom surface
    pyplot.plot(x,bottom_,'r',label="bottom")
    # plot the basal topography
    pyplot.plot(full_x, full_bed,'g',label="bed")
    #pyplot.legend(loc=4,prop={'size':6})

    def water_plot(x):
        if x <= L:
            return surface(x) >= 0 or h_b(x) >= 0
        else:
            return h_b(x) >= 0

    water = map(lambda x: 0, full_x)
    mask  = map(water_plot, full_mesh.coordinates())
    water = np.ma.masked_array(water,mask)

    pyplot.plot(full_x, water, 'c') # plot the water surface

    #pyplot.plot((max(L,x0)/1000,500), (0,0), 'c') # plot the water surface

    # plot the height at which ice should float
    if 'f' in command_line_arguments: 
      full_float = map(project(-h_b*(rho_w/rho-1), full_V), 1000*full_x)
      pyplot.plot(full_x, full_float, '--k')

    pyplot.ylabel('Height above sea level (m)')
    pyplot.ylim(-1000,3100)
    plot_details()

    print '\nElevation at the divide is %f, thickness is %f\n'\
            % (surface(0), surface(0)-bottom(0))

    pyplot.subplot(312)
    pyplot.plot(x,map(velocity,mesh.coordinates()))
    pyplot.plot(full_x,  full_W, 'g')
    pyplot.plot(full_x, -full_W, 'g')
    pyplot.xlabel('Distance from ice divide (km)')
    pyplot.ylabel('Velocity (m/a)')
    pyplot.ylim(-50,300)
    plot_details()

    pyplot.subplot(313)
    pyplot.plot(x,map(t_float,mesh.coordinates()))
    pyplot.ylim(-.5,1.5)
    plot_details()

    """
    pyplot.subplot(414)
    pyplot.plot(x,map(project(H_t,V),mesh.coordinates()))
    pyplot.plot(x,map(project(H,V),mesh.coordinates()))
    #pyplot.ylim(-.5,1.5)
    pyplot.ylim(-100,2000)
    plot_details()
    """

    ### Figure 2 ###

    pyplot.figure(2, figsize=(6,6))
    pyplot.clf()

    tau_d   = project(-driving_stress,V)
    tau_b   = project(basal_drag,V)
    tau_lat = project(lateral_drag,V)
    tau_lon = project(-longitudinal_stress.dx(0),V)

    # Plot the surface mass balance

    if 'M' in command_line_arguments:
      accumulation = project(M,V)
      pyplot.subplot(211)
      pyplot.plot(x,map(accumulation,mesh.coordinates()))
      pyplot.ylabel('Surface mass balance (m/yr)')
      plot_details()
      pyplot.subplot(212)
      pyplot.gcf().subplots_adjust(left=0.175, right=0.95)

    # Plot terms in the force balance
    pyplot.plot(x,map(tau_d,mesh.coordinates()),label=r'$\tau_d$')
    pyplot.plot(x,map(tau_b,mesh.coordinates()),label=r'$\tau_b$')
    pyplot.plot(x,map(tau_lat,mesh.coordinates()),label=r'$\tau_\perp$')
    pyplot.plot(x,map(tau_lon,mesh.coordinates()),label=r'$\tau_-$')
    pyplot.ylim(-10,200)
    pyplot.legend(loc='upper right').get_frame().set_alpha(0.5)
    pyplot.xlabel('Distance from ice divide (km)')
    plot_details()

    # Plot the residual
    if 'r' in command_line_arguments:
      if 'o' in command_line_arguments:
        pyplot.plot(x,map(\
                lambda z: tau_b(z)+tau_lat(z), mesh.coordinates()), '--k')
        pyplot.plot(x,map(\
            lambda z: tau_d(z)-tau_b(z)-tau_lat(z), mesh.coordinates()), '--')
      else:
        pyplot.plot(x,map(\
        lambda z: tau_b(z)+tau_lat(z)+tau_lon(z), mesh.coordinates()), '--k')
        pyplot.plot(x,map(\
lambda z: tau_d(z)-tau_b(z)-tau_lat(z)-tau_lon(z), mesh.coordinates()), '--')

    ### Figure 3 ###  Plot the surface slope (only if requested)

    if 's' in command_line_arguments:
      slope = project(dhdx,V)
      pyplot.figure(3, figsize=(6,6))
      pyplot.clf()
      pyplot.plot(x,map(slope,mesh.coordinates()),label='dh/dx')
      pyplot.grid(True)
      pyplot.legend(loc='best').get_frame().set_alpha(0.5)

    # Terminate or pause if appropriate

    if 'i' in command_line_arguments: # Only the initial condition is plotted
      break

    print '\ntime = %f\n' % ((i+1)*float(dt))

    if 'p' in command_line_arguments: # Pause
      response = raw_input('Press ENTER (or enter "q" to quit, "l" to loop) ')
      if response == 'q': sys.exit()
      if response == 'l': command_line_arguments.remove('p')
    else:
      pyplot.draw()

    # Update the domain length and mesh coordinates
    L = update_mesh(n, L, mesh, h_u_dhdx, dt, maximum_L)
    #temp = project(Floating(),V)
    temp = smooth_step_conditional(Floating(), V, mesh)
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

