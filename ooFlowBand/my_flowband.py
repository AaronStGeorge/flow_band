import dolfin as dolf
import sys
import numpy as npy
from matplotlib import pyplot

dt = dolf.Constant(250)
nSteps = 80
L = 100e3     # Initial length

# Using the notation in the paper (not the book)

n = 3.0         # Flow law exponent
g = 9.8         # Gravitational constant
rho   = 0.917   # Density of ice
rho_w = 1.025   # Density of sea water

mu  = 1.0       # "a variable friction parameter"
A_s = 0.01      # "a sliding constant"  (B_s in the book)
p   = 1.0       # Exponent on effective pressure term (q in the book)
B_s = 540.0     # Flow law constant (B in the book)

# half width of glacier
W = dolf.Expression(\
        "25000/(1 + 200*exp(0.05e-3*(x[0]-200e3))) + 5000",cell=dolf.interval)

# Bed elevation
bed = "1000/(1 + 200*exp(0.10e-3*(x[0]-250e3))) - 950"
h_b = dolf.Expression(bed, cell=dolf.interval)  

####????########################################################################
# A useful expression for the effective pressure term:
x0 = 250.0e3 + 10.0e3*dolf.ln(1.0/3800.0) # x-coordinate where h_b = 0
reduction = dolf.Expression('x[0] <= x0? 0.0 : ratio*('+bed+')', x0=x0, ratio=rho_w/rho)
####????########################################################################

# Accumulation (curve 2 from "Beyond Back Stress")
M = dolf.Expression("5.3/(1 + 200*exp(0.05e-3*(x[0]-500e3))) - 5")   
maximum_L = 1e10     # Maximum length
c = 115.90      # Surface elevation at the calving front

#####################
# Initial Condition #
#####################

def initial_h(h0,hL,xi,L):  # Smoothly join a parabola and a line at xi.
  if xi <= 0:
    return dolf.Expression('m*x[0]+b', m=(hL-h0)/L, b=h0, cell=dolf.interval)
  A = (h0-hL)/(xi-2*L)/xi
  m = 2*A*xi
  b = hL-m*L
  return dolf.Expression('x[0] < xi ? A*x[0]*x[0]+C : m*x[0]+b', 
                     xi=xi, A=A, C=h0, m=m, b=b, cell=dolf.interval)

h = initial_h(120+0.0003*L, c, 0.4*L, L)

###########################################
# Define the Mesh and Mark its Boundaries #
###########################################

N    = 100  # Number of finite elements
mesh = dolf.IntervalMesh(N,0,L)

class Divide(dolf.SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and dolf.near(x[0], 0)

class Terminus(dolf.SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and dolf.near(x[0], L)

boundary = dolf.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
ds = dolf.Measure('ds')[boundary]

INTERIOR, DIVIDE, TERMINUS = 99, 0, 1
boundary.set_all(INTERIOR)
Divide().mark(boundary, DIVIDE)
Terminus().mark(boundary, TERMINUS)
#############################################################
# Set Up the Finite Element Approximation for the Time Loop #
#############################################################

order = 1  # Linear finite elements
V  = dolf.FunctionSpace(mesh, 'Lagrange', order)
V3 = dolf.VectorFunctionSpace(mesh, 'Lagrange', order, 3)

#############################################################
# Set Up the Finite Element Approximation for the Time Loop #
#############################################################

order = 1  # Linear finite elements
V  = dolf.FunctionSpace(mesh, 'Lagrange', order)
V3 = dolf.VectorFunctionSpace(mesh, 'Lagrange', order, 3)

boundary_conditions = [
  dolf.DirichletBC(V3.sub(0), dolf.Constant(c), boundary, TERMINUS), # h(L) = c
  dolf.DirichletBC(V3.sub(1), dolf.Constant(0), boundary, DIVIDE),   # u(0) = 0
  dolf.DirichletBC(V3.sub(2), dolf.Constant(0), boundary, DIVIDE)]   # dh/dx(0) = 0

h_old = dolf.project(h, V)
u = dolf.project(dolf.Expression('x[0]/L', L=1000*L), V) # initial guess of velocity
h_u_dhdx = dolf.project(dolf.as_vector((h,u,h.dx(0))), V3)
h,u,dhdx = dolf.split(h_u_dhdx)
H = h - h_b  # ice thickness = surface elevation - bed elevation

testFunction  = dolf.TestFunction(V3)
trialFunction = dolf.TrialFunction(V3)
phi1, phi2, phi3 = dolf.split(testFunction)

driving_stress = rho*g*H*dhdx
####????########################################################################
basal_drag     = mu*A_s*(H+reduction)**p*u**(1/n)
####????########################################################################
lateral_drag   = (B_s*H/W)*((n+2)*u/(2*W))**(1/n)

force_balance = (driving_stress + basal_drag + lateral_drag) * phi2 

####????########################################################################
mass_conservation = ((h-h_old)/dt + (H*u*W).dx(0)/W - M)*phi1
####????########################################################################

F = (mass_conservation + force_balance)*dolf.dx

###????#########################################################################
invariant_squared = ((n+2.)/(n+1.)*u/W)**2+u.dx(0)**2
longitudinal_stress = 2*B_s*H*invariant_squared**((1-n)/(2*n))*u.dx(0)

F += longitudinal_stress * phi2.dx(0) * dolf.dx
F -= longitudinal_stress * phi2 * ds(TERMINUS)
F += (h.dx(0)-dhdx)*phi3*dolf.dx
###????#########################################################################

J = dolf.derivative(F, h_u_dhdx, trialFunction)
problem = dolf.NonlinearVariationalProblem(F, h_u_dhdx, boundary_conditions, J)
solver  = dolf.NonlinearVariationalSolver(problem)

# To see a list of all parameters: info(solver.parameters, True)
solver_parameters = solver.parameters['newton_solver']
solver_parameters['maximum_iterations'] = 20
solver_parameters['absolute_tolerance'] = 1e-8
solver_parameters['relative_tolerance'] = 1e-8

# Create variables for plotting the topography
full_mesh = dolf.IntervalMesh(100, 0, 500e3)
full_x    = full_mesh.coordinates()
full_V    = dolf.FunctionSpace(full_mesh, 'Lagrange', 1)
full_bed  = map(dolf.project(h_b, full_V), full_x)
full_W    = npy.array(map(dolf.project(W, full_V), full_x))/1000.0
full_x    = full_x * 0.001 # convert from meters to kilometers

def plot_details():
  pyplot.xlim(0,500)
  labels = '0, ,100, ,200, ,300, ,400, ,500'.split(',')
  pyplot.xticks(range(0,501,50), labels)
  pyplot.grid(True)

########################################
# A function to update the moving mesh #
########################################

mesh_coordinates = dolf.interpolate(dolf.Expression("x[0]"), V).vector().array()
indices = []
for i in range(N+1):
  for j in range(N+1):
    if mesh_coordinates[i] == mesh.coordinates()[j,0]:
      indices.append(j)
      break
del(mesh_coordinates)

if len(indices) != N+1:
  print '\nERROR: Correspondence of indices of h.vector.array to mesh.coordinates failed.'
  sys.exit()

def update_mesh(n,L):
  new_L = min(L + float(dt)*h_u_dhdx(L)[1], maximum_L)
  h_ = npy.array(map(h_u_dhdx, mesh.coordinates()))[:,0]
  new_mesh_coordinates = npy.linspace(0, new_L, N+1)
  h_on_new_mesh = npy.empty(n+1)

  for i in range(n+1):
    if new_mesh_coordinates[i] < L:
      h_on_new_mesh[indices[i]] = h_u_dhdx(new_mesh_coordinates[i])[0]
    else:
      h_on_new_mesh[indices[i]] = h_[-1]

  h_old.vector()[:] = h_on_new_mesh
  mesh.coordinates()[:,0] = new_mesh_coordinates

  if dolf.dolfin.__version__[:3] == '1.3':
    mesh.bounding_box_tree().build(mesh)
  else:
    mesh.intersection_operator().clear()

  return new_L
  
#############
# Time Loop #
#############
  
for i in range(nSteps):
  try:

    #########
    # Solve #
    #########

    try:
      solver.solve()
    except RuntimeError as message:
      print message
      dolf.end()
      response = raw_input('Press ENTER to continue ("q" to quit) ')
      if response == 'q': sys.exit()

  ########
  # Plot #
  ########

  ### Figure 1 ###

    pyplot.ion()
    pyplot.figure(1, figsize=(6,6)).subplots_adjust(left=0.175, right=0.95)
    pyplot.clf()
    pyplot.subplot(211)

    surface  = dolf.project(h,V)
    bottom   = dolf.project(h-H,V)
    velocity = dolf.project(u,V)

    surface_ = map(surface, mesh.coordinates())+[bottom(L)]
    bottom_  = map(bottom, mesh.coordinates())
    x = 0.001*mesh.coordinates().flatten()
    pyplot.plot(list(x)+[x[-1]],surface_,'b') # plot the glacier top surface
    pyplot.plot(x,bottom_,'b')                # plot the glacier bottom surface
    pyplot.plot(full_x, full_bed, 'g')        # plot the basal topography
    pyplot.plot((max(L,x0)/1000,500), (0,0), 'c') # plot the water surface

    pyplot.ylabel('Height above sea level (m)')
    pyplot.ylim(-1000,3100)
    plot_details()

    print '\nElevation at the divide is %f, thickness is %f\n' % (surface(0),\
            surface(0)-bottom(0))

    pyplot.subplot(212)
    pyplot.plot(x,map(velocity,mesh.coordinates()))
    pyplot.plot(full_x,  full_W, 'g')
    pyplot.plot(full_x, -full_W, 'g')
    pyplot.xlabel('Distance from ice divide (km)')
    pyplot.ylabel('Velocity (m/a)')
    pyplot.ylim(-50,300)
    plot_details()

  ### Figure 2 ###

    pyplot.figure(2, figsize=(6,6))
    pyplot.clf()

    tau_d   = dolf.project(-driving_stress,V)
    tau_b   = dolf.project(basal_drag,V)
    tau_lat = dolf.project(lateral_drag,V)
    tau_lon = dolf.project(-longitudinal_stress.dx(0),V)

  # Plot terms in the force balance

    pyplot.plot(x,map(tau_d,mesh.coordinates()),label=r'$\tau_d$')
    pyplot.plot(x,map(tau_b,mesh.coordinates()),label=r'$\tau_b$')
    pyplot.plot(x,map(tau_lat,mesh.coordinates()),label=r'$\tau_\perp$')
    pyplot.plot(x,map(tau_lon,mesh.coordinates()),label=r'$\tau_-$')
    pyplot.ylim(-10,200)
    pyplot.legend(loc='upper right').get_frame().set_alpha(0.5)
    pyplot.xlabel('Distance from ice divide (km)')
    plot_details()

  ### Figure 3 ###  Plot the surface slope (only if requested)


  # Terminate or pause if appropriate

    print '\ntime = %f\n' % ((i+1)*float(dt))

    pyplot.draw()

  # Update the domain length and mesh coordinates

    L = update_mesh(N,L)

  except KeyboardInterrupt:
    print '\n##############################################################'
    print   '### Control C was pressed (this can have some bad effects) ###'
    print   '##############################################################\n'

try:
  sys.__IPYTHON__
except:
  raw_input('Press ENTER to end the program')
